"""Train error-aware SEMamba++.

torchrun --nproc_per_node=N -m src.se.error_aware_se_mamba_pp.train --run_dir DIR [--config JSON]
"""

import src.se.common.cuda_local as _cuda_local  # noqa: F401  # isort: skip

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.asr.parakeet_tdt_0_6b_v2 import ParakeetTDT06BV2
from src.asr.timestamp_cache import crop_batch_token_ids, load_timestamp_cache, missing_timestamp_keys
from src.data.corpora.librispeech import utterance_key
from src.data.schema import AsrTimestamp
from src.se.common.pesq import pesq_sum
from src.se.common.stft import Spec, mag_pha_istft, mag_pha_stft
from src.se.common.training import (
    all_reduce_sum,
    init_distributed,
    load_checkpoint,
    load_config,
    log_scalars,
    seed_everything,
    start_run,
    unpadded,
)
from src.se.error_aware_se_mamba_pp.alignment_cache import (
    AlignmentStore,
    assert_row_matches,
    build_fingerprint,
    require_cache_file,
    state_dict_sha256,
    verify_cache_file,
)
from src.se.error_aware_se_mamba_pp.asr_guidance import ASR_HOP, ENCODER_LAYER
from src.se.error_aware_se_mamba_pp.dataset import (
    ScheduledNoiseDataset,
    build_scheduled_loaders,
    content_lengths,
    datasets_from_cfg,
)
from src.se.error_aware_se_mamba_pp.discriminator import DiscriminatorOutputs, SEMambaPPDiscriminator
from src.se.error_aware_se_mamba_pp.error_mask import sample_intervals_from_frames
from src.se.error_aware_se_mamba_pp.loss import (
    MultiScaleMelSpectrogramLoss,
    discriminator_loss,
    error_aware_generator_loss,
    generator_loss,
)
from src.se.error_aware_se_mamba_pp.model import SEMambaPP, require_asr_guidance_dim

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.json"
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
torch.backends.cudnn.benchmark = True


def parse_args(argv: list[str] | None = None) -> tuple[SimpleNamespace, Path, Path | None, int | None]:
    """Read --config / --run_dir / --alignment-cache / --max-steps.

    A saved config.json is a resume: --config and --max-steps are rejected.
    --alignment-cache is required when error_aware is enabled, and rejected when it is not.
    The cache path and max_steps are not written into config.json.
    """
    parser = argparse.ArgumentParser(description="Train error-aware SEMamba++")
    parser.add_argument("--config", default=None, help=f"default: {DEFAULT_CONFIG}")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--alignment-cache", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    saved = run_dir / "config.json"
    if saved.is_file():
        if args.config is not None or args.max_steps is not None:
            parser.error(f"{saved} exists (resume); do not pass --config or --max-steps")
        cfg = load_config(saved)
        max_steps = None
    else:
        if args.max_steps is not None and args.max_steps < 1:
            parser.error("--max-steps must be >= 1")
        cfg = load_config(args.config or DEFAULT_CONFIG)
        max_steps = None if args.max_steps is None else int(args.max_steps)
    enabled = bool(cfg.error_aware.enabled)
    cache = None if args.alignment_cache is None else Path(args.alignment_cache)
    if enabled and cache is None:
        parser.error("--alignment-cache is required when error_aware.enabled is true")
    if not enabled and cache is not None:
        parser.error("--alignment-cache is not used when error_aware.enabled is false")
    if enabled:
        alpha = float(cfg.error_aware.alpha)
        if alpha < 0:
            parser.error("error_aware.alpha must be >= 0")
    variants = cfg.data.variants_per_utterance
    if variants is not None and int(variants) < 1:
        parser.error("data.variants_per_utterance must be >= 1 or null")
    return cfg, run_dir, cache, max_steps


def _guidance_features(
    asr: ParakeetTDT06BV2,
    audio: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = asr.encode(audio, lengths, layers=(ENCODER_LAYER,))
    return encoded.layer_outputs[ENCODER_LAYER].detach(), encoded.encoded_length


def _tdt_targets_from_cache(
    cache: dict[str, AsrTimestamp],
    utterance_keys: Sequence[str],
    crop_starts: torch.Tensor,
    crop_ends: torch.Tensor,
) -> tuple[list[list[int]], int]:
    token_ids = crop_batch_token_ids(cache, utterance_keys, crop_starts.tolist(), crop_ends.tolist())
    return token_ids, sum(len(ids) == 0 for ids in token_ids)


def _require_timestamp_coverage(cache: dict[str, AsrTimestamp], dataset: ScheduledNoiseDataset, name: str) -> None:
    keys = [utterance_key(utterance) for utterance in dataset.utterances]
    missing = missing_timestamp_keys(cache, keys)
    if missing:
        shown = ", ".join(missing[:5])
        raise KeyError(f"timestamp cache missing {len(missing)} {name} keys, e.g. {shown}")


def _forward_generator(
    generator: nn.Module,
    noisy_mag: torch.Tensor,
    noisy_pha: torch.Tensor,
    asr_hidden: torch.Tensor,
    asr_lengths: torch.Tensor,
) -> Spec:
    return generator(noisy_mag, noisy_pha, asr_hidden, asr_lengths)


def batch_sample_error(
    store: AlignmentStore,
    dataset: ScheduledNoiseDataset,
    split: str,
    epoch: int,
    indices: torch.Tensor,
    keys: tuple[str, ...],
    crop_starts: torch.Tensor,
    crop_ends: torch.Tensor,
    waveform_length: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    variant = dataset.variant_for_epoch(epoch)
    error = torch.zeros(indices.numel(), waveform_length, dtype=torch.bool)
    fractions: list[float] = []
    for batch_index, utterance_index in enumerate(indices.tolist()):
        row = store.get(split, int(utterance_index), variant)
        spec = dataset.spec_for(int(utterance_index), variant)
        assert_row_matches(row, spec, keys[batch_index])
        content = int(crop_ends[batch_index] - crop_starts[batch_index])
        if content != row.content_samples:
            raise RuntimeError(f"content length {content} != cache {row.content_samples} for {keys[batch_index]}")
        for start, end in sample_intervals_from_frames(row.error_frames, content, ASR_HOP):
            error[batch_index, start:end] = True
        fractions.append((float(error[batch_index, :content].sum()) / content) if content else 0.0)
    fraction = sum(fractions) / len(fractions) if fractions else 0.0
    return error, content_lengths(crop_starts, crop_ends), fraction


def _generator_losses(
    clean: Spec,
    gen: Spec,
    gen_hat: Spec,
    disc_out: DiscriminatorOutputs,
    tdt: torch.Tensor,
    weights: SimpleNamespace,
    stft: SimpleNamespace,
    mel_loss: MultiScaleMelSpectrogramLoss,
    clean_audio: torch.Tensor,
    enhanced_audio: torch.Tensor,
    enabled: bool,
    alpha: float,
    sample_error: torch.Tensor | None,
    content_length: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if not enabled:
        mel = mel_loss(clean_audio.unsqueeze(1), enhanced_audio.unsqueeze(1))
        return generator_loss(clean, gen, gen_hat, disc_out, mel, tdt, weights, stft.n_fft)
    if sample_error is None or content_length is None:
        raise RuntimeError("error-aware weighting requires a sample error mask")
    return error_aware_generator_loss(
        clean,
        gen,
        gen_hat,
        disc_out,
        tdt,
        weights,
        stft.n_fft,
        mel_loss,
        clean_audio,
        enhanced_audio,
        sample_error,
        content_length,
        stft.hop_size,
        alpha,
    )


def _write_hashes(run_dir: Path, generator_hash: str, discriminator_hash: str, seed: int, resumed: bool) -> None:
    payload = {
        "initial_generator_hash": None if resumed else generator_hash,
        "initial_discriminator_hash": None if resumed else discriminator_hash,
        "resumed_generator_hash": generator_hash if resumed else None,
        "resumed_discriminator_hash": discriminator_hash if resumed else None,
        "seed": seed,
        "distributed_sampler_seed": seed,
        "torch_version": torch.__version__,
    }
    with open(run_dir / "initial_model_hash.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    kind = "resumed" if resumed else "initial"
    print(f"{kind}_generator_hash={generator_hash}", flush=True)
    print(f"{kind}_discriminator_hash={discriminator_hash}", flush=True)


@torch.no_grad()
def validate(
    generator: nn.Module,
    discriminator: nn.Module,
    mel_loss: MultiScaleMelSpectrogramLoss,
    asr: ParakeetTDT06BV2,
    timestamp_cache: dict[str, AsrTimestamp],
    loader: DataLoader,
    validset: ScheduledNoiseDataset,
    cfg: SimpleNamespace,
    device: torch.device,
    enabled: bool,
    alpha: float,
    store: AlignmentStore | None,
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()
    stft, sample_rate = cfg.data.stft, cfg.data.sampling_rate
    totals: dict[str, float] = {}
    n = 0
    empty = 0
    clean_list, enhanced_list = [], []
    for clean_audio, noisy_audio, lengths, keys, crop_starts, crop_ends, indices in loader:
        clean_audio = clean_audio.to(device, non_blocking=True)
        noisy_audio = noisy_audio.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        clean = mag_pha_stft(clean_audio, stft)
        noisy = mag_pha_stft(noisy_audio, stft)
        asr_hidden, asr_lengths = _guidance_features(asr, noisy_audio, lengths)
        gen = _forward_generator(generator, noisy.mag, noisy.pha, asr_hidden, asr_lengths)
        enhanced_audio = mag_pha_istft(gen.mag, gen.pha, stft, length=clean_audio.size(1))
        gen_hat = mag_pha_stft(enhanced_audio, stft, eps=1e-10)
        sample_error = None
        content = None
        if enabled:
            if store is None:
                raise RuntimeError("validation weighting requires the alignment cache")
            sample_error, content, _fraction = batch_sample_error(
                store, validset, VALID_SPLIT, 0, indices, keys, crop_starts, crop_ends, clean_audio.size(1)
            )
            sample_error = sample_error.to(device)
            content = content.to(device)
        disc_out = discriminator(clean_audio.unsqueeze(1), enhanced_audio.unsqueeze(1))
        token_ids, empty_n = _tdt_targets_from_cache(timestamp_cache, keys, crop_starts, crop_ends)
        tdt = asr.loss_from_ids(enhanced_audio, lengths, token_ids).mean()
        losses = _generator_losses(
            clean,
            gen,
            gen_hat,
            disc_out,
            tdt,
            cfg.train.loss,
            stft,
            mel_loss,
            clean_audio,
            enhanced_audio,
            enabled,
            alpha,
            sample_error,
            content,
        )
        batch = clean_audio.size(0)
        n += batch
        empty += empty_n
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch
        batch_clean, batch_enhanced = unpadded(clean_audio, enhanced_audio, lengths)
        clean_list.extend(batch_clean)
        enhanced_list.extend(batch_enhanced)
    score_sum, score_n = pesq_sum(clean_list, enhanced_list, sample_rate, cfg.train.pesq_num_workers)
    n = all_reduce_sum(n, device)
    metrics = {
        "pesq": all_reduce_sum(score_sum, device) / max(all_reduce_sum(score_n, device), 1.0),
        "empty_target": all_reduce_sum(float(empty), device) / max(n, 1.0),
    }
    for name, value in totals.items():
        metrics[name] = all_reduce_sum(value, device) / max(n, 1.0)
    generator.train()
    discriminator.train()
    return metrics


def main() -> None:
    cfg, run_dir, alignment_cache, max_steps = parse_args()
    enabled = bool(cfg.error_aware.enabled)
    if enabled:
        if alignment_cache is None:
            raise RuntimeError("--alignment-cache is required when error_aware.enabled is true")
        require_cache_file(alignment_cache)
    device, rank = init_distributed()
    seed_everything(cfg.train.seed)
    writer = start_run(run_dir, cfg, rank)
    if require_asr_guidance_dim(cfg.model) <= 0:
        raise RuntimeError("this experiment requires asr_guidance_dim > 0")
    stft, sample_rate, weights, optim_cfg = (
        cfg.data.stft,
        cfg.data.sampling_rate,
        cfg.train.loss,
        cfg.train.optim,
    )
    alpha = float(cfg.error_aware.alpha) if enabled else 0.0
    variants = cfg.data.variants_per_utterance

    generator = SEMambaPP(cfg.model, stft.n_fft).to(device)
    discriminator = SEMambaPPDiscriminator().to(device)
    fresh_generator_hash = state_dict_sha256(generator)
    fresh_discriminator_hash = state_dict_sha256(discriminator)
    mel_loss = MultiScaleMelSpectrogramLoss(sample_rate)
    asr = ParakeetTDT06BV2()
    trainset, validset = datasets_from_cfg(cfg)
    fingerprint = build_fingerprint(
        asr,
        trainset,
        validset,
        sampling_rate=int(sample_rate),
        segment_size=int(cfg.data.segment_size),
        normalize=str(cfg.data.normalize),
        train_splits=list(cfg.data.train_splits),
        valid_splits=list(cfg.data.validation_splits),
        base_seed=int(cfg.train.seed),
        variants_per_utterance=None if variants is None else int(variants),
    )
    if enabled:
        if alignment_cache is None:
            raise RuntimeError("--alignment-cache is required when error_aware.enabled is true")
        verify_cache_file(alignment_cache, fingerprint)
    asr = asr.to(device)
    asr.eval()
    for parameter in asr.parameters():
        parameter.requires_grad_(False)
    optim_g = torch.optim.AdamW(
        generator.parameters(),
        optim_cfg.learning_rate,
        betas=(optim_cfg.adam_b1, optim_cfg.adam_b2),
    )
    optim_d = torch.optim.AdamW(
        discriminator.parameters(),
        optim_cfg.learning_rate,
        betas=(optim_cfg.adam_b1, optim_cfg.adam_b2),
    )
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=optim_cfg.lr_decay)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=optim_cfg.lr_decay)

    start_epoch, steps, best_pesq = 0, 0, 0.0
    state = load_checkpoint(run_dir, device)
    resumed = state is not None
    if state is not None:
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        optim_g.load_state_dict(state["optim_g"])
        optim_d.load_state_dict(state["optim_d"])
        sched_g.load_state_dict(state["sched_g"])
        sched_d.load_state_dict(state["sched_d"])
        start_epoch, steps, best_pesq = state["epoch"] + 1, state["steps"], state["best_pesq"]
        if rank == 0:
            print(f"Resumed from epoch {start_epoch} (step {steps}, best_pesq={best_pesq:.3f})", flush=True)
    if rank == 0:
        if resumed:
            _write_hashes(run_dir, state_dict_sha256(generator), state_dict_sha256(discriminator), cfg.train.seed, True)
        else:
            _write_hashes(run_dir, fresh_generator_hash, fresh_discriminator_hash, cfg.train.seed, False)
        print(
            f"distributed_sampler_seed={cfg.train.seed} error_aware={enabled} alpha={alpha} variants={variants}",
            flush=True,
        )

    timestamp_cache = load_timestamp_cache()
    _require_timestamp_coverage(timestamp_cache, trainset, "train")
    _require_timestamp_coverage(timestamp_cache, validset, "validation")
    generator = DDP(generator, device_ids=[device.index])
    discriminator = DDP(discriminator, device_ids=[device.index])
    train_loader, valid_loader = build_scheduled_loaders(trainset, validset, cfg.train, int(cfg.train.seed))
    store = AlignmentStore(alignment_cache, fingerprint) if enabled and alignment_cache is not None else None
    if rank == 0:
        print(
            f"SEMamba++: {sum(p.numel() for p in generator.parameters()) / 1e6:.3f}M params, "
            f"{dist.get_world_size()} GPU(s) x batch {cfg.train.batch_size}, "
            f"asr_timestamps={len(timestamp_cache)}",
            flush=True,
        )

    for epoch in range(start_epoch, cfg.train.epochs):
        train_loader.sampler.set_epoch(epoch)  # type: ignore[union-attr]
        trainset.set_epoch(epoch)
        generator.train()
        discriminator.train()
        for clean_audio, noisy_audio, lengths, keys, crop_starts, crop_ends, indices in train_loader:
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)
            wav_lengths = lengths.to(device, non_blocking=True)
            clean = mag_pha_stft(clean_audio, stft)
            noisy = mag_pha_stft(noisy_audio, stft)
            with torch.no_grad():
                asr_hidden, asr_lengths = _guidance_features(asr, noisy_audio, wav_lengths)
            gen = _forward_generator(generator, noisy.mag, noisy.pha, asr_hidden, asr_lengths)
            enhanced_audio = mag_pha_istft(gen.mag, gen.pha, stft)
            gen_hat = mag_pha_stft(enhanced_audio, stft, eps=1e-10)

            optim_d.zero_grad(set_to_none=True)
            disc_out = discriminator(clean_audio.unsqueeze(1), enhanced_audio.detach().unsqueeze(1))
            loss_d = discriminator_loss(disc_out)
            loss_d["total"].backward()
            optim_d.step()

            token_ids, _empty_n = _tdt_targets_from_cache(timestamp_cache, keys, crop_starts, crop_ends)
            optim_g.zero_grad(set_to_none=True)
            disc_out = discriminator(clean_audio.unsqueeze(1), enhanced_audio.unsqueeze(1))
            tdt = asr.loss_from_ids(enhanced_audio, wav_lengths, token_ids).mean()
            sample_error = None
            content = None
            error_fraction = 0.0
            if enabled:
                if store is None:
                    raise RuntimeError("training weighting requires the alignment cache")
                sample_error, content, error_fraction = batch_sample_error(
                    store, trainset, TRAIN_SPLIT, epoch, indices, keys, crop_starts, crop_ends, clean_audio.size(1)
                )
                sample_error = sample_error.to(device)
                content = content.to(device)
            losses = _generator_losses(
                clean,
                gen,
                gen_hat,
                disc_out,
                tdt,
                weights,
                stft,
                mel_loss,
                clean_audio,
                enhanced_audio,
                enabled,
                alpha,
                sample_error,
                content,
            )
            losses["total"].backward()
            optim_g.step()

            steps += 1
            if max_steps is not None and steps >= max_steps:
                if rank == 0:
                    print(
                        f"stopped at step {steps}: gen={float(losses['total']):.3f} "
                        f"disc={float(loss_d['total']):.3f} tdt={float(losses['tdt']):.3f} "
                        f"peak_mib={torch.cuda.max_memory_allocated(device) / (1024**2):.0f}",
                        flush=True,
                    )
                break
            if rank == 0 and steps % cfg.train.log_interval == 0:
                print(
                    f"epoch {epoch + 1} step {steps}: "
                    f"gen={float(losses['total']):.3f} disc={float(loss_d['total']):.3f} "
                    f"tdt={float(losses['tdt']):.3f} error_frame_fraction={error_fraction:.3f}",
                    flush=True,
                )
                log_scalars(
                    writer,
                    "train",
                    {**losses, **{f"disc_{k}": v for k, v in loss_d.items()}, "error_frame_fraction": error_fraction},
                    steps,
                )
        if max_steps is not None and steps >= max_steps:
            break

        metrics = validate(
            generator,
            discriminator,
            mel_loss,
            asr,
            timestamp_cache,
            valid_loader,
            validset,
            cfg,
            device,
            enabled,
            alpha,
            store,
        )
        sched_g.step()
        sched_d.step()
        if rank == 0:
            print(
                f"validation epoch {epoch + 1}/{cfg.train.epochs}: "
                f"PESQ={metrics['pesq']:.3f} gen={metrics['total']:.3f} "
                f"tdt={metrics['tdt']:.3f} empty={metrics['empty_target']:.4f}",
                flush=True,
            )
            log_scalars(writer, "valid", metrics, epoch + 1)
            if metrics["pesq"] > best_pesq:
                best_pesq = metrics["pesq"]
                torch.save({"generator": generator.module.state_dict()}, run_dir / "best.pt")
            torch.save(
                {
                    "generator": generator.module.state_dict(),
                    "discriminator": discriminator.module.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                    "sched_g": sched_g.state_dict(),
                    "sched_d": sched_d.state_dict(),
                    "epoch": epoch,
                    "steps": steps,
                    "best_pesq": best_pesq,
                },
                run_dir / "latest.pt",
            )
        dist.barrier()

    if store is not None:
        store.close()
    if writer is not None:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
