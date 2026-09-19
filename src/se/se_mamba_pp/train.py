"""Train SEMamba++. torchrun --nproc_per_node=N -m src.se.se_mamba_pp.train --run_dir DIR [--config JSON]"""

import src.se.common.cuda_local as _cuda_local  # noqa: F401

import math
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.asr.parakeet_tdt_0_6b_v2 import ParakeetTDT06BV2
from src.asr.timestamp_cache import (
    crop_batch_token_ids,
    load_timestamp_cache,
    missing_timestamp_keys,
)
from src.data.corpora.librispeech import utterance_key
from src.data.schema import AsrTimestamp
from src.se.common.dataset import AdditiveNoiseDataset, build_datasets, pad_collate
from src.se.common.pesq import pesq_sum
from src.se.common.stft import Spec, mag_pha_istft, mag_pha_stft
from src.se.common.training import (
    all_reduce_sum,
    build_loaders,
    init_distributed,
    load_checkpoint,
    log_scalars,
    parse_args,
    seed_everything,
    start_run,
    unpadded,
)
from src.se.se_mamba_pp.asr_guidance import ENCODER_LAYER
from src.se.se_mamba_pp.model import SEMambaPP

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.json"
torch.backends.cudnn.benchmark = True


def _wav_lengths(audio: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
    if lengths is not None:
        return lengths.to(device=audio.device, dtype=torch.int64)
    return torch.full((audio.size(0),), audio.size(1), device=audio.device, dtype=torch.int64)


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


def _grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().item() ** 2)
    return total**0.5


def _require_cache_coverage(cache: dict[str, AsrTimestamp], dataset: AdditiveNoiseDataset, name: str) -> None:
    keys = [utterance_key(utterance) for utterance in dataset.utterances]
    missing = missing_timestamp_keys(cache, keys)
    if missing:
        shown = ", ".join(missing[:5])
        raise KeyError(f"timestamp cache missing {len(missing)} {name} keys, e.g. {shown}")


def _forward_generator(
    generator: nn.Module,
    noisy_mag: torch.Tensor,
    noisy_pha: torch.Tensor,
    asr_hidden: torch.Tensor | None,
    asr_lengths: torch.Tensor | None,
) -> Spec:
    if asr_hidden is None:
        return generator(noisy_mag, noisy_pha)
    return generator(noisy_mag, noisy_pha, asr_hidden, asr_lengths)


@torch.no_grad()
def validate(
    generator: nn.Module,
    asr: ParakeetTDT06BV2,
    cache: dict[str, AsrTimestamp],
    loader: DataLoader,
    cfg: SimpleNamespace,
    device: torch.device,
    guidance: bool,
) -> dict[str, float]:
    generator.eval()
    stft, sample_rate = cfg.data.stft, cfg.data.sampling_rate
    max_val_batches = getattr(cfg.train, "max_val_batches", None)
    total_tdt = 0.0
    n = 0
    empty = 0
    clean_list, enhanced_list = [], []
    for batch_index, (clean_audio, noisy_audio, lengths, utterance_keys, crop_starts, crop_ends) in enumerate(loader):
        if max_val_batches is not None and batch_index >= int(max_val_batches):
            break
        clean_audio = clean_audio.to(device, non_blocking=True)
        noisy_audio = noisy_audio.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        noisy = mag_pha_stft(noisy_audio, stft)
        asr_hidden, asr_lengths = (None, None)
        if guidance:
            asr_hidden, asr_lengths = _guidance_features(asr, noisy_audio, lengths)
        gen = _forward_generator(generator, noisy.mag, noisy.pha, asr_hidden, asr_lengths)
        enhanced_audio = mag_pha_istft(gen.mag, gen.pha, stft, length=clean_audio.size(1))
        token_ids, empty_n = _tdt_targets_from_cache(cache, utterance_keys, crop_starts, crop_ends)
        tdt = asr.loss_from_ids(enhanced_audio, lengths, token_ids)
        batch = clean_audio.size(0)
        n += batch
        empty += empty_n
        total_tdt += float(tdt.sum())
        batch_clean, batch_enhanced = unpadded(clean_audio, enhanced_audio, lengths)
        clean_list.extend(batch_clean)
        enhanced_list.extend(batch_enhanced)

    score_sum, score_n = pesq_sum(clean_list, enhanced_list, sample_rate, cfg.train.pesq_num_workers)
    n = all_reduce_sum(n, device)
    metrics = {
        "pesq": all_reduce_sum(score_sum, device) / max(all_reduce_sum(score_n, device), 1.0),
        "tdt": all_reduce_sum(total_tdt, device) / max(n, 1.0),
        "empty_target": all_reduce_sum(float(empty), device) / max(n, 1.0),
    }
    generator.train()
    return metrics


def main() -> None:
    cfg, run_dir = parse_args(DEFAULT_CONFIG, "Train SEMamba++")
    device, rank = init_distributed()
    seed_everything(cfg.train.seed)
    writer = start_run(run_dir, cfg, rank)
    stft, weights, optim_cfg = cfg.data.stft, cfg.train.loss, cfg.train.optim
    guidance = int(getattr(cfg.model, "asr_guidance_dim", 0) or 0) > 0
    max_steps = getattr(cfg.train, "max_steps", None)

    trainset, validset = build_datasets(cfg)
    timestamp_cache = load_timestamp_cache()
    _require_cache_coverage(timestamp_cache, trainset, "train")
    _require_cache_coverage(timestamp_cache, validset, "validation")

    generator = SEMambaPP(cfg.model, stft.n_fft).to(device)
    asr = ParakeetTDT06BV2().to(device)
    asr.eval()
    for parameter in asr.parameters():
        parameter.requires_grad_(False)
    optim_g = torch.optim.AdamW(
        generator.parameters(),
        optim_cfg.learning_rate,
        betas=(optim_cfg.adam_b1, optim_cfg.adam_b2),
    )
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=optim_cfg.lr_decay)

    start_epoch, steps, best_pesq = 0, 0, 0.0
    state = load_checkpoint(run_dir, device)
    if state is not None:
        generator.load_state_dict(state["generator"])
        optim_g.load_state_dict(state["optim_g"])
        sched_g.load_state_dict(state["sched_g"])
        start_epoch, steps, best_pesq = state["epoch"] + 1, state["steps"], state["best_pesq"]
        if rank == 0:
            print(f"Resumed from epoch {start_epoch} (step {steps}, best_pesq={best_pesq:.3f})", flush=True)

    generator = DDP(generator, device_ids=[device.index])
    train_loader, valid_loader = build_loaders(trainset, validset, cfg.train, train_collate_fn=pad_collate)
    if rank == 0:
        print(
            f"SEMamba++: {sum(p.numel() for p in generator.parameters()) / 1e6:.3f}M params, "
            f"{dist.get_world_size()} GPU(s) x batch {cfg.train.batch_size}, "
            f"asr_timestamps={len(timestamp_cache)}",
            flush=True,
        )

    empty_n = 0
    seen_n = 0
    stop = False
    for epoch in range(start_epoch, cfg.train.epochs):
        train_loader.sampler.set_epoch(epoch)  # type: ignore[union-attr]
        generator.train()
        for clean_audio, noisy_audio, lengths, utterance_keys, crop_starts, crop_ends in train_loader:
            step_t0 = time.perf_counter()
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)
            wav_lengths = lengths.to(device, non_blocking=True)
            noisy = mag_pha_stft(noisy_audio, stft)
            asr_hidden, asr_lengths = (None, None)
            if guidance:
                with torch.no_grad():
                    asr_hidden, asr_lengths = _guidance_features(asr, noisy_audio, wav_lengths)
            gen = _forward_generator(generator, noisy.mag, noisy.pha, asr_hidden, asr_lengths)
            enhanced_audio = mag_pha_istft(gen.mag, gen.pha, stft)
            token_ids, empty_batch = _tdt_targets_from_cache(
                timestamp_cache,
                utterance_keys,
                crop_starts,
                crop_ends,
            )
            empty_n += empty_batch
            seen_n += clean_audio.size(0)
            losses = {"tdt": weights.tdt * asr.loss_from_ids(enhanced_audio, wav_lengths, token_ids).mean()}

            optim_g.zero_grad(set_to_none=True)
            losses["tdt"].backward()
            optim_g.step()

            steps += 1
            if rank == 0 and steps % cfg.train.log_interval == 0:
                step_s = time.perf_counter() - step_t0
                tdt_val = float(losses["tdt"].detach())
                gen_mod = generator.module
                proj, fuse = gen_mod.asr_proj, gen_mod.asr_fuse
                asr_has_grad = any(parameter.grad is not None for parameter in asr.parameters())
                print(
                    f"epoch {epoch + 1} step {steps}: tdt={tdt_val:.3f} finite={math.isfinite(tdt_val)} "
                    f"empty={empty_n}/{seen_n} step_s={step_s:.3f} "
                    f"se_grad={_grad_norm(gen_mod):.4g} "
                    f"proj_grad={_grad_norm(proj) if proj is not None else 0.0:.4g} "
                    f"fuse_grad={_grad_norm(fuse) if fuse is not None else 0.0:.4g} "
                    f"asr_grad={int(asr_has_grad)} "
                    f"vram_mb={torch.cuda.max_memory_allocated(device) / (1024**2):.0f}",
                    flush=True,
                )
                log_scalars(writer, "train", losses, steps)
            if max_steps is not None and steps >= int(max_steps):
                stop = True
                break

        metrics = validate(generator, asr, timestamp_cache, valid_loader, cfg, device, guidance)
        sched_g.step()
        if rank == 0:
            print(
                f"validation epoch {epoch + 1}/{cfg.train.epochs}: "
                f"PESQ={metrics['pesq']:.3f} tdt={metrics['tdt']:.3f} empty={metrics['empty_target']:.4f}",
                flush=True,
            )
            log_scalars(writer, "valid", metrics, epoch + 1)
            if metrics["pesq"] > best_pesq:
                best_pesq = metrics["pesq"]
                torch.save({"generator": generator.module.state_dict()}, run_dir / "best.pt")
            torch.save(
                {
                    "generator": generator.module.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "sched_g": sched_g.state_dict(),
                    "epoch": epoch,
                    "steps": steps,
                    "best_pesq": best_pesq,
                },
                run_dir / "latest.pt",
            )
        dist.barrier()
        if stop:
            break

    if writer is not None:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
