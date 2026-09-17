"""Train SEMamba++. torchrun --nproc_per_node=N -m src.se.se_mamba_pp.train --run_dir DIR [--config JSON]"""
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.se.common.dataset import build_datasets
from src.se.common.pesq import pesq_sum
from src.se.se_mamba_pp.discriminator import SEMambaPPDiscriminator
from src.se.se_mamba_pp.loss import MultiScaleMelSpectrogramLoss, discriminator_loss, generator_loss
from src.se.se_mamba_pp.model import SEMambaPP
from src.se.common.stft import mag_pha_istft, mag_pha_stft
from src.se.common.training import (
    all_reduce_sum,
    build_loaders,
    init_distributed,
    load_checkpoint,
    log,
    log_scalars,
    parse_args,
    seed_everything,
    start_run,
    unpadded,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.json"
torch.backends.cudnn.benchmark = True


@torch.no_grad()
def validate(generator, discriminator, mel_loss, loader, cfg, device) -> dict[str, float]:
    """All generator loss terms plus PESQ; reduce shards across ranks."""
    generator.eval()
    discriminator.eval()
    stft, sr = cfg.data.stft, cfg.data.sampling_rate
    totals: dict[str, float] = {}
    n = 0
    refs, ests = [], []
    for clean_audio, noisy_audio, lengths in loader:
        clean_audio = clean_audio.to(device, non_blocking=True)
        noisy_audio = noisy_audio.to(device, non_blocking=True)
        clean = mag_pha_stft(clean_audio, stft)
        noisy = mag_pha_stft(noisy_audio, stft)
        gen = generator(noisy.mag, noisy.pha)
        audio_g = mag_pha_istft(gen.mag, gen.pha, stft, length=clean_audio.size(1))
        gen_hat = mag_pha_stft(audio_g, stft, eps=1e-10)
        d = discriminator(clean_audio.unsqueeze(1), audio_g.unsqueeze(1))
        mel = mel_loss(clean_audio.unsqueeze(1), audio_g.unsqueeze(1))
        losses = generator_loss(clean, gen, gen_hat, d, mel, cfg.train.loss, stft.n_fft)
        batch = clean_audio.size(0)
        n += batch
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch
        batch_refs, batch_ests = unpadded(clean_audio, audio_g, lengths)
        refs.extend(batch_refs)
        ests.extend(batch_ests)

    score_sum, score_n = pesq_sum(refs, ests, sr, cfg.train.pesq_num_workers)
    n = all_reduce_sum(n, device)
    metrics = {"pesq": all_reduce_sum(score_sum, device) / max(all_reduce_sum(score_n, device), 1.0)}
    for name, value in totals.items():
        metrics[name] = all_reduce_sum(value, device) / max(n, 1.0)
    generator.train()
    discriminator.train()
    return metrics


def main() -> None:
    cfg, run_dir = parse_args(DEFAULT_CONFIG, "Train SEMamba++")
    device, rank = init_distributed()
    seed_everything(cfg.train.seed)
    writer = start_run(run_dir, cfg, rank)
    stft, sr, w, optim_cfg = cfg.data.stft, cfg.data.sampling_rate, cfg.train.loss, cfg.train.optim

    generator = SEMambaPP(cfg.model, stft.n_fft).to(device)
    discriminator = SEMambaPPDiscriminator().to(device)
    mel_loss = MultiScaleMelSpectrogramLoss(sr)
    optim_g = torch.optim.AdamW(generator.parameters(), optim_cfg.learning_rate, betas=(optim_cfg.adam_b1, optim_cfg.adam_b2))
    optim_d = torch.optim.AdamW(discriminator.parameters(), optim_cfg.learning_rate, betas=(optim_cfg.adam_b1, optim_cfg.adam_b2))
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=optim_cfg.lr_decay)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=optim_cfg.lr_decay)

    start_epoch, steps, best_pesq = 0, 0, 0.0
    state = load_checkpoint(run_dir, device)
    if state is not None:
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        optim_g.load_state_dict(state["optim_g"])
        optim_d.load_state_dict(state["optim_d"])
        sched_g.load_state_dict(state["sched_g"])
        sched_d.load_state_dict(state["sched_d"])
        start_epoch, steps, best_pesq = state["epoch"] + 1, state["steps"], state["best_pesq"]
        if rank == 0:
            log(f"Resumed from epoch {start_epoch} (step {steps}, best_pesq={best_pesq:.3f})")

    generator = DDP(generator, device_ids=[device.index])
    discriminator = DDP(discriminator, device_ids=[device.index])
    trainset, validset = build_datasets(cfg)
    train_loader, valid_loader = build_loaders(trainset, validset, cfg.train)
    if rank == 0:
        log(f"SEMamba++: {sum(p.numel() for p in generator.parameters()) / 1e6:.3f}M params, "
            f"{dist.get_world_size()} GPU(s) x batch {cfg.train.batch_size}")

    for epoch in range(start_epoch, cfg.train.epochs):
        train_loader.sampler.set_epoch(epoch)
        generator.train()
        discriminator.train()
        for clean_audio, noisy_audio in train_loader:
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)
            clean = mag_pha_stft(clean_audio, stft)
            noisy = mag_pha_stft(noisy_audio, stft)

            # Generator forward: mag/phase -> waveform -> re-STFT (consistency)
            gen = generator(noisy.mag, noisy.pha)
            audio_g = mag_pha_istft(gen.mag, gen.pha, stft)
            gen_hat = mag_pha_stft(audio_g, stft, eps=1e-10)

            # Discriminator (LSGAN): CQT and multi-resolution discriminators on clean / generated waveforms
            optim_d.zero_grad(set_to_none=True)
            d = discriminator(clean_audio.unsqueeze(1), audio_g.detach().unsqueeze(1))
            loss_d = discriminator_loss(d)
            loss_d["total"].backward()
            optim_d.step()

            # Generator: adversarial and feature-matching from the updated discriminator, plus mel and spectral losses
            optim_g.zero_grad(set_to_none=True)
            d = discriminator(clean_audio.unsqueeze(1), audio_g.unsqueeze(1))
            mel = mel_loss(clean_audio.unsqueeze(1), audio_g.unsqueeze(1))
            losses = generator_loss(clean, gen, gen_hat, d, mel, w, stft.n_fft)
            losses["total"].backward()
            optim_g.step()

            steps += 1
            if rank == 0 and steps % cfg.train.log_interval == 0:
                log(f"epoch {epoch + 1} step {steps}: gen={float(losses['total']):.3f} disc={float(loss_d['total']):.3f}")
                log_scalars(writer, "train", {**losses, **{f"disc_{k}": v for k, v in loss_d.items()}}, steps)

        metrics = validate(generator, discriminator, mel_loss, valid_loader, cfg, device)
        sched_g.step()
        sched_d.step()
        if rank == 0:
            log(f"validation epoch {epoch + 1}/{cfg.train.epochs}: PESQ={metrics['pesq']:.3f} gen={metrics['total']:.3f}")
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

    if writer is not None:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
