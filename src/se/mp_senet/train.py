"""Train MP-SENet. torchrun --nproc_per_node=N -m src.se.mp_senet.train --run_dir DIR [--config JSON]"""

import src.se.common.cuda_local as _cuda_local  # noqa: F401

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.se.common.dataset import build_datasets
from src.se.common.metric_discriminator import MetricDiscriminator
from src.se.common.pesq import pesq_batch_target, pesq_sum
from src.se.common.stft import mag_pha_istft, mag_pha_stft
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
from src.se.mp_senet.loss import generator_loss
from src.se.mp_senet.model import MPNet

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.json"
torch.backends.cudnn.benchmark = True


@torch.no_grad()
def validate(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    cfg: SimpleNamespace,
    device: torch.device,
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()
    stft, sample_rate = cfg.data.stft, cfg.data.sampling_rate
    totals: dict[str, float] = {}
    n = 0
    clean_list, enhanced_list = [], []
    for clean_audio, noisy_audio, lengths, *_ in loader:
        clean_audio = clean_audio.to(device, non_blocking=True)
        noisy_audio = noisy_audio.to(device, non_blocking=True)
        clean = mag_pha_stft(clean_audio, stft)
        noisy = mag_pha_stft(noisy_audio, stft)
        gen = generator(noisy.mag, noisy.pha)
        enhanced_audio = mag_pha_istft(gen.mag, gen.pha, stft, length=clean_audio.size(1))
        gen_hat = mag_pha_stft(enhanced_audio, stft, eps=1e-10)
        metric_g = discriminator(clean.mag, gen_hat.mag)
        losses = generator_loss(clean, clean_audio, gen, enhanced_audio, gen_hat, metric_g, cfg.train.loss)
        batch = clean_audio.size(0)
        n += batch
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch
        batch_clean, batch_enhanced = unpadded(clean_audio, enhanced_audio, lengths)
        clean_list.extend(batch_clean)
        enhanced_list.extend(batch_enhanced)

    score_sum, score_n = pesq_sum(clean_list, enhanced_list, sample_rate, cfg.train.pesq_num_workers)
    n = all_reduce_sum(n, device)
    metrics = {"pesq": all_reduce_sum(score_sum, device) / max(all_reduce_sum(score_n, device), 1.0)}
    for name, value in totals.items():
        metrics[name] = all_reduce_sum(value, device) / max(n, 1.0)
    generator.train()
    discriminator.train()
    return metrics


def main() -> None:
    cfg, run_dir = parse_args(DEFAULT_CONFIG, "Train MP-SENet")
    device, rank = init_distributed()
    seed_everything(cfg.train.seed)
    writer = start_run(run_dir, cfg, rank)
    stft, sample_rate, weights, optim_cfg = (
        cfg.data.stft,
        cfg.data.sampling_rate,
        cfg.train.loss,
        cfg.train.optim,
    )

    generator = MPNet(cfg.model, stft.n_fft).to(device)
    discriminator = MetricDiscriminator().to(device)
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

    generator = DDP(generator, device_ids=[device.index])
    discriminator = DDP(discriminator, device_ids=[device.index])
    trainset, validset = build_datasets(cfg)
    train_loader, valid_loader = build_loaders(trainset, validset, cfg.train)
    if rank == 0:
        print(
            f"MP-SENet: {sum(p.numel() for p in generator.parameters()) / 1e6:.3f}M params, "
            f"{dist.get_world_size()} GPU(s) x batch {cfg.train.batch_size}",
            flush=True,
        )

    for epoch in range(start_epoch, cfg.train.epochs):
        train_loader.sampler.set_epoch(epoch)  # type: ignore[union-attr]
        generator.train()
        discriminator.train()
        for clean_audio, noisy_audio, *_ in train_loader:
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)
            clean = mag_pha_stft(clean_audio, stft)
            noisy = mag_pha_stft(noisy_audio, stft)

            # Generator forward: masked mag/phase -> waveform -> re-STFT (consistency + discriminator input)
            gen = generator(noisy.mag, noisy.pha)
            enhanced_audio = mag_pha_istft(gen.mag, gen.pha, stft)
            gen_hat = mag_pha_stft(enhanced_audio, stft, eps=1e-10)

            # Discriminator (MetricGAN): regress 1 on clean pairs, normalized PESQ on generated pairs
            pesq_target = pesq_batch_target(clean_audio, enhanced_audio.detach(), sample_rate)
            optim_d.zero_grad(set_to_none=True)
            metric_r = discriminator(clean.mag, clean.mag)
            metric_g = discriminator(clean.mag, gen_hat.mag.detach())
            loss_d = F.mse_loss(metric_r.flatten(), torch.ones_like(metric_r.flatten()))
            if pesq_target is not None:
                loss_d = loss_d + F.mse_loss(metric_g.flatten(), pesq_target.to(device))
            loss_d.backward()
            optim_d.step()

            # Generator: metric term from the updated discriminator plus spectral losses
            optim_g.zero_grad(set_to_none=True)
            metric_g = discriminator(clean.mag, gen_hat.mag)
            losses = generator_loss(clean, clean_audio, gen, enhanced_audio, gen_hat, metric_g, weights)
            losses["total"].backward()
            optim_g.step()

            steps += 1
            if rank == 0 and steps % cfg.train.log_interval == 0:
                print(
                    f"epoch {epoch + 1} step {steps}: gen={float(losses['total']):.3f} disc={float(loss_d):.3f}",
                    flush=True,
                )
                log_scalars(writer, "train", {**losses, "disc": loss_d}, steps)

        metrics = validate(generator, discriminator, valid_loader, cfg, device)
        sched_g.step()
        sched_d.step()
        if rank == 0:
            print(
                f"validation epoch {epoch + 1}/{cfg.train.epochs}: "
                f"PESQ={metrics['pesq']:.3f} gen={metrics['total']:.3f}",
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

    if writer is not None:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
