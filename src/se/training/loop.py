import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.se.api import canonical_name, get, load_config_file, to_dict
from src.se.dataset import build_datasets, pad_collate, worker_init_fn
from src.se.model import SEModel, log_dict, unwrap
from src.se.training.metrics import mean_pesq
from src.se.training.parallel import GpuParallel

torch.backends.cudnn.benchmark = True


def log(message: str) -> None:
    print(message, flush=True)


def collect_unpadded(
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    refs = []
    ests = []
    for i in range(clean.size(0)):
        length = min(int(lengths[i].item()), int(enhanced.size(1)))
        refs.append(clean[i, :length].detach().cpu().numpy())
        ests.append(enhanced[i, :length].detach().cpu().numpy())
    return refs, ests


class TorchrunTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("file", sys.stderr)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("mininterval", 0.5)
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        if self.disable:
            return
        if msg is None:
            msg = self.__str__()
        if self.fp.isatty():
            return super().display(msg=msg, pos=pos)
        self.fp.write(msg + "\n")
        self.fp.flush()


def build_parser(default_config: Path, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint_root", default=None)
    return parser


def parse_args(
    name: str,
    default_config: Path,
    description: str,
) -> tuple[argparse.Namespace, SimpleNamespace]:
    parser = build_parser(default_config, description)
    args = parser.parse_args()

    if args.resume is not None:
        config = load_config_file(Path(args.resume) / "config.json")
    else:
        config = load_config_file(args.config)

    loaded = getattr(config, "name", None)
    if loaded is not None and canonical_name(loaded) != name:
        parser.error(f"config name {loaded!r} is not {name}")
    config.name = name
    return args, config


def prepare_run(config: SimpleNamespace, args: argparse.Namespace) -> SimpleNamespace:
    if args.checkpoint_root is not None:
        parent = Path(args.checkpoint_root)
    elif args.resume is not None:
        parent = Path(args.resume).parent
    else:
        parent = Path(config.checkpoint_root)

    run_dir = parent / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        resume_dir = Path(args.resume)
        for name in ("g_latest", "do_latest", "g_best"):
            src = resume_dir / name
            if src.is_file():
                shutil.copy2(src, run_dir / name)
        logs_src = resume_dir / "logs"
        if logs_src.is_dir():
            shutil.copytree(logs_src, run_dir / "logs")

    config.checkpoint_root = str(run_dir)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(to_dict(config), f, indent=4)
    log(f"checkpoints directory: {run_dir}")
    return config


def load_latest(run_dir: Path, device: torch.device):
    g_path = run_dir / "g_latest"
    do_path = run_dir / "do_latest"
    if not (g_path.is_file() and do_path.is_file()):
        return None, None
    return (
        torch.load(g_path, map_location=device, weights_only=False),
        torch.load(do_path, map_location=device, weights_only=False),
    )


def save_latest(
    run_dir: Path,
    components: SEModel,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    steps: int,
    epoch: int,
    best_pesq: float,
) -> None:
    torch.save(
        {"generator": unwrap(components.model).state_dict()},
        run_dir / "g_latest",
    )
    torch.save(
        {
            "discriminator": unwrap(components.discriminator).state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "steps": steps,
            "epoch": epoch,
            "best_pesq": best_pesq,
        },
        run_dir / "do_latest",
    )


def save_best(run_dir: Path, components: SEModel) -> None:
    torch.save(
        {"generator": unwrap(components.model).state_dict()},
        run_dir / "g_best",
    )


def train_loader(
    dataset,
    batch_size: int,
    env: SimpleNamespace,
    parallel: GpuParallel,
):
    sampler = parallel.train_sampler(dataset)
    kwargs = {}
    if env.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = env.prefetch_factor
        kwargs["worker_init_fn"] = worker_init_fn
    return DataLoader(
        dataset,
        num_workers=env.num_workers,
        shuffle=sampler is None,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        **kwargs,
    ), sampler


def valid_loader(dataset, env: SimpleNamespace, parallel: GpuParallel):
    sampler = parallel.valid_sampler(dataset)
    return DataLoader(
        dataset,
        num_workers=0,
        shuffle=False,
        sampler=sampler,
        batch_size=max(1, int(env.val_batch_size)),
        pin_memory=True,
        drop_last=False,
        collate_fn=pad_collate,
    )


@torch.no_grad()
def validate(
    components: SEModel,
    dataset,
    config: SimpleNamespace,
    parallel: GpuParallel,
    epoch: int,
) -> dict[str, float]:
    components.eval()
    torch.cuda.empty_cache()
    env = config.train.env
    device = parallel.device
    loader = valid_loader(dataset, env, parallel)
    totals: dict[str, float] = {"n": 0.0}
    refs: list[np.ndarray] = []
    ests: list[np.ndarray] = []
    pbar = None
    if parallel.is_main:
        pbar = TorchrunTqdm(
            total=len(loader),
            desc=f"Validation {epoch + 1}/{env.epochs}",
            unit="batch",
            leave=False,
        )

    for clean, noisy, lengths in loader:
        clean = clean.to(device, non_blocking=True)
        noisy = noisy.to(device, non_blocking=True)
        pred, target = components.forward_pair(noisy, clean)
        terms = components.generator_loss(pred, target)
        batch_n = clean.size(0)
        totals["n"] += batch_n
        for name, value in log_dict(terms).items():
            totals[name] = totals.get(name, 0.0) + value * batch_n
        batch_refs, batch_ests = collect_unpadded(clean, pred.audio, lengths)
        refs.extend(batch_refs)
        ests.extend(batch_ests)
        if pbar is not None:
            pbar.set_postfix(
                {"gen": f"{float(terms.total):.3f}"},
                refresh=False,
            )
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    pesq_sum, pesq_n = mean_pesq(
        refs,
        ests,
        config.data.sampling_rate,
        env.pesq_num_workers,
    )
    n = parallel.reduce_sum(totals["n"])
    metrics = {
        "pesq": parallel.reduce_sum(pesq_sum)
        / max(parallel.reduce_sum(pesq_n), 1.0),
    }
    for name, value in totals.items():
        if name == "n":
            continue
        metrics[name] = parallel.reduce_sum(value) / max(n, 1.0)
    return metrics


def train(config: SimpleNamespace, parallel: GpuParallel) -> None:
    env = config.train.env
    device = parallel.device
    batch_size = parallel.per_gpu_batch_size(env.batch_size)
    random.seed(env.seed)
    np.random.seed(env.seed)
    torch.manual_seed(env.seed)
    torch.cuda.manual_seed_all(env.seed)

    run_dir = Path(config.checkpoint_root)
    state_g, state_d = load_latest(run_dir, device)
    steps = 0
    last_epoch = -1
    best_pesq = 0.0
    if state_d is not None:
        steps = int(state_d["steps"])
        last_epoch = int(state_d["epoch"])
        best_pesq = float(state_d.get("best_pesq", 0.0))
        if parallel.is_main:
            log(
                f"Loaded checkpoint (step {state_d['steps']}, "
                f"epoch {state_d['epoch'] + 1}, best_pesq={best_pesq:.3f})"
            )

    components = get(config.name, config)
    components.to(device)

    optim = config.train.optim
    optim_g = torch.optim.AdamW(
        components.model.parameters(),
        optim.learning_rate,
        betas=[optim.adam_b1, optim.adam_b2],
    )
    optim_d = torch.optim.AdamW(
        components.discriminator.parameters(),
        optim.learning_rate,
        betas=[optim.adam_b1, optim.adam_b2],
    )
    if state_d is not None:
        components.model.load_state_dict(state_g["generator"])
        components.discriminator.load_state_dict(state_d["discriminator"])
        optim_g.load_state_dict(state_d["optim_g"])
        optim_d.load_state_dict(state_d["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=optim.lr_decay,
        last_epoch=last_epoch,
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d,
        gamma=optim.lr_decay,
        last_epoch=last_epoch,
    )
    parallel.wrap(components)

    if parallel.is_main:
        n_params = sum(p.numel() for p in unwrap(components.model).parameters())
        log(f"model: {config.name}")
        log(f"Total Parameters: {n_params / 1e6:.3f}M")
        log(f"Batch size per GPU: {batch_size}")
        log(f"world_size: {parallel.world_size}")
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    trainset, validset = build_datasets(config)
    loader, sampler = train_loader(trainset, batch_size, env, parallel)
    writer = SummaryWriter(str(run_dir / "logs")) if parallel.is_main else None
    start_epoch = 0 if last_epoch < 0 else last_epoch + 1
    stop = False

    components.train()

    for epoch in range(start_epoch, env.epochs):
        if parallel.is_main:
            log(f"Epoch: {epoch + 1}")
        if sampler is not None:
            sampler.set_epoch(epoch)

        pbar = None
        if parallel.is_main:
            pbar = TorchrunTqdm(
                total=len(loader) * batch_size,
                desc=f"Epoch {epoch + 1}/{env.epochs}",
                unit="sample",
            )

        for clean, noisy in loader:
            clean = clean.to(device, non_blocking=True)
            noisy = noisy.to(device, non_blocking=True)
            pred, target = components.forward_pair(noisy, clean)

            optim_d.zero_grad(set_to_none=True)
            d_terms = components.discriminator_loss(pred, target)
            d_terms.total.backward()
            optim_d.step()

            optim_g.zero_grad(set_to_none=True)
            g_terms = components.generator_loss(pred, target)
            g_total = g_terms.total
            if not torch.isfinite(g_total):
                raise ValueError(
                    f"non-finite generator loss at step {steps}: {g_total}"
                )
            g_total.backward()
            optim_g.step()

            if parallel.is_main:
                g_log = log_dict(g_terms)
                d_log = log_dict(d_terms)
                pbar.update(batch_size)
                pbar.set_postfix(
                    {
                        "step": steps + 1,
                        "gen": f"{g_log['total']:.3f}",
                        "disc": f"{d_log['total']:.3f}",
                    },
                    refresh=False,
                )
                if steps % env.summary_interval == 0:
                    writer.add_scalar("Training/Generator Loss", g_log["total"], steps)
                    writer.add_scalar(
                        "Training/Discriminator Loss",
                        d_log["total"],
                        steps,
                    )
                    for name, value in g_log.items():
                        if name == "total":
                            continue
                        writer.add_scalar(f"Training/G/{name}", value, steps)
                    for name, value in d_log.items():
                        if name == "total":
                            continue
                        writer.add_scalar(f"Training/D/{name}", value, steps)

            steps += 1
            if env.max_steps is not None and steps >= env.max_steps:
                stop = True
                break

        if pbar is not None:
            pbar.close()

        if stop:
            if parallel.is_main:
                save_latest(
                    run_dir,
                    components,
                    optim_g,
                    optim_d,
                    steps,
                    epoch - 1,
                    best_pesq,
                )
                log(f"Stopped at max_steps={env.max_steps} (step {steps})")
            break

        metrics = validate(
            components,
            validset,
            config,
            parallel,
            epoch,
        )
        if parallel.is_main:
            parts = [
                f"Validation (epoch {epoch + 1}/{env.epochs}):",
                f"PESQ={metrics['pesq']:.3f}",
            ]
            if "total" in metrics:
                parts.append(f"gen={metrics['total']:.3f}")
            message = " ".join(parts)
            tqdm.write(message)
            log(message)
            writer.add_scalar("Validation/PESQ Score", metrics["pesq"], epoch + 1)
            for name, value in metrics.items():
                if name == "pesq":
                    continue
                writer.add_scalar(f"Validation/{name}", value, epoch + 1)
            if metrics["pesq"] > best_pesq:
                best_pesq = metrics["pesq"]
                save_best(run_dir, components)
                log(
                    f"Updated best checkpoint (PESQ={best_pesq:.3f}) "
                    f"at epoch {epoch + 1}"
                )
            save_latest(
                run_dir,
                components,
                optim_g,
                optim_d,
                steps,
                epoch,
                best_pesq,
            )
            log(f"Saved latest checkpoint at end of epoch {epoch + 1} (step {steps})")

        components.train()
        parallel.barrier()
        scheduler_g.step()
        scheduler_d.step()

    if writer is not None:
        writer.close()
    parallel.close()


def main(name: str, default_config: Path, description: str) -> None:
    args, config = parse_args(name, default_config, description)
    parallel = GpuParallel.start()
    if parallel.is_main:
        config = prepare_run(config, args)
    config.checkpoint_root = parallel.broadcast(
        config.checkpoint_root if parallel.is_main else None
    )
    train(config, parallel)
