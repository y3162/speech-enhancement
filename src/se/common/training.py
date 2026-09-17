"""Shared helpers for train scripts. Loops and G/D updates stay in each model's train.py."""

import argparse
import json
import os
import random
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

from src.se.common.dataset import pad_collate, worker_init_fn


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_namespace(v) for v in value]
    return value


def to_dict(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(value).items()}
    if isinstance(value, list):
        return [to_dict(v) for v in value]
    return value


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path, encoding="utf-8") as f:
        return to_namespace(json.load(f))


def parse_args(default_config: Path, description: str) -> tuple[SimpleNamespace, Path]:
    """Read --config / --run_dir. If run_dir/config.json exists, resume from it."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help=f"default: {default_config}")
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    saved = run_dir / "config.json"
    if saved.is_file():
        if args.config is not None:
            parser.error(f"{saved} exists (resume); do not pass --config")
        return load_config(saved), run_dir
    return load_config(args.config or default_config), run_dir


def init_distributed() -> tuple[torch.device, int]:
    """Requires torchrun. Always creates a process group, including 1 GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training.")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=120), device_id=device)
    return device, dist.get_rank()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def start_run(run_dir: Path, cfg: SimpleNamespace, rank: int) -> SummaryWriter | None:
    """Rank 0 creates run_dir, writes config.json, and opens a TensorBoard writer."""
    if rank != 0:
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = run_dir / "config.json"
    if not saved.is_file():
        with open(saved, "w", encoding="utf-8") as f:
            json.dump(to_dict(cfg), f, indent=4)
    return SummaryWriter(str(run_dir / "logs"))


def build_loaders(
    trainset: Dataset,
    validset: Dataset,
    train: SimpleNamespace,
) -> tuple[DataLoader, DataLoader]:
    train_kwargs: dict[str, Any] = {}
    if train.num_workers > 0:
        train_kwargs["persistent_workers"] = True
        train_kwargs["prefetch_factor"] = train.prefetch_factor
        train_kwargs["worker_init_fn"] = worker_init_fn
    train_loader = DataLoader(
        trainset,
        batch_size=train.batch_size,
        sampler=DistributedSampler(trainset),
        num_workers=train.num_workers,
        pin_memory=True,
        drop_last=True,
        **train_kwargs,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=train.val_batch_size,
        sampler=DistributedSampler(validset, shuffle=False, drop_last=False),
        num_workers=0,
        pin_memory=True,
        collate_fn=pad_collate,
    )
    return train_loader, valid_loader


def load_checkpoint(run_dir: Path, device: torch.device) -> dict[str, Any] | None:
    path = run_dir / "latest.pt"
    if not path.is_file():
        return None
    return torch.load(path, map_location=device, weights_only=False)


def all_reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def unpadded(
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    clean_list, enhanced_list = [], []
    for i in range(clean.size(0)):
        length = min(int(lengths[i]), int(enhanced.size(1)))
        clean_list.append(clean[i, :length].detach().cpu().numpy())
        enhanced_list.append(enhanced[i, :length].detach().cpu().numpy())
    return clean_list, enhanced_list


def log_scalars(
    writer: SummaryWriter | None,
    prefix: str,
    values: dict[str, Any],
    step: int,
) -> None:
    if writer is None:
        return
    for name, value in values.items():
        writer.add_scalar(f"{prefix}/{name}", float(value), step)


def log(message: str) -> None:
    print(message, flush=True)
