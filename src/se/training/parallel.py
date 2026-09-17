import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from src.se.model import SEModel


def _configure_runtime() -> None:
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


class GpuParallel:
    def __init__(self, rank: int, local_rank: int, world_size: int) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device("cuda", local_rank)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @classmethod
    def start(cls) -> "GpuParallel":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training.")
        _configure_runtime()
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        if world_size > 1:
            init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=120),
                device_id=torch.device("cuda", local_rank),
            )
        return cls(rank, local_rank, world_size)

    def close(self) -> None:
        if self.world_size > 1:
            destroy_process_group()

    def wrap(self, components: SEModel) -> None:
        if self.world_size <= 1:
            return
        components.model = DistributedDataParallel(
            components.model,
            device_ids=[self.local_rank],
        )
        discriminator = DistributedDataParallel(
            components.discriminator,
            device_ids=[self.local_rank],
        )
        components.loss.set_discriminator(discriminator)

    def reduce_sum(self, value: float) -> float:
        tensor = torch.tensor([value], device=self.device, dtype=torch.float64)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def broadcast(self, value):
        objects = [value]
        if self.world_size > 1:
            dist.broadcast_object_list(objects, src=0)
        return objects[0]

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def per_gpu_batch_size(self, batch_size: int) -> int:
        return max(1, int(batch_size // self.world_size))

    def train_sampler(self, dataset):
        if self.world_size > 1:
            return DistributedSampler(dataset)
        return None

    def valid_sampler(self, dataset):
        if self.world_size > 1:
            return DistributedSampler(dataset, shuffle=False, drop_last=False)
        return None
