"""Bind this process to LOCAL_RANK's GPU. Import before torch so CUDA sees one device."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def pin_local_cuda_device() -> None:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[int(local_rank)]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = local_rank


pin_local_cuda_device()
