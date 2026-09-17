from types import SimpleNamespace

import torch
import torch.nn.functional as F


def magnitude_loss(pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(target_mag, pred_mag)


def complex_loss(pred_com: torch.Tensor, target_com: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(target_com, pred_com) * 2


def consistency_loss(
    pred_com: torch.Tensor,
    reconstructed_com: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(pred_com, reconstructed_com) * 2


def time_loss(pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(target_audio, pred_audio)


def weighted_total(terms: SimpleNamespace, weights) -> torch.Tensor:
    total = None
    for name, value in vars(terms).items():
        if name == "total" or value is None:
            continue
        weight = getattr(weights, name, None)
        if weight is None:
            continue
        term = value * weight
        total = term if total is None else total + term
    if total is None:
        raise ValueError("no weighted loss terms were found")
    return total
