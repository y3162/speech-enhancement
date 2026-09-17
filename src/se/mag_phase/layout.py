import torch


def stack_mag_pha(mag: torch.Tensor, pha: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            mag.transpose(1, 2).unsqueeze(1),
            pha.transpose(1, 2).unsqueeze(1),
        ),
        dim=1,
    )


def complex_from_mag_pha(mag: torch.Tensor, pha: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (mag * torch.cos(pha), mag * torch.sin(pha)),
        dim=-1,
    )
