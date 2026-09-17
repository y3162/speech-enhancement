import numpy as np
import torch


def _anti_wrapping(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_loss_difference(
    phase_r: torch.Tensor,
    phase_g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ip_loss = torch.mean(_anti_wrapping(phase_r - phase_g))
    gd_loss = torch.mean(_anti_wrapping(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(_anti_wrapping(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))
    return ip_loss, gd_loss, iaf_loss


def phase_loss_gradient_matrix(
    phase_r: torch.Tensor,
    phase_g: torch.Tensor,
    n_fft: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim_freq = n_fft // 2 + 1
    dim_time = phase_r.size(-1)
    gd_matrix = (
        torch.triu(torch.ones(dim_freq, dim_freq, device=phase_g.device), diagonal=1)
        - torch.triu(torch.ones(dim_freq, dim_freq, device=phase_g.device), diagonal=2)
        - torch.eye(dim_freq, device=phase_g.device)
    )
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)
    iaf_matrix = (
        torch.triu(torch.ones(dim_time, dim_time, device=phase_g.device), diagonal=1)
        - torch.triu(torch.ones(dim_time, dim_time, device=phase_g.device), diagonal=2)
        - torch.eye(dim_time, device=phase_g.device)
    )
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)
    ip_loss = torch.mean(_anti_wrapping(phase_r - phase_g))
    gd_loss = torch.mean(_anti_wrapping(gd_r - gd_g))
    iaf_loss = torch.mean(_anti_wrapping(iaf_r - iaf_g))
    return ip_loss, gd_loss, iaf_loss
