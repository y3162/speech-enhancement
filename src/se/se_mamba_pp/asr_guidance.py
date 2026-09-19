import torch
import torch.nn as nn

STFT_HOP = 100
ASR_HOP = 1280
ASR_DIM = 1024
ENCODER_LAYER = 15


def nearest_align(
    hidden: torch.Tensor,
    n_stft: int,
    asr_lengths: torch.Tensor | None = None,
    stft_hop: int = STFT_HOP,
    asr_hop: int = ASR_HOP,
) -> torch.Tensor:
    """Align [B, T_asr, D] onto STFT frames by nearest sample centers 100j vs 1280i."""
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [B, T_asr, D], got {tuple(hidden.shape)}")
    batch_size, n_asr, dim = hidden.shape
    device = hidden.device
    stft_centers = torch.arange(n_stft, device=device, dtype=hidden.dtype) * stft_hop
    asr_centers = torch.arange(n_asr, device=device, dtype=hidden.dtype) * asr_hop
    distance = (stft_centers[:, None] - asr_centers[None, :]).abs()
    if asr_lengths is None:
        indices = distance.argmin(dim=1)
        return hidden.index_select(1, indices)
    valid = torch.arange(n_asr, device=device)[None, :] < asr_lengths.to(device)[:, None]
    batched = distance[None, :, :].expand(batch_size, -1, -1)
    batched = batched.masked_fill(~valid[:, None, :], torch.inf)
    indices = batched.argmin(dim=-1)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, dim)
    return hidden.gather(1, gather_index)


def broadcast_frequency(projected: torch.Tensor, n_freq: int) -> torch.Tensor:
    """[B, T, C] -> [B, C, T, F]."""
    channel_first = projected.transpose(1, 2).unsqueeze(-1)
    return channel_first.expand(-1, -1, -1, n_freq).contiguous()


def project_asr_features(
    proj: nn.Linear,
    hidden: torch.Tensor,
    n_stft: int,
    asr_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project [B, ASR_DIM, T_asr] to [B, T_stft, guidance_dim] with nearest alignment."""
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [B, {ASR_DIM}, T_asr], got {tuple(hidden.shape)}")
    projected = proj(hidden.transpose(1, 2))
    return nearest_align(projected, n_stft, asr_lengths)


def fuse_asr_features(fuse: nn.Conv2d, se_feat: torch.Tensor, aligned: torch.Tensor) -> torch.Tensor:
    """Concat DenseEncoder [B, C, T, F] with aligned ASR [B, T, G] and project back to C."""
    asr_feat = broadcast_frequency(aligned, se_feat.size(-1))
    return fuse(torch.cat([se_feat, asr_feat], dim=1))
