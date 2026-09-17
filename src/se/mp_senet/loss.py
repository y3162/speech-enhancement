from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.se.common.phase_loss import phase_loss_difference
from src.se.common.stft import Spec


def generator_loss(
    clean: Spec,
    clean_audio: torch.Tensor,
    gen: Spec,
    enhanced_audio: torch.Tensor,
    gen_hat: Spec,
    metric_g: torch.Tensor,
    weights: SimpleNamespace,
) -> dict[str, torch.Tensor]:
    """MP-SENet generator loss. gen_hat is a re-STFT of the generated waveform; metric_g is the discriminator output."""
    ip_loss, gd_loss, iaf_loss = phase_loss_difference(clean.pha, gen.pha)
    losses = {
        "magnitude": F.mse_loss(clean.mag, gen.mag),
        "phase": ip_loss + gd_loss + iaf_loss,
        "complex": F.mse_loss(clean.com, gen.com) * 2,
        "consistency": F.mse_loss(gen.com, gen_hat.com) * 2,
        "time": F.l1_loss(clean_audio, enhanced_audio),
        "metric": F.mse_loss(metric_g.flatten(), torch.ones_like(metric_g.flatten())),
    }
    losses["total"] = (
        weights.magnitude * losses["magnitude"]
        + weights.phase * losses["phase"]
        + weights.complex * losses["complex"]
        + weights.consistency * losses["consistency"]
        + weights.time * losses["time"]
        + weights.metric * losses["metric"]
    )
    return losses
