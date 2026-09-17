import torch
import torch.nn.functional as F

from src.se.common.phase_loss import phase_loss_difference
from src.se.common.stft import Spec


def generator_loss(
    clean: Spec,
    clean_audio: torch.Tensor,
    gen: Spec,
    gen_audio: torch.Tensor,
    gen_hat: Spec,
    metric_g: torch.Tensor,
    w,
) -> dict[str, torch.Tensor]:
    """MP-SENet generator loss. gen_hat is a re-STFT of the generated waveform; metric_g is the discriminator output."""
    ip_loss, gd_loss, iaf_loss = phase_loss_difference(clean.pha, gen.pha)
    losses = {
        "magnitude": F.mse_loss(clean.mag, gen.mag),
        "phase": ip_loss + gd_loss + iaf_loss,
        "complex": F.mse_loss(clean.com, gen.com) * 2,
        "consistency": F.mse_loss(gen.com, gen_hat.com) * 2,
        "time": F.l1_loss(clean_audio, gen_audio),
        "metric": F.mse_loss(metric_g.flatten(), torch.ones_like(metric_g.flatten())),
    }
    losses["total"] = (
        w.magnitude * losses["magnitude"]
        + w.phase * losses["phase"]
        + w.complex * losses["complex"]
        + w.consistency * losses["consistency"]
        + w.time * losses["time"]
        + w.metric * losses["metric"]
    )
    return losses
