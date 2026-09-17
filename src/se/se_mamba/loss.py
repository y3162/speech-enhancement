import torch
import torch.nn.functional as F

from src.se.common.phase_loss import phase_loss_gradient_matrix
from src.se.common.stft import Spec


def generator_loss(
    clean: Spec,
    clean_audio: torch.Tensor,
    gen: Spec,
    gen_audio: torch.Tensor,
    gen_hat: Spec,
    metric_g: torch.Tensor,
    w,
    n_fft: int,
) -> dict[str, torch.Tensor]:
    """SEMamba generator loss. Differs from MP-SENet by using the gradient-matrix phase loss."""
    ip_loss, gd_loss, iaf_loss = phase_loss_gradient_matrix(clean.pha, gen.pha, n_fft)
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
