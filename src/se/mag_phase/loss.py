from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.se.mag_phase.metric_discriminator import MetricDiscriminator
from src.se.mag_phase.terms import (
    complex_loss,
    consistency_loss,
    magnitude_loss,
    time_loss,
    weighted_total,
)


def mag_phase_terms(
    pred: SimpleNamespace,
    target: SimpleNamespace,
    phase_fn,
) -> SimpleNamespace:
    ip_loss, gd_loss, iaf_loss = phase_fn(target.pha, pred.pha)
    return SimpleNamespace(
        magnitude=magnitude_loss(pred.mag, target.mag),
        phase=ip_loss + gd_loss + iaf_loss,
        complex=complex_loss(pred.com, target.com),
        consistancy=consistency_loss(pred.com, pred.reconstructed_com),
        time=time_loss(pred.audio, target.audio),
    )


def attach_total(terms: SimpleNamespace, weights) -> SimpleNamespace:
    terms.total = weighted_total(terms, weights)
    return terms


def _ones_per_sample(mag: torch.Tensor) -> torch.Tensor:
    return torch.ones(mag.size(0), device=mag.device, dtype=mag.dtype)


class MagPhaseMetricGANLoss(nn.Module):
    needs_metric_target = True

    def __init__(self, weights, generated_from: str) -> None:
        super().__init__()
        if generated_from not in ("reconstructed_mag", "mag"):
            raise ValueError(
                "generated_from must be 'reconstructed_mag' or 'mag', "
                f"got {generated_from!r}"
            )
        self.weights = weights
        self.generated_from = generated_from
        self.discriminator = MetricDiscriminator()

    def set_discriminator(self, discriminator: nn.Module) -> None:
        self.discriminator = discriminator

    def generated_mag(self, pred: SimpleNamespace) -> torch.Tensor:
        if self.generated_from == "reconstructed_mag":
            return pred.reconstructed_mag
        return pred.mag

    def generator_metric(self, pred: SimpleNamespace, target: SimpleNamespace) -> torch.Tensor:
        metric = self.discriminator(target.mag, self.generated_mag(pred))
        return F.mse_loss(metric.flatten(), _ones_per_sample(target.mag))

    def discriminator_loss(
        self,
        pred: SimpleNamespace,
        target: SimpleNamespace,
        metric_target: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        metric_r = self.discriminator(target.mag, target.mag)
        metric_g = self.discriminator(target.mag, self.generated_mag(pred).detach())
        real = F.mse_loss(_ones_per_sample(target.mag), metric_r.flatten())
        if metric_target is None:
            fake = torch.zeros((), device=real.device, dtype=real.dtype)
        else:
            fake = F.mse_loss(metric_target.to(device=metric_g.device, dtype=metric_g.dtype), metric_g.flatten())
        terms = SimpleNamespace(real=real, fake=fake)
        terms.total = real + fake
        return terms
