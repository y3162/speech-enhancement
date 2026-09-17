from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.se.mag_phase.loss import attach_total, mag_phase_terms
from src.se.mag_phase.phase_loss import phase_loss_gradient_matrix
from src.se.se_mamba_pp.discriminator import (
    SEMambaPPDiscriminator,
    feature_loss,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
)
from src.se.se_mamba_pp.mel import MultiScaleMelSpectrogramLoss


class SEMambaPPLoss(nn.Module):
    needs_metric_target = False

    def __init__(self, weights, n_fft: int, sampling_rate: int) -> None:
        super().__init__()
        self.weights = weights
        self.phase_fn = partial(phase_loss_gradient_matrix, n_fft=n_fft)
        self.discriminator = SEMambaPPDiscriminator()
        self.mel = MultiScaleMelSpectrogramLoss(sampling_rate=sampling_rate)

    def set_discriminator(self, discriminator: nn.Module) -> None:
        self.discriminator = discriminator

    def _disc_out(self, target_audio: torch.Tensor, pred_audio: torch.Tensor) -> SimpleNamespace:
        return self.discriminator(target_audio.unsqueeze(1), pred_audio.unsqueeze(1))

    def generator_loss(
        self,
        pred: SimpleNamespace,
        target: SimpleNamespace,
    ) -> SimpleNamespace:
        spectral = mag_phase_terms(pred, target, self.phase_fn)
        disc_out = self._disc_out(target.audio, pred.audio)
        terms = SimpleNamespace(
            magnitude=spectral.magnitude,
            phase=spectral.phase,
            complex=spectral.complex,
            consistancy=spectral.consistancy,
            adv_g=lsgan_generator_loss(disc_out.cqt_fake) + lsgan_generator_loss(disc_out.mrd_fake),
            fm_g=feature_loss(disc_out.cqt_fmap_real, disc_out.cqt_fmap_fake)
            + feature_loss(disc_out.mrd_fmap_real, disc_out.mrd_fmap_fake),
            mel=self.mel(target.audio.unsqueeze(1), pred.audio.unsqueeze(1)),
        )
        return attach_total(terms, self.weights)

    def discriminator_loss(
        self,
        pred: SimpleNamespace,
        target: SimpleNamespace,
    ) -> SimpleNamespace:
        disc_out = self._disc_out(target.audio, pred.audio.detach())
        cqt = lsgan_discriminator_loss(disc_out.cqt_real, disc_out.cqt_fake)
        mrd = lsgan_discriminator_loss(disc_out.mrd_real, disc_out.mrd_fake)
        terms = SimpleNamespace(cqt=cqt, mrd=mrd)
        terms.total = cqt + mrd
        return terms
