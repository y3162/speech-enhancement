from types import SimpleNamespace

from src.se.mag_phase.loss import MagPhaseMetricGANLoss, attach_total, mag_phase_terms
from src.se.mag_phase.phase_loss import phase_loss_difference


class MPSENetLoss(MagPhaseMetricGANLoss):
    def __init__(self, weights) -> None:
        super().__init__(weights, generated_from="reconstructed_mag")

    def generator_loss(
        self,
        pred: SimpleNamespace,
        target: SimpleNamespace,
    ) -> SimpleNamespace:
        spectral = mag_phase_terms(pred, target, phase_loss_difference)
        terms = SimpleNamespace(
            magnitude=spectral.magnitude,
            phase=spectral.phase,
            complex=spectral.complex,
            stft=spectral.consistancy,
            time=spectral.time,
            metric=self.generator_metric(pred, target),
        )
        return attach_total(terms, self.weights)
