from functools import partial
from types import SimpleNamespace

from src.se.mag_phase.loss import MagPhaseMetricGANLoss, attach_total, mag_phase_terms
from src.se.mag_phase.phase_loss import phase_loss_gradient_matrix


class SEMambaLoss(MagPhaseMetricGANLoss):
    def __init__(self, weights, n_fft: int) -> None:
        super().__init__(weights, generated_from="mag")
        self.phase_fn = partial(phase_loss_gradient_matrix, n_fft=n_fft)

    def generator_loss(
        self,
        pred: SimpleNamespace,
        target: SimpleNamespace,
    ) -> SimpleNamespace:
        spectral = mag_phase_terms(pred, target, self.phase_fn)
        terms = SimpleNamespace(
            magnitude=spectral.magnitude,
            phase=spectral.phase,
            complex=spectral.complex,
            consistancy=spectral.consistancy,
            time=spectral.time,
            metric=self.generator_metric(pred, target),
        )
        return attach_total(terms, self.weights)
