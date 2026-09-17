from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from src.se.mag_phase.network import MagPhaseNetwork
from src.se.mag_phase.processor import MagPhaseProcessor
from src.se.training.metrics import batch_metric_target


def unwrap(module: nn.Module) -> nn.Module:
    if isinstance(module, DistributedDataParallel):
        return module.module
    return module


def log_dict(terms: SimpleNamespace) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, value in vars(terms).items():
        if torch.is_tensor(value) and value.ndim == 0:
            values[name] = float(value.detach().cpu())
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            values[name] = float(value)
    return values


class SEModel(nn.Module):
    def __init__(
        self,
        name: str,
        config: SimpleNamespace,
        model: MagPhaseNetwork,
        loss: nn.Module,
        processor: MagPhaseProcessor,
    ) -> None:
        super().__init__()
        self.name = name
        self.config = config
        self.model = model
        self.loss = loss
        self.processor = processor
        self.model.processor = processor
        self.sampling_rate = int(config.data.sampling_rate)

    @property
    def discriminator(self) -> nn.Module:
        return self.loss.discriminator

    @property
    def needs_metric_target(self) -> bool:
        return bool(self.loss.needs_metric_target)

    def forward_pair(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        processor = self.processor
        target = processor.encode(clean)
        target.audio = clean
        pred = self.model(processor.encode(noisy))
        pred.audio = processor.decode(pred)
        reconstructed = processor.encode(
            pred.audio,
            add_eps=processor.reconstruct_add_eps,
        )
        pred.reconstructed_com = reconstructed.com
        pred.reconstructed_mag = reconstructed.mag
        return pred, target

    @torch.inference_mode()
    def enhance(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        return unwrap(self.model).enhance(noisy_audio)

    def generator_loss(
        self,
        pred: SimpleNamespace,
        target: SimpleNamespace,
    ) -> SimpleNamespace:
        return self.loss.generator_loss(pred, target)

    def discriminator_loss(
        self,
        pred: SimpleNamespace,
        target: SimpleNamespace,
    ) -> SimpleNamespace:
        if not self.needs_metric_target:
            return self.loss.discriminator_loss(pred, target)
        metric_target = batch_metric_target(
            target.audio,
            pred.audio,
            self.sampling_rate,
        )
        return self.loss.discriminator_loss(
            pred,
            target,
            metric_target=metric_target,
        )
