import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.se.mag_phase.processor import MagPhaseProcessor
from src.se.model import SEModel


CANONICAL_NAMES = ("mp_senet", "se_mamba", "se_mamba_pp")


def canonical_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key not in CANONICAL_NAMES:
        raise ValueError(
            f"unknown SE name {name!r}; available: {', '.join(CANONICAL_NAMES)}"
        )
    return key


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def to_dict(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: to_dict(val) for key, val in vars(value).items()}
    if isinstance(value, dict):
        return {key: to_dict(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    return value


def load_config_file(path: str | Path) -> SimpleNamespace:
    with open(path, encoding="utf-8") as f:
        return to_namespace(json.load(f))


def get(name: str, config: SimpleNamespace) -> SEModel:
    name = canonical_name(name)
    stft = config.data.stft
    processor = MagPhaseProcessor(
        n_fft=stft.n_fft,
        hop_size=stft.hop_size,
        win_size=stft.win_size,
        compress_factor=stft.compress_factor,
        stft_kind="mp_senet" if name == "mp_senet" else "abs_angle",
        reconstruct_add_eps=name != "mp_senet",
    )

    if name == "mp_senet":
        from src.se.mp_senet.loss import MPSENetLoss
        from src.se.mp_senet.model import MPNet

        model = MPNet(config.model, stft.n_fft)
        loss = MPSENetLoss(config.train.loss)
    elif name == "se_mamba":
        from src.se.se_mamba.loss import SEMambaLoss
        from src.se.se_mamba.model import SEMamba

        model = SEMamba(config.model, stft.n_fft)
        loss = SEMambaLoss(config.train.loss, n_fft=stft.n_fft)
    else:
        from src.se.se_mamba_pp.loss import SEMambaPPLoss
        from src.se.se_mamba_pp.model import SEMambaPP

        model = SEMambaPP(config.model, stft.n_fft)
        loss = SEMambaPPLoss(
            config.train.loss,
            n_fft=stft.n_fft,
            sampling_rate=config.data.sampling_rate,
        )

    return SEModel(
        name=name,
        config=config,
        model=model,
        loss=loss,
        processor=processor,
    )
