import numpy as np

from ..database.connection import Connection
from .config import NoiseConfig
from .versions import v1_0_0


def generate(
    clean: np.ndarray,
    sample_rate: int,
    config: NoiseConfig,
    connection: Connection,
) -> np.ndarray:
    if clean.ndim != 2 or clean.shape[0] < 1 or clean.shape[1] < 1:
        raise ValueError(f"clean must be a non-empty 2D array (frames, channels), got shape {clean.shape}")

    version = config.json["version"]
    if version != "1.0":
        raise ValueError(f"Unsupported noise config version: {version!r}")
    return v1_0_0.generate(
        clean,
        sample_rate,
        v1_0_0.parse(config.json),
        connection,
    )
