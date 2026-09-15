import numpy as np

from ..database.connection import Connection
from .config import NoiseConfig
from .parser import parse
from .versions import v1_0_0


def generate(
    clean: np.ndarray,
    sample_rate: int,
    config: NoiseConfig,
    connection: Connection,
) -> np.ndarray:
    if clean.ndim != 2 or clean.shape[0] < 1 or clean.shape[1] < 1:
        raise ValueError(
            "clean must be a non-empty 2D array (frames, channels), "
            f"got shape {clean.shape}"
        )

    pipeline = parse(config.json)
    if isinstance(pipeline, v1_0_0.Pipeline):
        return v1_0_0.generate(clean, sample_rate, pipeline, connection)
    raise TypeError(f"Unsupported pipeline type: {type(pipeline).__name__}")
