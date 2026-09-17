from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


def _pesq_one(clean: np.ndarray, enhanced: np.ndarray, sample_rate: int) -> float:
    try:
        from pesq import pesq
    except ImportError as exc:
        raise ImportError(
            "PESQ is required for validation and MetricGAN training. "
            "Install it with: pip install pesq"
        ) from exc
    try:
        return float(pesq(sample_rate, clean, enhanced, "wb"))
    except Exception:
        return -1.0


def pesq_scores(
    refs: list[np.ndarray],
    ests: list[np.ndarray],
    sample_rate: int,
    num_workers: int = 1,
) -> list[float]:
    """Per-utterance wideband PESQ. Failed utterances are -1."""
    pairs = list(zip(refs, ests))
    workers = max(1, int(num_workers))
    if workers == 1 or len(pairs) <= 1:
        return [_pesq_one(ref, est, sample_rate) for ref, est in pairs]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda p: _pesq_one(p[0], p[1], sample_rate), pairs))


def pesq_batch_target(
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor | None:
    """MetricGAN regression target: PESQ mapped to [0, 1]. None if any utterance fails."""
    scores = np.array(
        pesq_scores(
            [row.detach().cpu().numpy() for row in clean],
            [row.detach().cpu().numpy() for row in enhanced],
            sample_rate,
        )
    )
    if np.any(scores < 0):
        return None
    return torch.tensor((scores - 1.0) / 3.5, dtype=torch.float32)


def pesq_sum(
    refs: list[np.ndarray],
    ests: list[np.ndarray],
    sample_rate: int,
    num_workers: int,
) -> tuple[float, int]:
    """Validation: sum and count of successful PESQ scores."""
    valid = [s for s in pesq_scores(refs, ests, sample_rate, num_workers) if s >= 0]
    return float(sum(valid)), len(valid)
