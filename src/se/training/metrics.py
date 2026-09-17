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


def batch_metric_target(
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor | None:
    scores = np.array(
        [
            _pesq_one(row.detach().cpu().numpy(), est.detach().cpu().numpy(), sample_rate)
            for row, est in zip(clean, enhanced)
        ]
    )
    if np.any(scores < 0):
        return None
    return torch.tensor((scores - 1.0) / 3.5, dtype=torch.float32)


def mean_pesq(
    refs: list[np.ndarray],
    ests: list[np.ndarray],
    sample_rate: int,
    num_workers: int,
) -> tuple[float, int]:
    if not refs:
        return 0.0, 0
    payloads = list(zip(refs, ests))
    workers = max(1, int(num_workers))
    if workers == 1 or len(payloads) == 1:
        scores = [_pesq_one(ref, est, sample_rate) for ref, est in payloads]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            scores = list(
                executor.map(
                    lambda pair: _pesq_one(pair[0], pair[1], sample_rate),
                    payloads,
                )
            )
    valid = [score for score in scores if score >= 0]
    if not valid:
        return 0.0, 0
    return float(sum(valid)), len(valid)
