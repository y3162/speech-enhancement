"""S/D error frames on the clean-ASR reference, then loss-frame centers."""

from collections.abc import Sequence

from src.se.error_aware_se_mamba_pp.asr_guidance import ASR_HOP


def align_token_ids(clean_ids: list[int], noisy_ids: list[int]) -> list[tuple[str, int | None, int | None]]:
    """Token-id Levenshtein. Ties prefer C, S, D, I."""
    n = len(clean_ids)
    m = len(noisy_ids)
    inf = n + m + 1
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"
    rank = {"C": 0, "S": 1, "D": 2, "I": 3}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cands = []
            if clean_ids[i - 1] == noisy_ids[j - 1]:
                cands.append((dp[i - 1][j - 1], rank["C"], "C"))
            else:
                cands.append((dp[i - 1][j - 1] + 1, rank["S"], "S"))
            cands.append((dp[i - 1][j] + 1, rank["D"], "D"))
            cands.append((dp[i][j - 1] + 1, rank["I"], "I"))
            cost, _, move = min(cands)
            dp[i][j] = cost
            back[i][j] = move
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "C":
            ops.append(("C", i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == "S":
            ops.append(("S", i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == "D":
            ops.append(("D", i - 1, None))
            i -= 1
        elif move == "I":
            ops.append(("I", None, j - 1))
            j -= 1
        else:
            raise RuntimeError("alignment traceback failed")
    ops.reverse()
    return ops


def stft_frame_count(n_samples: int, hop: int) -> int:
    """center=True frame count. Matches torch.stft when win_length == n_fft."""
    if hop <= 0:
        raise ValueError(f"hop must be positive, got {hop}")
    if n_samples <= 0:
        return 0
    return int(n_samples) // int(hop) + 1


def error_frames_from_alignment(
    clean_tokens: Sequence[dict[str, object]],
    alignment: Sequence[tuple[str, int | None, int | None]],
) -> list[int]:
    """Union of clean-reference frames for substitution and deletion. Insertion is omitted."""
    frames: set[int] = set()
    for operation, clean_index, _noisy_index in alignment:
        if operation not in ("S", "D") or clean_index is None:
            continue
        token = clean_tokens[clean_index]
        start = int(token["start_offset"])
        end = int(token["end_offset"])
        if end < start:
            raise ValueError(f"token window ends before it starts: {start}, {end}")
        frames.update(range(start, end))
    return sorted(frames)


def error_frames_from_tokens(
    clean_tokens: Sequence[dict[str, object]],
    enhanced_tokens: Sequence[dict[str, object]],
) -> list[int]:
    """S/D frames of a clean reference aligned to an enhanced token sequence."""
    alignment = align_token_ids(
        [int(token["token_id"]) for token in clean_tokens],
        [int(token["token_id"]) for token in enhanced_tokens],
    )
    return error_frames_from_alignment(clean_tokens, alignment)


def clip_sample_interval(start: int, end: int, n_samples: int) -> tuple[int, int] | None:
    clipped_start = max(0, int(start))
    clipped_end = min(int(n_samples), int(end))
    if clipped_start >= clipped_end:
        return None
    return clipped_start, clipped_end


def sample_intervals_from_frames(
    frames: Sequence[int],
    n_samples: int,
    hop: int = ASR_HOP,
) -> list[tuple[int, int]]:
    """Encoder frame f covers [f*hop, (f+1)*hop), clipped to [0, n_samples)."""
    if hop <= 0:
        raise ValueError(f"hop must be positive, got {hop}")
    intervals: list[tuple[int, int]] = []
    for frame in frames:
        clipped = clip_sample_interval(int(frame) * hop, (int(frame) + 1) * hop, n_samples)
        if clipped is not None:
            intervals.append(clipped)
    return intervals


def fill_sample_error(n_samples: int, intervals: Sequence[tuple[int, int]]) -> list[bool]:
    """Binary union. Overlapping intervals stay 1."""
    error = [False] * int(n_samples)
    for start, end in intervals:
        clipped = clip_sample_interval(start, end, n_samples)
        if clipped is None:
            continue
        for index in range(clipped[0], clipped[1]):
            error[index] = True
    return error


def alignment_counts(
    alignment: Sequence[tuple[str, int | None, int | None]],
) -> dict[str, int]:
    counts = {"C": 0, "S": 0, "D": 0, "I": 0}
    for operation, _clean_index, _noisy_index in alignment:
        if operation not in counts:
            raise ValueError(f"unknown alignment operation {operation!r}")
        counts[operation] += 1
    return counts


def boundary_sd_counts(
    clean_tokens: Sequence[dict[str, object]],
    alignment: Sequence[tuple[str, int | None, int | None]],
    encoded_length: int,
) -> tuple[int, int]:
    """S/D tokens whose clean window overlaps the first or last encoder frame."""
    touched = 0
    total = 0
    last = encoded_length - 1
    for operation, clean_index, _noisy_index in alignment:
        if operation not in ("S", "D") or clean_index is None:
            continue
        total += 1
        token = clean_tokens[clean_index]
        start = int(token["start_offset"])
        end = int(token["end_offset"])
        overlaps_first = encoded_length > 0 and start <= 0 < end
        overlaps_last = encoded_length > 0 and start <= last < end
        if overlaps_first or overlaps_last:
            touched += 1
    return touched, total
