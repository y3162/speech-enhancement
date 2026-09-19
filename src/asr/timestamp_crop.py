from collections.abc import Sequence
from typing import Protocol, TypeVar

SAMPLES_PER_ENCODER_FRAME = 1280


class HasOffsets(Protocol):
    @property
    def start_offset(self) -> int: ...

    @property
    def end_offset(self) -> int: ...


class HasTokenText(HasOffsets, Protocol):
    @property
    def token(self) -> str: ...


T = TypeVar("T", bound=HasOffsets)


def token_center_sample(
    start_offset: int,
    end_offset: int,
    samples_per_encoder_frame: int = SAMPLES_PER_ENCODER_FRAME,
) -> float:
    if end_offset <= start_offset:
        raise ValueError(f"end_offset must be > start_offset, got {start_offset}, {end_offset}")
    if samples_per_encoder_frame <= 0:
        raise ValueError(f"samples_per_encoder_frame must be > 0, got {samples_per_encoder_frame}")
    return (start_offset + end_offset) * samples_per_encoder_frame / 2.0


def slice_tokens_for_crop(
    tokens: Sequence[T],
    crop_start_sample: int,
    crop_end_sample: int,
    samples_per_encoder_frame: int = SAMPLES_PER_ENCODER_FRAME,
) -> list[T]:
    if crop_end_sample < crop_start_sample:
        raise ValueError(f"crop_end_sample must be >= crop_start_sample, got {crop_start_sample}, {crop_end_sample}")
    selected: list[T] = []
    for token in tokens:
        center = token_center_sample(token.start_offset, token.end_offset, samples_per_encoder_frame)
        if crop_start_sample <= center < crop_end_sample:
            selected.append(token)
    return selected


def join_token_texts(token_texts: Sequence[str]) -> str:
    return "".join(token_texts).replace("\u2581", " ").strip()


def crop_transcript(
    tokens: Sequence[HasTokenText],
    crop_start_sample: int,
    crop_end_sample: int,
    samples_per_encoder_frame: int = SAMPLES_PER_ENCODER_FRAME,
) -> str:
    selected = slice_tokens_for_crop(tokens, crop_start_sample, crop_end_sample, samples_per_encoder_frame)
    return join_token_texts([token.token for token in selected])
