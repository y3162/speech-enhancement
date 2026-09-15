from typing import Iterator

from . import demand, librispeech, libritts, vctk
from .dto import Utterance


def utterance_generator() -> Iterator[Utterance]:
    yield from librispeech.utterance_generator()
    yield from libritts.utterance_generator()
    yield from vctk.utterance_generator()
    yield from demand.utterance_generator()


__all__ = [
    "utterance_generator",
]
