from collections.abc import Iterator
from pathlib import Path

from ..audio import read_audio_stream_info
from ..db import corpora_root_dir
from ..schema import Utterance
from .scan import iter_in_threads, iter_speaker_chapter_transcripts


def utterance_key(utterance: Utterance) -> str:
    """LibriSpeech trans.txt id: speaker-chapter-utterance. Unique across the corpus; the last field alone is not."""
    if utterance.corpus != "LibriSpeech":
        raise ValueError(f"utterance_key expects corpus='LibriSpeech', got {utterance.corpus!r}")
    if utterance.speaker_id is None or utterance.chapter_id is None or utterance.utterance_id is None:
        raise ValueError(f"LibriSpeech utterance is missing speaker/chapter/id: {utterance.audio_path}")
    return f"{utterance.speaker_id}-{utterance.chapter_id}-{utterance.utterance_id}"


def utterance_iterator() -> Iterator[Utterance]:
    return iter_in_threads(
        iter_speaker_chapter_transcripts(
            corpora_root_dir() / "LibriSpeech",
            "{speaker}-{chapter}.trans.txt",
        ),
        parse_transcript_file,
    )


"""
LibriSpeech
└── <subset_name>
    └── <speaker_id>
        └── <chapter_id>
            ├── <speaker_id>-<chapter_id>-0000.flac
            ├── <speaker_id>-<chapter_id>-0001.flac
            ...
            └── <speaker_id>-<chapter_id>.trans.txt
"""


def parse_transcript_file(
    transcript_file: Path,
) -> list[Utterance]:
    chapter_dir = transcript_file.parent
    speaker_dir = chapter_dir.parent
    subset_name = speaker_dir.parent.name
    speaker_id = speaker_dir.name
    chapter_id = chapter_dir.name
    results = []
    with open(transcript_file) as f:
        for line in f:
            utt_key, _, transcript = line.strip().partition(" ")
            audio_path = chapter_dir / (utt_key + ".flac")
            utterance_id = utt_key.rsplit("-", 1)[-1]
            sample_rate, frames, channels = read_audio_stream_info(audio_path)
            results.append(
                Utterance(
                    corpus="LibriSpeech",
                    subset=subset_name,
                    chapter_id=chapter_id,
                    utterance_id=utterance_id,
                    speaker_id=speaker_id,
                    audio_path=audio_path,
                    sample_rate=sample_rate,
                    frames=frames,
                    channels=channels,
                    text=transcript,
                )
            )
    return results
