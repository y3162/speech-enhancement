from collections.abc import Iterator
from pathlib import Path

from ..constants import SPEECH_UTILS_CORPORA_LIBRISPEECH_DIR
from .audio import read_audio_stream_info
from .dto import Utterance
from .scan import iter_in_threads, iter_speaker_chapter_transcripts


def utterance_iterator() -> Iterator[Utterance]:
    return iter_in_threads(
        iter_speaker_chapter_transcripts(
            SPEECH_UTILS_CORPORA_LIBRISPEECH_DIR,
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
