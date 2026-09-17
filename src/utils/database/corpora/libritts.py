from collections.abc import Iterator
from pathlib import Path

from ..constants import SPEECH_UTILS_CORPORA_LIBRITTS_DIR
from .audio import read_audio_stream_info
from .dto import Utterance
from .scan import iter_in_threads, iter_speaker_chapter_transcripts


def utterance_iterator() -> Iterator[Utterance]:
    return iter_in_threads(
        iter_speaker_chapter_transcripts(
            SPEECH_UTILS_CORPORA_LIBRITTS_DIR,
            "{speaker}_{chapter}.trans.tsv",
        ),
        parse_transcript_file,
    )


"""
LibriTTS
└── <subset_name>
    └── <speaker_id>
        └── <chapter_id>
            ├── <speaker_id>_<chapter_id>_<utterance_id>.wav
            ├── <speaker_id>_<chapter_id>_<utterance_id>.normalized.txt
            ├── <speaker_id>_<chapter_id>_<utterance_id>.original.txt
            └── <speaker_id>_<chapter_id>.trans.tsv
"""


def parse_transcript_file(
    transcript_file: Path,
) -> list[Utterance]:
    chapter_dir = transcript_file.parent
    speaker_dir = chapter_dir.parent
    subset_name = speaker_dir.parent.name
    speaker_id = speaker_dir.name
    chapter_id = chapter_dir.name
    prefix = f"{speaker_id}_{chapter_id}_"
    results = []
    with open(transcript_file) as f:
        for line in f:
            utt_key, _, rest = line.strip().partition("\t")
            _, _, transcript = rest.partition("\t")
            audio_path = chapter_dir / (utt_key + ".wav")
            assert utt_key.startswith(prefix), f"Unexpected utterance id: {utt_key}"
            utterance_id = utt_key[len(prefix) :]
            sample_rate, frames, channels = read_audio_stream_info(audio_path)
            results.append(
                Utterance(
                    corpus="LibriTTS",
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
