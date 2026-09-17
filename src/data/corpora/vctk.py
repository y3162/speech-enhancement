import os
from collections.abc import Iterator
from pathlib import Path

from ..audio import read_audio_stream_info
from ..db import corpora_root_dir
from ..schema import Utterance
from .scan import iter_in_threads


def utterance_iterator() -> Iterator[Utterance]:
    speaker_dirs = []
    with os.scandir(corpora_root_dir() / "VCTK" / "wav48") as speakers:
        for speaker in speakers:
            if speaker.is_dir():
                speaker_dirs.append(Path(speaker.path))
    return iter_in_threads(speaker_dirs, parse_speaker_dir)


"""
VCTK
├── wav48
│   └── <speaker_id>
│       ├── <speaker_id>_001.wav
│       └── ...
└── txt
    └── <speaker_id>
        ├── <speaker_id>_001.txt
        └── ...
"""


def parse_speaker_dir(
    speaker_dir: Path,
) -> list[Utterance]:
    speaker_id = speaker_dir.name
    txt_dir = speaker_dir.parent.parent / "txt" / speaker_id
    results = []
    with os.scandir(speaker_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".wav"):
                continue
            audio_path = Path(entry.path)
            stem = audio_path.stem
            prefix = f"{speaker_id}_"
            assert stem.startswith(prefix), f"Unexpected utterance id: {stem}"
            utterance_id = stem[len(prefix) :]
            transcript_path = txt_dir / f"{stem}.txt"
            assert transcript_path.exists(), f"Transcript file not found: {transcript_path}"
            with open(transcript_path) as f:
                transcript = f.read().strip()
            sample_rate, frames, channels = read_audio_stream_info(audio_path)
            results.append(
                Utterance(
                    corpus="VCTK",
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
