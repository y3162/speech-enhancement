import os
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import (
    Iterator,
    List,
)

from ..config import SPEECH_UTILS_CORPORA_VCTK_DIR
from .audio import read_audio_stream_info
from .dto import Utterance


_IO_WORKERS = min(os.cpu_count() or 1, 64)


def utterance_generator() -> Iterator[Utterance]:
    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as executor:
        pending = set()
        paths = iter_speaker_dirs(SPEECH_UTILS_CORPORA_VCTK_DIR)
        listing_done = False
        in_flight_limit = _IO_WORKERS * 2
        while pending or not listing_done:
            while not listing_done and len(pending) < in_flight_limit:
                speaker_dir = next(paths, None)
                if speaker_dir is None:
                    listing_done = True
                    break
                pending.add(executor.submit(parse_speaker_dir, speaker_dir))
            if not pending:
                break
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield from future.result()


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

def iter_speaker_dirs(root: Path) -> Iterator[Path]:
    wav_root = root / "wav48"
    with os.scandir(wav_root) as speakers:
        for speaker in speakers:
            if speaker.is_dir():
                yield Path(speaker.path)


def parse_speaker_dir(
    speaker_dir: Path,
) -> List[Utterance]:
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
            utterance_id = stem[len(prefix):]
            transcript_path = txt_dir / f"{stem}.txt"
            assert transcript_path.exists(), f"Transcript file not found: {transcript_path}"
            with open(transcript_path, "r") as f:
                transcript = f.read().strip()
            sample_rate, frames, channels = read_audio_stream_info(audio_path)
            results.append(
                Utterance(
                    corpus="VCTK",
                    subset=None,
                    chapter_id=None,
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
