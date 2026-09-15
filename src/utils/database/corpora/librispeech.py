import os
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from typing import (
    Iterator,
    List,
)

from ..constants import SPEECH_UTILS_CORPORA_LIBRISPEECH_DIR
from .audio import read_audio_stream_info
from .dto import Utterance


_IO_WORKERS = min(os.cpu_count() or 1, 64)

def utterance_iterator() -> Iterator[Utterance]:
    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as executor:
        pending = set()
        paths = iter_transcript_files(SPEECH_UTILS_CORPORA_LIBRISPEECH_DIR)
        listing_done = False
        in_flight_limit = _IO_WORKERS * 2
        while pending or not listing_done:
            while not listing_done and len(pending) < in_flight_limit:
                transcript_file = next(paths, None)
                if transcript_file is None:
                    listing_done = True
                    break
                pending.add(executor.submit(parse_transcript_file, transcript_file))
            if not pending:
                break
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield from future.result()


def iter_transcript_files(root: Path) -> Iterator[Path]:
    subsets = []
    with os.scandir(root) as entries:
        for subset in entries:
            if subset.is_dir() and subset.name.startswith(("train-", "dev-", "test-")):
                subsets.append(subset)
    if not subsets:
        return
    with ThreadPoolExecutor(max_workers=len(subsets)) as executor:
        futures = [executor.submit(_list_subset_transcripts, subset) for subset in subsets]
        for future in as_completed(futures):
            yield from future.result()


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

def _list_subset_transcripts(subset: os.DirEntry) -> list[Path]:
    paths = []
    with os.scandir(subset.path) as speakers:
        for speaker in speakers:
            if not speaker.is_dir():
                continue
            with os.scandir(speaker.path) as chapters:
                for chapter in chapters:
                    if not chapter.is_dir():
                        continue
                    transcript_path = Path(chapter.path) / f"{speaker.name}-{chapter.name}.trans.txt"
                    assert transcript_path.exists(), f"Transcript file not found: {transcript_path}"
                    paths.append(transcript_path)
    return paths


def parse_transcript_file(
    transcript_file: Path,
) -> List[Utterance]:
    chapter_dir = transcript_file.parent
    speaker_dir = chapter_dir.parent
    subset_name = speaker_dir.parent.name
    speaker_id = speaker_dir.name
    chapter_id = chapter_dir.name
    results = []
    with open(transcript_file, "r") as f:
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
