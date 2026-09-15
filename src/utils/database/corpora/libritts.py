import os
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from queue import SimpleQueue
from typing import (
    Iterator,
    List,
)

from ..config import SPEECH_UTILS_CORPORA_LIBRITTS_DIR
from .audio import read_audio_stream_info
from .dto import Utterance


_IO_WORKERS = min(os.cpu_count() or 1, 64)


def utterance_generator() -> Iterator[Utterance]:
    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as executor:
        pending = set()
        paths = iter_transcript_files(SPEECH_UTILS_CORPORA_LIBRITTS_DIR)
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
    done = object()
    path_queue: SimpleQueue = SimpleQueue()

    def list_subset(subset: os.DirEntry) -> None:
        try:
            for transcript_path in _list_subset_transcripts(subset):
                path_queue.put(transcript_path)
        except Exception as exc:
            path_queue.put(exc)
        finally:
            path_queue.put(done)

    with ThreadPoolExecutor(max_workers=len(subsets)) as executor:
        for subset in subsets:
            executor.submit(list_subset, subset)
        remaining = len(subsets)
        while remaining:
            item = path_queue.get()
            if item is done:
                remaining -= 1
                continue
            if isinstance(item, Exception):
                raise item
            yield item


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

def _list_subset_transcripts(subset: os.DirEntry) -> Iterator[Path]:
    with os.scandir(subset.path) as speakers:
        for speaker in speakers:
            if not speaker.is_dir():
                continue
            with os.scandir(speaker.path) as chapters:
                for chapter in chapters:
                    if not chapter.is_dir():
                        continue
                    transcript_path = Path(chapter.path) / f"{speaker.name}_{chapter.name}.trans.tsv"
                    assert transcript_path.exists(), f"Transcript file not found: {transcript_path}"
                    yield transcript_path


def parse_transcript_file(
    transcript_file: Path,
) -> List[Utterance]:
    chapter_dir = transcript_file.parent
    speaker_dir = chapter_dir.parent
    subset_name = speaker_dir.parent.name
    speaker_id = speaker_dir.name
    chapter_id = chapter_dir.name
    prefix = f"{speaker_id}_{chapter_id}_"
    results = []
    with open(transcript_file, "r") as f:
        for line in f:
            utt_key, _, rest = line.strip().partition("\t")
            _, _, transcript = rest.partition("\t")
            audio_path = chapter_dir / (utt_key + ".wav")
            assert utt_key.startswith(prefix), f"Unexpected utterance id: {utt_key}"
            utterance_id = utt_key[len(prefix):]
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
