import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")

IO_WORKERS = min(os.cpu_count() or 1, 64)


def iter_in_threads(
    items: Iterable[T],
    worker: Callable[[T], list[U]],
    *,
    max_workers: int = IO_WORKERS,
) -> Iterator[U]:
    item_iter = iter(items)
    in_flight_limit = max_workers * 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = set()
        listing_done = False
        while pending or not listing_done:
            while not listing_done and len(pending) < in_flight_limit:
                try:
                    item = next(item_iter)
                except StopIteration:
                    listing_done = True
                    break
                pending.add(executor.submit(worker, item))
            if not pending:
                break
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield from future.result()


def iter_speaker_chapter_transcripts(root: Path, name_format: str) -> Iterator[Path]:
    subsets = []
    with os.scandir(root) as entries:
        for subset in entries:
            if subset.is_dir() and subset.name.startswith(("train-", "dev-", "test-")):
                subsets.append(subset)
    if not subsets:
        return
    with ThreadPoolExecutor(max_workers=len(subsets)) as executor:
        futures = [executor.submit(_list_subset_transcripts, subset, name_format) for subset in subsets]
        for future in as_completed(futures):
            yield from future.result()


def _list_subset_transcripts(subset: os.DirEntry, name_format: str) -> list[Path]:
    paths = []
    with os.scandir(subset.path) as speakers:
        for speaker in speakers:
            if not speaker.is_dir():
                continue
            with os.scandir(speaker.path) as chapters:
                for chapter in chapters:
                    if not chapter.is_dir():
                        continue
                    transcript_path = Path(chapter.path) / name_format.format(
                        speaker=speaker.name,
                        chapter=chapter.name,
                    )
                    assert transcript_path.exists(), f"Transcript file not found: {transcript_path}"
                    paths.append(transcript_path)
    return paths
