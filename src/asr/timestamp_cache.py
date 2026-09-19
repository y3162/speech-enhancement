"""Offline Frozen-ASR timestamp cache.

python -m src.asr.timestamp_cache --splits train-clean-100 --output PATH
python -m src.asr.timestamp_cache --splits train-clean-100 --output PATH --check
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.asr.timestamp_crop import slice_tokens_for_crop
from src.data.audio import read_audio_segment
from src.data.corpora.librispeech import utterance_key
from src.data.db import Connection, fetch_utterances, metadata_path
from src.data.schema import Utterance

_LIBRISPEECH_CORPUS = "LibriSpeech"


@dataclass(frozen=True)
class CachedToken:
    token_id: int
    token: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class TimestampRecord:
    utterance_key: str
    text: str
    samples_per_encoder_frame: int
    tokens: list[CachedToken]


def timestamp_record_to_json(record: TimestampRecord) -> dict[str, object]:
    return {
        "utterance_key": record.utterance_key,
        "text": record.text,
        "samples_per_encoder_frame": record.samples_per_encoder_frame,
        "tokens": [
            {
                "token_id": token.token_id,
                "token": token.token,
                "start_frame": token.start_offset,
                "end_frame": token.end_offset,
            }
            for token in record.tokens
        ],
    }


def timestamp_record_from_json(payload: object) -> TimestampRecord:
    if not isinstance(payload, dict):
        raise ValueError(f"timestamp record must be an object, got {type(payload).__name__}")
    tokens_raw = payload.get("tokens")
    if not isinstance(tokens_raw, list):
        raise ValueError("timestamp record is missing tokens")
    tokens = []
    for item in tokens_raw:
        if not isinstance(item, dict):
            raise ValueError("timestamp token must be an object")
        tokens.append(
            CachedToken(
                token_id=int(item["token_id"]),
                token=str(item["token"]),
                start_offset=int(item["start_frame"]),
                end_offset=int(item["end_frame"]),
            )
        )
    return TimestampRecord(
        utterance_key=str(payload["utterance_key"]),
        text=str(payload["text"]),
        samples_per_encoder_frame=int(payload["samples_per_encoder_frame"]),
        tokens=tokens,
    )


def load_timestamp_cache(path: Path) -> dict[str, TimestampRecord]:
    cache: dict[str, TimestampRecord] = {}
    duplicates: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = timestamp_record_from_json(json.loads(line))
            if record.utterance_key in cache:
                duplicates.append(record.utterance_key)
            cache[record.utterance_key] = record
    if duplicates:
        shown = ", ".join(duplicates[:5])
        raise ValueError(f"duplicate timestamp cache keys ({len(duplicates)}): {shown}")
    return cache


def require_timestamp_record(cache: dict[str, TimestampRecord], key: str) -> TimestampRecord:
    try:
        return cache[key]
    except KeyError:
        raise KeyError(f"timestamp cache miss for utterance_key={key!r}") from None


def missing_timestamp_keys(cache: dict[str, TimestampRecord], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if key not in cache]


def crop_token_ids(record: TimestampRecord, crop_start: int, crop_end: int) -> list[int]:
    selected = slice_tokens_for_crop(
        record.tokens,
        crop_start,
        crop_end,
        record.samples_per_encoder_frame,
    )
    return [token.token_id for token in selected]


def _fetch_split_utterances(splits: Sequence[str]) -> list[Utterance]:
    with Connection(metadata_path(), read_only=True) as connection:
        return fetch_utterances(connection, _LIBRISPEECH_CORPUS, splits)


def _coverage(cache: dict[str, TimestampRecord], utterances: Sequence[Utterance]) -> tuple[int, int, int, list[str]]:
    keys = [utterance_key(utterance) for utterance in utterances]
    missing = missing_timestamp_keys(cache, keys)
    return len(keys), len(cache), len(missing), missing


def _print_coverage(n_utterances: int, n_entries: int, n_missing: int, missing: Sequence[str]) -> None:
    print(
        f"utterances={n_utterances} cache_entries={n_entries} missing={n_missing} duplicates=0",
        flush=True,
    )
    if missing:
        print("missing_examples " + " ".join(missing[:5]), flush=True)


def _load_clean_waveform(utterance: Utterance, sampling_rate: int) -> np.ndarray:
    if utterance.sample_rate is None or utterance.sample_rate != sampling_rate:
        raise ValueError(f"{utterance.audio_path} sample_rate {utterance.sample_rate} does not match {sampling_rate}")
    if utterance.frames is None or utterance.frames < 1:
        raise ValueError(f"{utterance.audio_path} has invalid frames")
    audio = read_audio_segment(utterance.audio_path, 0, utterance.frames)
    if audio.ndim != 2 or audio.shape[0] < 1 or audio.shape[1] < 1:
        raise ValueError(f"{utterance.audio_path} has invalid shape {audio.shape}")
    if audio.shape[1] != 1:
        raise ValueError(f"{utterance.audio_path} must be mono, got channels={audio.shape[1]}")
    return np.ascontiguousarray(audio[:, 0])


def _check_cache(path: Path, splits: Sequence[str], limit: int | None) -> int:
    utterances = _fetch_split_utterances(splits)
    if limit is not None:
        utterances = utterances[:limit]
    cache = load_timestamp_cache(path)
    n_utterances, n_entries, n_missing, missing = _coverage(cache, utterances)
    _print_coverage(n_utterances, n_entries, n_missing, missing)
    return 1 if n_missing else 0


def _generate_cache(
    path: Path,
    splits: Sequence[str],
    sampling_rate: int,
    batch_size: int,
    device_name: str,
    limit: int | None,
    overwrite: bool,
    skip_existing: bool,
) -> None:
    import torch

    from src.asr.parakeet_tdt_0_6b_v2 import ParakeetTDT06BV2

    utterances = _fetch_split_utterances(splits)
    if limit is not None:
        utterances = utterances[:limit]
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, TimestampRecord] = {}
    if path.is_file():
        if overwrite:
            path.unlink()
        elif skip_existing:
            existing = load_timestamp_cache(path)
        else:
            raise FileExistsError(f"{path} exists; pass --overwrite or --skip-existing")
    pending = [utterance for utterance in utterances if utterance_key(utterance) not in existing]
    print(
        f"cache {path}: {len(utterances)} utterances, {len(existing)} existing, {len(pending)} to recognize",
        flush=True,
    )
    if not pending:
        cache = existing if existing else load_timestamp_cache(path)
        n_utterances, n_entries, n_missing, missing = _coverage(cache, utterances)
        _print_coverage(n_utterances, n_entries, n_missing, missing)
        return

    device = torch.device(device_name)
    asr = ParakeetTDT06BV2().to(device)
    asr.eval()
    written = 0
    with open(path, "a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            waves = [_load_clean_waveform(utterance, sampling_rate) for utterance in chunk]
            max_len = max(int(wave.shape[0]) for wave in waves)
            batch = torch.zeros(len(waves), max_len, device=device, dtype=torch.float32)
            lengths = torch.zeros(len(waves), device=device, dtype=torch.int64)
            for i, wave in enumerate(waves):
                length = int(wave.shape[0])
                batch[i, :length] = torch.from_numpy(wave)
                lengths[i] = length
            recognitions = asr.recognize(batch, lengths)
            for utterance, recognition in zip(chunk, recognitions):
                record = TimestampRecord(
                    utterance_key=utterance_key(utterance),
                    text=recognition.text,
                    samples_per_encoder_frame=recognition.samples_per_encoder_frame,
                    tokens=[
                        CachedToken(
                            token_id=token.token_id,
                            token=token.token,
                            start_offset=token.start_offset,
                            end_offset=token.end_offset,
                        )
                        for token in recognition.tokens
                    ],
                )
                handle.write(json.dumps(timestamp_record_to_json(record), ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            done = min(start + batch_size, len(pending))
            print(f"recognized {done}/{len(pending)}", flush=True)
    cache = load_timestamp_cache(path)
    n_utterances, n_entries, n_missing, missing = _coverage(cache, utterances)
    _print_coverage(n_utterances, n_entries, n_missing, missing)
    print(f"wrote {written} new records to {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or check a LibriSpeech Parakeet timestamp cache")
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.overwrite and args.skip_existing:
        parser.error("--overwrite and --skip-existing cannot be combined")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.check:
        raise SystemExit(_check_cache(args.output, args.splits, args.limit))
    _generate_cache(
        args.output,
        args.splits,
        args.sampling_rate,
        args.batch_size,
        args.device,
        args.limit,
        args.overwrite,
        args.skip_existing,
    )


if __name__ == "__main__":
    main()
