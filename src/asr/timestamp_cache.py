"""LibriSpeech Frozen-ASR timestamps in metadata.duckdb.

python -m src.asr.timestamp_cache --splits train-clean-100 train-clean-360 dev-clean --output PATH.jsonl
python -m src.asr.timestamp_cache --import PATH.jsonl
python -m src.asr.timestamp_cache --splits train-clean-100 train-clean-360 dev-clean
python -m src.asr.timestamp_cache --splits train-clean-100 train-clean-360 dev-clean --check
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.asr.timestamp_crop import slice_tokens_for_crop
from src.data.audio import read_audio_segment
from src.data.corpora.librispeech import utterance_key
from src.data.db import (
    Connection,
    ensure_table,
    fetch_asr_timestamp_keys,
    fetch_asr_timestamps,
    fetch_utterances,
    metadata_path,
)
from src.data.schema import ASR_TIMESTAMPS_TABLE, AsrTimestamp, AsrToken, Utterance

_LIBRISPEECH_CORPUS = "LibriSpeech"


def asr_timestamp_to_json(record: AsrTimestamp) -> dict[str, object]:
    return {
        "utterance_key": record.utterance_key,
        "text": record.text,
        "samples_per_encoder_frame": record.samples_per_encoder_frame,
        "tokens": [token.to_json() for token in record.tokens],
    }


def asr_timestamp_from_json(payload: object) -> AsrTimestamp:
    if not isinstance(payload, dict):
        raise ValueError(f"timestamp record must be an object, got {type(payload).__name__}")
    tokens_raw = payload.get("tokens")
    if not isinstance(tokens_raw, list):
        raise ValueError("timestamp record is missing tokens")
    return AsrTimestamp(
        utterance_key=str(payload["utterance_key"]),
        text=str(payload["text"]),
        samples_per_encoder_frame=int(payload["samples_per_encoder_frame"]),
        tokens=[AsrToken.from_json(item) for item in tokens_raw],
    )


def load_timestamp_cache() -> dict[str, AsrTimestamp]:
    with Connection(metadata_path(), read_only=True) as connection:
        records = fetch_asr_timestamps(connection)
    return {record.utterance_key: record for record in records}


def require_timestamp_record(cache: dict[str, AsrTimestamp], key: str) -> AsrTimestamp:
    try:
        return cache[key]
    except KeyError:
        raise KeyError(f"timestamp cache miss for utterance_key={key!r}") from None


def missing_timestamp_keys(cache: dict[str, AsrTimestamp], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if key not in cache]


def crop_token_ids(record: AsrTimestamp, crop_start: int, crop_end: int) -> list[int]:
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


def _coverage(existing: set[str], utterances: Sequence[Utterance]) -> tuple[int, int, int, list[str]]:
    keys = [utterance_key(utterance) for utterance in utterances]
    missing = [key for key in keys if key not in existing]
    return len(keys), len(existing), len(missing), missing


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


def _read_jsonl_records(path: Path) -> list[AsrTimestamp]:
    records: dict[str, AsrTimestamp] = {}
    duplicates: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = asr_timestamp_from_json(json.loads(line))
            if record.utterance_key in records:
                duplicates.append(record.utterance_key)
            records[record.utterance_key] = record
    if duplicates:
        shown = ", ".join(duplicates[:5])
        raise ValueError(f"duplicate timestamp cache keys ({len(duplicates)}): {shown}")
    return list(records.values())


def _import_jsonl(path: Path, replace: bool) -> None:
    records = _read_jsonl_records(path)
    with ensure_table(metadata_path(), ASR_TIMESTAMPS_TABLE, replace=replace) as connection:
        existing = set() if replace else fetch_asr_timestamp_keys(connection)
        pending = [record for record in records if record.utterance_key not in existing]
        connection.insert(ASR_TIMESTAMPS_TABLE, pending)
        connection.commit()
        stored = fetch_asr_timestamp_keys(connection)
    print(
        f"import {path}: {len(records)} records, {len(existing)} existing, {len(pending)} inserted, "
        f"table_entries={len(stored)}",
        flush=True,
    )


def _asr_records_for_utterances(
    asr: object,
    utterances: Sequence[Utterance],
    sampling_rate: int,
    device: object,
) -> list[AsrTimestamp]:
    import torch

    waves = [_load_clean_waveform(utterance, sampling_rate) for utterance in utterances]
    max_len = max(int(wave.shape[0]) for wave in waves)
    batch = torch.zeros(len(waves), max_len, device=device, dtype=torch.float32)
    lengths = torch.zeros(len(waves), device=device, dtype=torch.int64)
    for i, wave in enumerate(waves):
        length = int(wave.shape[0])
        batch[i, :length] = torch.from_numpy(wave)
        lengths[i] = length
    recognitions = asr.recognize(batch, lengths)
    return [
        AsrTimestamp(
            utterance_key=utterance_key(utterance),
            text=recognition.text,
            samples_per_encoder_frame=recognition.samples_per_encoder_frame,
            tokens=[
                AsrToken(
                    token_id=token.token_id,
                    token=token.token,
                    start_offset=token.start_offset,
                    end_offset=token.end_offset,
                )
                for token in recognition.tokens
            ],
        )
        for utterance, recognition in zip(utterances, recognitions)
    ]


def _load_asr(device_name: str):
    import torch

    from src.asr.parakeet_tdt_0_6b_v2 import ParakeetTDT06BV2

    device = torch.device(device_name)
    asr = ParakeetTDT06BV2().to(device)
    asr.eval()
    return asr, device


def _check_cache(splits: Sequence[str], limit: int | None) -> int:
    utterances = _fetch_split_utterances(splits)
    if limit is not None:
        utterances = utterances[:limit]
    with ensure_table(metadata_path(), ASR_TIMESTAMPS_TABLE) as connection:
        existing = fetch_asr_timestamp_keys(connection)
    n_utterances, n_entries, n_missing, missing = _coverage(existing, utterances)
    _print_coverage(n_utterances, n_entries, n_missing, missing)
    return 1 if n_missing else 0


def _generate_jsonl(
    path: Path,
    splits: Sequence[str],
    sampling_rate: int,
    batch_size: int,
    device_name: str,
    limit: int | None,
    replace: bool,
) -> None:
    utterances = _fetch_split_utterances(splits)
    if limit is not None:
        utterances = utterances[:limit]
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, AsrTimestamp] = {}
    if path.is_file():
        if replace:
            path.unlink()
        else:
            existing = {record.utterance_key: record for record in _read_jsonl_records(path)}
    pending = [utterance for utterance in utterances if utterance_key(utterance) not in existing]
    print(
        f"cache {path}: {len(utterances)} utterances, {len(existing)} existing, {len(pending)} to recognize",
        flush=True,
    )
    if not pending:
        n_utterances, n_entries, n_missing, missing = _coverage(set(existing), utterances)
        _print_coverage(n_utterances, n_entries, n_missing, missing)
        return

    asr, device = _load_asr(device_name)
    written = 0
    with open(path, "a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            records = _asr_records_for_utterances(asr, chunk, sampling_rate, device)
            for record in records:
                handle.write(json.dumps(asr_timestamp_to_json(record), ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            done = min(start + batch_size, len(pending))
            print(f"recognized {done}/{len(pending)}", flush=True)
    stored = {record.utterance_key for record in _read_jsonl_records(path)}
    n_utterances, n_entries, n_missing, missing = _coverage(stored, utterances)
    _print_coverage(n_utterances, n_entries, n_missing, missing)
    print(f"wrote {written} new records to {path}", flush=True)


def _generate_cache(
    splits: Sequence[str],
    sampling_rate: int,
    batch_size: int,
    device_name: str,
    limit: int | None,
    replace: bool,
) -> None:
    utterances = _fetch_split_utterances(splits)
    if limit is not None:
        utterances = utterances[:limit]
    with ensure_table(metadata_path(), ASR_TIMESTAMPS_TABLE, replace=replace) as connection:
        existing = fetch_asr_timestamp_keys(connection)
        pending = [utterance for utterance in utterances if utterance_key(utterance) not in existing]
        print(
            f"asr_timestamps: {len(utterances)} utterances, {len(existing)} existing, {len(pending)} to recognize",
            flush=True,
        )
        if not pending:
            n_utterances, n_entries, n_missing, missing = _coverage(existing, utterances)
            _print_coverage(n_utterances, n_entries, n_missing, missing)
            return

        asr, device = _load_asr(device_name)
        written = 0
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            records = _asr_records_for_utterances(asr, chunk, sampling_rate, device)
            connection.insert(ASR_TIMESTAMPS_TABLE, records)
            connection.commit()
            written += len(records)
            done = min(start + batch_size, len(pending))
            print(f"recognized {done}/{len(pending)}", flush=True)
        existing = fetch_asr_timestamp_keys(connection)
    n_utterances, n_entries, n_missing, missing = _coverage(existing, utterances)
    _print_coverage(n_utterances, n_entries, n_missing, missing)
    print(f"wrote {written} new records to asr_timestamps", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete or check LibriSpeech Parakeet timestamps in DuckDB")
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--import", dest="import_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.import_path is not None:
        if args.check:
            parser.error("--import and --check cannot be combined")
        if args.splits:
            parser.error("--import and --splits cannot be combined")
        if args.output is not None:
            parser.error("--import and --output cannot be combined")
        _import_jsonl(args.import_path, args.replace)
        return
    if not args.splits:
        parser.error("--splits is required")
    if args.check:
        if args.replace:
            parser.error("--check and --replace cannot be combined")
        if args.output is not None:
            parser.error("--check and --output cannot be combined")
        raise SystemExit(_check_cache(args.splits, args.limit))
    if args.output is not None:
        _generate_jsonl(
            args.output,
            args.splits,
            args.sampling_rate,
            args.batch_size,
            args.device,
            args.limit,
            args.replace,
        )
        return
    _generate_cache(
        args.splits,
        args.sampling_rate,
        args.batch_size,
        args.device,
        args.limit,
        args.replace,
    )


if __name__ == "__main__":
    main()
