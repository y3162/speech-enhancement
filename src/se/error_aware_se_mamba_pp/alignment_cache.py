"""Read-only alignment cache. Connections are opened on first use, never before worker fork."""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn

from src.se.error_aware_se_mamba_pp.asr_guidance import ASR_DIM, ASR_HOP, ENCODER_LAYER
from src.se.error_aware_se_mamba_pp.dataset import (
    ScheduledNoiseDataset,
    noise_list_sha256,
    utterance_key_sha256,
)
from src.se.error_aware_se_mamba_pp.error_mask import error_frames_from_alignment
from src.se.error_aware_se_mamba_pp.schedule import AugmentationSpec

CACHE_SCHEMA = "2"
ALIGNMENT_VERSION = "error_aware_se_mamba_pp.align_token_ids.v1"
MASK_RULE = "center_j_times_hop"
SCHEDULE_VERSION = "seed_sequence_v1"


@dataclass(frozen=True)
class AlignmentRow:
    split: str
    utterance_index: int
    variant: int
    utterance_key: str
    crop_start: int
    crop_end: int
    source_frames: int
    pipeline_index: int
    noise_id: int
    noise_offset: int
    snr_db: float
    content_samples: int
    entropy: str
    clean_encoded_length: int
    noisy_encoded_length: int
    clean_text: str
    noisy_text: str
    clean_tokens: list[dict[str, object]]
    noisy_tokens: list[dict[str, object]]
    alignment: list[tuple[str, int | None, int | None]]
    error_frames: list[int]


def module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    items = [(f"p:{name}", tensor) for name, tensor in module.named_parameters()]
    items.extend((f"b:{name}", tensor) for name, tensor in module.named_buffers())
    for name, tensor in items:
        data = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(data.shape)).encode())
        digest.update(str(data.dtype).encode())
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def fingerprint_mismatches(stored: dict[str, str], current: dict[str, str]) -> list[str]:
    lines = []
    for key in sorted(set(stored) | set(current)):
        if stored.get(key) != current.get(key):
            lines.append(f"{key}: cache={stored.get(key)!r} current={current.get(key)!r}")
    return lines


def build_fingerprint(
    asr: Any,
    trainset: ScheduledNoiseDataset,
    validset: ScheduledNoiseDataset,
    *,
    sampling_rate: int,
    segment_size: int,
    normalize: str,
    train_splits: list[str],
    valid_splits: list[str],
    base_seed: int,
    variants_per_utterance: int | None,
) -> dict[str, str]:
    import nemo

    if int(asr.samples_per_encoder_frame) != ASR_HOP:
        raise RuntimeError(f"ASR hop {asr.samples_per_encoder_frame} != ASR_HOP {ASR_HOP}")
    decoding = asr.model.cfg.decoding
    return {
        "schema": CACHE_SCHEMA,
        "asr_model_name": "nvidia/parakeet-tdt-0.6b-v2",
        "nemo_version": nemo.__version__,
        "samples_per_encoder_frame": str(int(asr.samples_per_encoder_frame)),
        "encoder_layers": str(len(asr.model.encoder.layers)),
        "encoder_layer": str(ENCODER_LAYER),
        "asr_dim": str(ASR_DIM),
        "decoding_strategy": str(decoding.strategy),
        "compute_timestamps": str(bool(decoding.compute_timestamps)),
        "tdt_include_token_duration": str(bool(decoding.tdt_include_token_duration)),
        "use_cuda_graph_decoder": str(bool(decoding.greedy.use_cuda_graph_decoder)),
        "asr_parameter_sha256": module_sha256(asr),
        "alignment_version": ALIGNMENT_VERSION,
        "mask_rule": MASK_RULE,
        "schedule_version": SCHEDULE_VERSION,
        "sampling_rate": str(sampling_rate),
        "segment_size": str(segment_size),
        "normalize": normalize,
        "train_splits": ",".join(train_splits),
        "valid_splits": ",".join(valid_splits),
        "base_seed": str(base_seed),
        "variants_per_utterance": "epoch" if variants_per_utterance is None else str(variants_per_utterance),
        "train_utterances": str(len(trainset)),
        "valid_utterances": str(len(validset)),
        "train_keys_sha256": utterance_key_sha256(trainset.utterances),
        "valid_keys_sha256": utterance_key_sha256(validset.utterances),
        "train_noise_sha256": noise_list_sha256(trainset.noise_meta),
        "valid_noise_sha256": noise_list_sha256(validset.noise_meta),
    }


def _connect_rw(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def require_cache_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"alignment cache does not exist: {path}. "
            "Build it with python -m src.se.error_aware_se_mamba_pp.build_cache."
        )
    return resolved


def _connect_ro(path: Path) -> sqlite3.Connection:
    resolved = require_cache_file(path)
    connection = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS alignment (
            split TEXT NOT NULL,
            utterance_index INTEGER NOT NULL,
            variant INTEGER NOT NULL,
            utterance_key TEXT NOT NULL,
            crop_start INTEGER NOT NULL,
            crop_end INTEGER NOT NULL,
            source_frames INTEGER NOT NULL,
            pipeline_index INTEGER NOT NULL,
            noise_id INTEGER NOT NULL,
            noise_offset INTEGER NOT NULL,
            snr_db REAL NOT NULL,
            content_samples INTEGER NOT NULL,
            entropy TEXT NOT NULL,
            clean_encoded_length INTEGER NOT NULL,
            noisy_encoded_length INTEGER NOT NULL,
            clean_text TEXT NOT NULL,
            noisy_text TEXT NOT NULL,
            clean_tokens_json TEXT NOT NULL,
            noisy_tokens_json TEXT NOT NULL,
            alignment_json TEXT NOT NULL,
            error_frames_json TEXT NOT NULL,
            PRIMARY KEY (split, utterance_index, variant)
        )
        """
    )


def read_fingerprint(connection: sqlite3.Connection) -> dict[str, str]:
    row = connection.execute("SELECT value FROM meta WHERE key = 'fingerprint'").fetchone()
    if row is None:
        raise RuntimeError("alignment cache has no fingerprint")
    payload = json.loads(row["value"])
    if not isinstance(payload, dict):
        raise RuntimeError("alignment cache fingerprint is not an object")
    return {str(key): str(value) for key, value in payload.items()}


def write_fingerprint(connection: sqlite3.Connection, fingerprint: dict[str, str]) -> None:
    connection.execute(
        "INSERT INTO meta(key, value) VALUES ('fingerprint', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (json.dumps(fingerprint, sort_keys=True),),
    )


def assert_fingerprint(connection: sqlite3.Connection, current: dict[str, str]) -> None:
    mismatches = fingerprint_mismatches(read_fingerprint(connection), current)
    if mismatches:
        shown = "\n".join(mismatches[:40])
        raise RuntimeError("alignment cache fingerprint mismatch:\n" + shown)


class AlignmentWriter:
    def __init__(self, path: Path, fingerprint: dict[str, str], rebuild: bool) -> None:
        self.path = path
        if rebuild and path.is_file():
            path.unlink()
        self.connection = _connect_rw(path)
        _create_tables(self.connection)
        existing = self.connection.execute("SELECT value FROM meta WHERE key = 'fingerprint'").fetchone()
        if existing is None:
            write_fingerprint(self.connection, fingerprint)
            self.connection.commit()
        else:
            assert_fingerprint(self.connection, fingerprint)

    def has_row(self, split: str, index: int, variant: int) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM alignment WHERE split = ? AND utterance_index = ? AND variant = ?",
            (split, index, variant),
        ).fetchone()
        return row is not None

    def insert(self, row: AlignmentRow) -> None:
        recomputed = error_frames_from_alignment(row.clean_tokens, row.alignment)
        if recomputed != row.error_frames:
            raise RuntimeError("error frames do not match the alignment")
        self.connection.execute(
            """
            INSERT OR REPLACE INTO alignment (
                split, utterance_index, variant, utterance_key, crop_start, crop_end, source_frames,
                pipeline_index, noise_id, noise_offset, snr_db, content_samples, entropy,
                clean_encoded_length, noisy_encoded_length, clean_text, noisy_text,
                clean_tokens_json, noisy_tokens_json, alignment_json, error_frames_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.split,
                row.utterance_index,
                row.variant,
                row.utterance_key,
                row.crop_start,
                row.crop_end,
                row.source_frames,
                row.pipeline_index,
                row.noise_id,
                row.noise_offset,
                row.snr_db,
                row.content_samples,
                row.entropy,
                row.clean_encoded_length,
                row.noisy_encoded_length,
                row.clean_text,
                row.noisy_text,
                json.dumps(row.clean_tokens),
                json.dumps(row.noisy_tokens),
                json.dumps(row.alignment),
                json.dumps(row.error_frames),
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def _decode_alignment(payload: object) -> list[tuple[str, int | None, int | None]]:
    if not isinstance(payload, list):
        raise RuntimeError("alignment_json is not a list")
    ops = []
    for item in payload:
        operation, clean_index, noisy_index = item
        ops.append(
            (
                str(operation),
                None if clean_index is None else int(clean_index),
                None if noisy_index is None else int(noisy_index),
            )
        )
    return ops


def _row_from_sql(row: sqlite3.Row) -> AlignmentRow:
    clean_tokens = json.loads(row["clean_tokens_json"])
    alignment = _decode_alignment(json.loads(row["alignment_json"]))
    error_frames = [int(frame) for frame in json.loads(row["error_frames_json"])]
    if error_frames_from_alignment(clean_tokens, alignment) != error_frames:
        raise RuntimeError(f"stored error frames disagree with alignment for {row['split']} {row['utterance_index']}")
    return AlignmentRow(
        split=str(row["split"]),
        utterance_index=int(row["utterance_index"]),
        variant=int(row["variant"]),
        utterance_key=str(row["utterance_key"]),
        crop_start=int(row["crop_start"]),
        crop_end=int(row["crop_end"]),
        source_frames=int(row["source_frames"]),
        pipeline_index=int(row["pipeline_index"]),
        noise_id=int(row["noise_id"]),
        noise_offset=int(row["noise_offset"]),
        snr_db=float(row["snr_db"]),
        content_samples=int(row["content_samples"]),
        entropy=str(row["entropy"]),
        clean_encoded_length=int(row["clean_encoded_length"]),
        noisy_encoded_length=int(row["noisy_encoded_length"]),
        clean_text=str(row["clean_text"]),
        noisy_text=str(row["noisy_text"]),
        clean_tokens=clean_tokens,
        noisy_tokens=json.loads(row["noisy_tokens_json"]),
        alignment=alignment,
        error_frames=error_frames,
    )


class AlignmentStore:
    """Lazy read-only connection. Construct this before workers fork; do not query until they have started."""

    def __init__(self, path: Path, fingerprint: dict[str, str]) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self._connection: sqlite3.Connection | None = None

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            if not self.path.is_file():
                raise RuntimeError(f"alignment cache does not exist: {self.path}")
            self._connection = _connect_ro(self.path)
            assert_fingerprint(self._connection, self.fingerprint)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def get(self, split: str, index: int, variant: int) -> AlignmentRow:
        row = (
            self._conn()
            .execute(
                "SELECT * FROM alignment WHERE split = ? AND utterance_index = ? AND variant = ?",
                (split, int(index), int(variant)),
            )
            .fetchone()
        )
        if row is None:
            raise KeyError(
                f"alignment cache missing split={split} index={index} variant={variant}. "
                "Build it with python -m src.se.error_aware_se_mamba_pp.build_cache"
            )
        return _row_from_sql(row)


def assert_row_matches(row: AlignmentRow, spec: AugmentationSpec, key: str) -> None:
    if row.utterance_key != key or row.entropy != spec.entropy:
        raise RuntimeError(f"cache identity mismatch for {key}: {row.entropy} vs {spec.entropy}")
    if (row.crop_start, row.crop_end, row.source_frames) != (spec.crop_start, spec.crop_end, spec.source_frames):
        raise RuntimeError(f"cache crop mismatch for {key}")
    if row.pipeline_index != spec.pipeline_index or row.noise_id != spec.noise_id:
        raise RuntimeError(f"cache noise mismatch for {key}")
    if row.noise_offset != spec.noise_offset or float(row.snr_db) != float(spec.snr_db):
        raise RuntimeError(f"cache SNR/offset mismatch for {key}")
    if row.content_samples != spec.content_samples:
        raise RuntimeError(f"cache content length mismatch for {key}")


def shard_cache_path(output: Path, shard: int) -> Path:
    return output.with_name(f"{output.stem}.shard{shard}{output.suffix}")


def merge_shard_databases(output: Path, num_shards: int, rebuild: bool) -> int:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    shards = [shard_cache_path(output, shard) for shard in range(num_shards)]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing shard cache: " + ", ".join(missing))
    if output.exists() and not rebuild:
        raise FileExistsError(f"{output} exists; pass --rebuild to replace it")
    if output.exists():
        output.unlink()
    fingerprints = []
    for path in shards:
        connection = _connect_ro(path)
        try:
            fingerprints.append(read_fingerprint(connection))
        finally:
            connection.close()
    for shard, fingerprint in enumerate(fingerprints[1:], start=1):
        mismatches = fingerprint_mismatches(fingerprints[0], fingerprint)
        if mismatches:
            raise RuntimeError(f"shard {shard} fingerprint mismatch:\n" + "\n".join(mismatches[:20]))
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect_rw(output)
    try:
        _create_tables(connection)
        write_fingerprint(connection, fingerprints[0])
        total = 0
        for shard, path in enumerate(shards):
            alias = f"shard{shard}"
            connection.execute(f"ATTACH DATABASE ? AS {alias}", (str(path.resolve()),))
            inserted = connection.execute(f"SELECT COUNT(*) FROM {alias}.alignment").fetchone()[0]
            connection.execute(f"INSERT INTO alignment SELECT * FROM {alias}.alignment")
            connection.commit()
            connection.execute(f"DETACH DATABASE {alias}")
            total += int(inserted)
    finally:
        connection.close()
    return total


def verify_cache_file(path: Path, fingerprint: dict[str, str]) -> None:
    connection = _connect_ro(path)
    try:
        assert_fingerprint(connection, fingerprint)
    finally:
        connection.close()


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
