"""Build a static clean-ASR vs noisy-ASR alignment cache. Does not train.

python -m src.se.error_aware_se_mamba_pp.build_cache --alignment-cache PATH [--config JSON]
"""

import src.se.common.cuda_local as _cuda_local  # noqa: F401  # isort: skip

import argparse
import queue
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from src.asr.parakeet_tdt_0_6b_v2 import ParakeetTDT06BV2, Recognition, Token
from src.se.common.training import load_config
from src.se.error_aware_se_mamba_pp.alignment_cache import AlignmentRow, AlignmentWriter, build_fingerprint
from src.se.error_aware_se_mamba_pp.dataset import ScheduledNoiseDataset, datasets_from_cfg
from src.se.error_aware_se_mamba_pp.error_mask import align_token_ids, error_frames_from_alignment

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.json"
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"


def _token_dict(token: Token) -> dict[str, object]:
    return {
        "token_id": int(token.token_id),
        "token": str(token.token),
        "start_offset": int(token.start_offset),
        "end_offset": int(token.end_offset),
    }


def _parse_indices(text: str | None, size: int) -> list[int]:
    if text is None:
        return list(range(size))
    indices = [int(part) for part in text.split(",") if part.strip()]
    for index in indices:
        if index < 0 or index >= size:
            raise ValueError(f"utterance index {index} is outside 0..{size - 1}")
    return indices


def _variants(
    dataset: ScheduledNoiseDataset,
    epoch_start: int,
    epoch_end: int,
    variants_per_utterance: int | None,
) -> list[int]:
    if not dataset.crop:
        return [0]
    if variants_per_utterance is not None:
        return list(range(variants_per_utterance))
    if epoch_end <= epoch_start:
        raise ValueError(f"epoch range is empty: {epoch_start}..{epoch_end}")
    return list(range(epoch_start, epoch_end))


@dataclass
class _Prepared:
    split: str
    index: int
    variant: int
    key: str
    spec_entropy: str
    crop_start: int
    crop_end: int
    source_frames: int
    pipeline_index: int
    noise_id: int
    noise_offset: int
    snr_db: float
    content: int
    clean: torch.Tensor
    noisy: torch.Tensor


def _prepare(dataset: ScheduledNoiseDataset, split: str, index: int, variant: int) -> _Prepared:
    sample = dataset.materialize(index, variant)
    spec = dataset.spec_for(index, variant)
    content = spec.content_samples
    return _Prepared(
        split=split,
        index=index,
        variant=variant,
        key=sample.utterance_key,
        spec_entropy=spec.entropy,
        crop_start=spec.crop_start,
        crop_end=spec.crop_end,
        source_frames=spec.source_frames,
        pipeline_index=spec.pipeline_index,
        noise_id=spec.noise_id,
        noise_offset=spec.noise_offset,
        snr_db=spec.snr_db,
        content=content,
        clean=sample.clean[:content].contiguous(),
        noisy=sample.noisy[:content].contiguous(),
    )


def _recognize_many(asr: ParakeetTDT06BV2, waves: list[torch.Tensor], device: torch.device) -> list[Recognition]:
    lengths = [int(wave.numel()) for wave in waves]
    batch = torch.zeros(len(waves), max(lengths), dtype=waves[0].dtype)
    for index, wave in enumerate(waves):
        batch[index, : wave.numel()] = wave
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    with torch.inference_mode():
        return asr.recognize(batch.to(device, non_blocking=True), length_tensor)


def _insert_pair(writer: AlignmentWriter, item: _Prepared, clean_rec: Recognition, noisy_rec: Recognition) -> None:
    clean_tokens = [_token_dict(token) for token in clean_rec.tokens]
    noisy_tokens = [_token_dict(token) for token in noisy_rec.tokens]
    alignment = align_token_ids(
        [int(token["token_id"]) for token in clean_tokens],
        [int(token["token_id"]) for token in noisy_tokens],
    )
    error_frames = error_frames_from_alignment(clean_tokens, alignment)
    writer.insert(
        AlignmentRow(
            split=item.split,
            utterance_index=item.index,
            variant=item.variant,
            utterance_key=item.key,
            crop_start=item.crop_start,
            crop_end=item.crop_end,
            source_frames=item.source_frames,
            pipeline_index=item.pipeline_index,
            noise_id=item.noise_id,
            noise_offset=item.noise_offset,
            snr_db=item.snr_db,
            content_samples=item.content,
            entropy=item.spec_entropy,
            clean_encoded_length=int(clean_rec.encoded_length),
            noisy_encoded_length=int(noisy_rec.encoded_length),
            clean_text=clean_rec.text,
            noisy_text=noisy_rec.text,
            clean_tokens=clean_tokens,
            noisy_tokens=noisy_tokens,
            alignment=alignment,
            error_frames=error_frames,
        )
    )


def _produce_guard(fn: Callable[[], None], errors: list[BaseException], prepared: queue.Queue) -> None:
    try:
        fn()
    except BaseException as exc:
        errors.append(exc)
        prepared.put(None)


def build(argv: list[str] | None = None) -> dict[str, int]:
    parser = argparse.ArgumentParser(description="Cache clean-ASR vs noisy-ASR S/D masks")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--alignment-cache", required=True)
    parser.add_argument("--epoch-start", type=int, default=0)
    parser.add_argument("--epoch-end", type=int, default=None)
    parser.add_argument("--indices", default=None, help="comma-separated utterance indices for every selected split")
    parser.add_argument("--split", default="train,valid")
    parser.add_argument("--asr-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-batch-samples",
        type=int,
        default=0,
        help="cap clean+noisy samples per recognize call; 0 means no cap",
    )
    parser.add_argument("--prep-workers", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--merge-shards", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    if args.merge_shards is not None:
        from src.se.error_aware_se_mamba_pp.alignment_cache import merge_shard_databases

        count = merge_shard_databases(Path(args.alignment_cache), args.merge_shards, args.rebuild)
        print(f"merged {count} rows into {args.alignment_cache}", flush=True)
        return {"wrote": count, "skipped": 0, "jobs": count}
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        parser.error(f"--shard must be in 0..{args.num_shards - 1}")
    if args.asr_batch_size < 1:
        parser.error("--asr-batch-size must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the alignment cache")

    cfg = load_config(args.config)
    variants = cfg.data.variants_per_utterance
    if variants is not None:
        variants = int(variants)
        if variants < 1:
            parser.error("data.variants_per_utterance must be >= 1 or null")
        if args.epoch_start != 0 or args.epoch_end is not None:
            parser.error("--epoch-start and --epoch-end apply only when data.variants_per_utterance is null")
    elif args.epoch_end is None:
        args.epoch_end = int(cfg.train.epochs)
    trainset, validset = datasets_from_cfg(cfg)
    device = torch.device("cuda")
    asr = ParakeetTDT06BV2()
    fingerprint = build_fingerprint(
        asr,
        trainset,
        validset,
        sampling_rate=int(cfg.data.sampling_rate),
        segment_size=int(cfg.data.segment_size),
        normalize=str(cfg.data.normalize),
        train_splits=list(cfg.data.train_splits),
        valid_splits=list(cfg.data.validation_splits),
        base_seed=int(cfg.train.seed),
        variants_per_utterance=variants,
    )
    asr = asr.to(device)
    asr.eval()
    for parameter in asr.parameters():
        parameter.requires_grad_(False)

    splits = [part.strip() for part in args.split.split(",") if part.strip()]
    jobs: list[tuple[str, ScheduledNoiseDataset, int, int]] = []
    for split in splits:
        if split == TRAIN_SPLIT:
            dataset = trainset
        elif split == VALID_SPLIT:
            dataset = validset
        else:
            raise ValueError(f"unknown split {split!r}")
        indices = [
            index for index in _parse_indices(args.indices, len(dataset)) if index % args.num_shards == args.shard
        ]
        epoch_end = 0 if args.epoch_end is None else args.epoch_end
        for index in indices:
            for variant in _variants(dataset, args.epoch_start, epoch_end, variants):
                jobs.append((split, dataset, index, variant))

    print(f"shard {args.shard}/{args.num_shards} jobs={len(jobs)}", flush=True)
    writer = AlignmentWriter(Path(args.alignment_cache), fingerprint, args.rebuild)
    done_rows = {
        (str(row[0]), int(row[1]), int(row[2]))
        for row in writer.connection.execute("SELECT split, utterance_index, variant FROM alignment")
    }
    written = 0
    skipped = 0
    prepared: queue.Queue[_Prepared | None] = queue.Queue(maxsize=max(2, args.asr_batch_size))
    prep_error: list[BaseException] = []

    def produce() -> None:
        nonlocal skipped
        todo = [
            (split, dataset, index, variant)
            for split, dataset, index, variant in jobs
            if args.rebuild or (split, index, variant) not in done_rows
        ]
        skipped = len(jobs) - len(todo)

        def prepare_job(job: tuple[str, ScheduledNoiseDataset, int, int]) -> _Prepared:
            split, dataset, index, variant = job
            return _prepare(dataset, split, index, variant)

        with ThreadPoolExecutor(max_workers=max(1, args.prep_workers)) as pool:
            chunk = max(args.asr_batch_size, args.prep_workers)
            for start in range(0, len(todo), chunk):
                for item in pool.map(prepare_job, todo[start : start + chunk]):
                    prepared.put(item)
        prepared.put(None)

    producer = threading.Thread(target=_produce_guard, args=(produce, prep_error, prepared), daemon=True)
    producer.start()
    hold: _Prepared | None = None
    finished = False
    try:
        while not finished:
            if prep_error:
                raise prep_error[0]
            batch: list[_Prepared] = []
            samples = 0
            if hold is not None:
                batch.append(hold)
                samples = hold.content * 2
                hold = None
            while len(batch) < args.asr_batch_size:
                item = prepared.get()
                if item is None:
                    finished = True
                    break
                need = item.content * 2
                over_cap = args.max_batch_samples > 0 and batch and samples + need > args.max_batch_samples
                if over_cap:
                    hold = item
                    break
                batch.append(item)
                samples += need
            if not batch:
                break
            recordings = _recognize_many(
                asr,
                [item.clean for item in batch] + [item.noisy for item in batch],
                device,
            )
            half = len(batch)
            for item, clean_rec, noisy_rec in zip(batch, recordings[:half], recordings[half:]):
                _insert_pair(writer, item, clean_rec, noisy_rec)
                written += 1
            writer.commit()
            print(f"shard {args.shard} wrote {written} skipped {skipped}", flush=True)
        writer.commit()
        producer.join()
        if prep_error:
            raise prep_error[0]
        print(f"shard {args.shard} done wrote {written} skipped {skipped}", flush=True)
    finally:
        writer.close()
    return {"wrote": written, "skipped": skipped, "jobs": len(jobs)}


def main() -> None:
    build()


if __name__ == "__main__":
    main()
