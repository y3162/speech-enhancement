import os
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf


def read_audio_segment(
    file_path: str | Path,
    start_frame: int,
    num_frames: int,
    range_start_frame: int | None = None,
    range_end_frame: int | None = None,
) -> np.ndarray:
    chunks = []
    remaining_frames = num_frames

    with sf.SoundFile(file_path) as f:
        if range_start_frame is None:
            range_start_frame = 0
        if range_end_frame is None:
            range_end_frame = len(f)
        range_len = range_end_frame - range_start_frame
        if range_len < 1:
            raise ValueError(f"Invalid range [{range_start_frame}, {range_end_frame})")

        start_frame %= range_len
        pos = range_start_frame + start_frame
        f.seek(pos)

        while remaining_frames > 0:
            to_read = min(remaining_frames, range_end_frame - pos)
            audio = f.read(
                frames=to_read,
                dtype="float32",
                always_2d=True,
            )
            if audio.shape[0] == 0:
                raise ValueError(f"Failed to read audio from {file_path} at frame {pos}")

            chunks.append(audio)
            remaining_frames -= audio.shape[0]

            if remaining_frames > 0:
                pos = range_start_frame
                f.seek(range_start_frame)

    return np.concatenate(chunks, axis=0)


def read_audio_stream_info(audio_path: Path) -> tuple[int, int, int]:
    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        return read_wav_stream_info(audio_path)
    if suffix == ".flac":
        return read_flac_stream_info(audio_path)
    raise ValueError(f"Unsupported audio format: {audio_path}")


def read_wav_stream_info(audio_path: Path) -> tuple[int, int, int]:
    with open(audio_path, "rb", buffering=0) as f:
        _advise_random(f)
        header = f.read(44)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"Not a WAV file: {audio_path}")
        if (
            len(header) >= 44
            and header[12:16] == b"fmt "
            and int.from_bytes(header[16:20], "little") == 16
            and header[36:40] == b"data"
        ):
            audio_format = int.from_bytes(header[20:22], "little")
            channels = int.from_bytes(header[22:24], "little")
            sample_rate = int.from_bytes(header[24:28], "little")
            block_align = int.from_bytes(header[32:34], "little")
            data_size = int.from_bytes(header[40:44], "little")
        else:
            f.seek(12)
            audio_format, channels, sample_rate, block_align, data_size = _read_wav_chunks(f, audio_path)

    if audio_format != 1:
        raise ValueError(f"Unsupported WAV format {audio_format}: {audio_path}")
    if block_align == 0:
        raise ValueError(f"Invalid WAV block align: {audio_path}")
    return sample_rate, data_size // block_align, channels


def _advise_random(f: BinaryIO) -> None:
    try:
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
    except OSError:
        pass


def _read_wav_chunks(f: BinaryIO, audio_path: Path) -> tuple[int, int, int, int, int]:
    fmt = None
    data_size = None
    while True:
        chunk_header = f.read(8)
        if len(chunk_header) < 8:
            break
        chunk_id = chunk_header[:4]
        chunk_size = int.from_bytes(chunk_header[4:8], "little")
        if chunk_id == b"fmt ":
            fmt_data = f.read(chunk_size)
            if len(fmt_data) < 16:
                raise ValueError(f"Invalid WAV fmt chunk: {audio_path}")
            audio_format = int.from_bytes(fmt_data[0:2], "little")
            channels = int.from_bytes(fmt_data[2:4], "little")
            sample_rate = int.from_bytes(fmt_data[4:8], "little")
            block_align = int.from_bytes(fmt_data[12:14], "little")
            fmt = (audio_format, channels, sample_rate, block_align)
            if chunk_size % 2:
                f.seek(1, 1)
        elif chunk_id == b"data":
            data_size = chunk_size
        else:
            f.seek(chunk_size, 1)
            if chunk_size % 2:
                f.seek(1, 1)
        if fmt is not None and data_size is not None:
            break

    if fmt is None or data_size is None:
        raise ValueError(f"Missing WAV fmt or data chunk: {audio_path}")
    audio_format, channels, sample_rate, block_align = fmt
    return audio_format, channels, sample_rate, block_align, data_size


def read_flac_stream_info(audio_path: Path) -> tuple[int, int, int]:
    with open(audio_path, "rb", buffering=0) as f:
        _advise_random(f)
        header = f.read(42)
    if len(header) < 42 or header[:4] != b"fLaC":
        raise ValueError(f"Not a FLAC file: {audio_path}")

    """
    | sample rate | channels-1 | bits/sample-1 | total samples |
    |   20 bits   |   3 bits   |    5 bits     |    36 bits    |
    """
    packed = int.from_bytes(header[18:26], "big")
    sample_rate = packed >> 44
    channels = ((packed >> 41) & 0x7) + 1
    frames = packed & 0xFFFFFFFFF
    return sample_rate, frames, channels
