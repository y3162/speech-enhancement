from pathlib import Path
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
            raise ValueError(
                f"Invalid range [{range_start_frame}, {range_end_frame})"
            )

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
                raise ValueError(
                    f"Failed to read audio from {file_path} at frame {pos}"
                )

            chunks.append(audio)
            remaining_frames -= audio.shape[0]

            if remaining_frames > 0:
                pos = range_start_frame
                f.seek(range_start_frame)

    return np.concatenate(chunks, axis=0)
