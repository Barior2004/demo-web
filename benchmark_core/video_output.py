from pathlib import Path

import imageio_ffmpeg
import numpy as np


class H264VideoWriter:
    """Encode BGR frames to a browser-compatible H.264 MP4 file."""

    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._writer = None

        if self.width < 1 or self.height < 1:
            raise ValueError("Kích thước video không hợp lệ")

        if self.fps <= 0:
            raise ValueError("FPS video phải lớn hơn 0")

    def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio_ffmpeg.write_frames(
            str(self.path),
            (self.width, self.height),
            pix_fmt_in="bgr24",
            pix_fmt_out="yuv420p",
            fps=self.fps,
            quality=6,
            macro_block_size=2,
            ffmpeg_log_level="error",
            output_params=["-movflags", "+faststart"],
        )
        self._writer.send(None)
        return self

    def write(self, frame):
        if self._writer is None:
            raise RuntimeError("Video writer chưa được mở")

        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                "Frame không khớp kích thước video output"
            )

        self._writer.send(np.ascontiguousarray(frame))

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

