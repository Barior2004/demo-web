import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import imageio_ffmpeg

from benchmark_core.video_output import H264VideoWriter


class H264VideoWriterTest(unittest.TestCase):
    def test_writes_browser_compatible_mp4(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.mp4"
            frame = np.zeros((16, 16, 3), dtype=np.uint8)

            with H264VideoWriter(
                output_path,
                width=16,
                height=16,
                fps=30,
            ) as writer:
                writer.write(frame)
                writer.write(frame)

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

            probe = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-hide_banner",
                    "-i",
                    str(output_path),
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                check=True,
                text=True,
            )

            self.assertIn("Video: h264", probe.stderr)
            self.assertIn("yuv420p", probe.stderr)
            self.assertIn("16x16", probe.stderr)


if __name__ == "__main__":
    unittest.main()
