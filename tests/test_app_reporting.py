"""Optional application integration checks; run where app dependencies are installed."""

import importlib.util
import io
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


APP_DEPENDENCIES = ("streamlit", "torch", "ultralytics", "cv2", "onnxruntime")
HAS_APP_DEPENDENCIES = all(importlib.util.find_spec(name) for name in APP_DEPENDENCIES)


@unittest.skipUnless(HAS_APP_DEPENDENCIES, "Requires installed app dependencies")
class AppReportingTests(unittest.TestCase):
    def exercise_app(self, kind, broken_storage=False):
        import numpy as np
        from PIL import Image
        from streamlit.delta_generator import DeltaGenerator
        from streamlit.testing.v1 import AppTest

        from benchmark_core import config
        from benchmark_core.reporting import BenchmarkStore
        from benchmark_core.video_output import H264VideoWriter

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            model_path = root / "test.onnx"
            model_path.touch()
            db_path = root if broken_storage else root / "results.db"
            if kind == "image":
                upload = io.BytesIO()
                Image.new("RGB", (16, 16)).save(upload, format="PNG")
                upload.name = "image.png"
            else:
                source = root / "source.mp4"
                with H264VideoWriter(source, 16, 16, 30) as writer:
                    for _ in range(5):
                        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
                upload = io.BytesIO(source.read_bytes())
                upload.name = "source.mp4"
            upload.size = upload.getbuffer().nbytes

            class FakeModel:
                def __init__(self, *args, **kwargs):
                    pass

                def predict(self, source, **kwargs):
                    return [SimpleNamespace(
                        speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 1.0},
                        plot=lambda: source.copy(),
                    )]

            stack.enter_context(patch.object(config, "DEFAULT_MODEL_PATH", str(model_path)))
            stack.enter_context(patch.object(config, "BENCHMARK_DB_PATH", str(db_path)))
            stack.enter_context(patch.object(config, "WARMUP_RUNS", 1))
            stack.enter_context(patch.object(config, "IMAGE_BENCHMARK_RUNS", 3))
            stack.enter_context(patch("ultralytics.YOLO", FakeModel))
            stack.enter_context(patch("torch.cuda.is_available", return_value=False))
            stack.enter_context(patch.object(DeltaGenerator, "file_uploader", return_value=upload))

            app_path = Path(__file__).resolve().parents[1] / "app.py"
            app = AppTest.from_file(str(app_path)).run(timeout=30)
            app.checkbox[0].uncheck().run(timeout=30)
            app.slider[0].set_value(3).run(timeout=30)
            app.button[0].click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            if broken_storage:
                self.assertTrue(any("KHÔNG lưu" in element.value for element in app.error))
                self.assertFalse(any("Đã lưu kết quả" in element.value for element in app.success))
                return

            unexpected_errors = [
                element.value for element in app.error
                if "Chưa phát hiện đầy đủ NVIDIA CUDA" not in element.value
            ]
            self.assertEqual(unexpected_errors, [])
            run_id = app.session_state["last_benchmark_run_id"]
            record = BenchmarkStore(db_path).get(run_id)
            self.assertEqual(record["kind"], kind)
            if kind == "video":
                self.assertEqual(record["metrics"]["frame_count"], 5)
                self.assertEqual(record["metrics"]["preview_submissions"], 2)
                self.assertGreater(record["metrics"]["encoder_finalize_ms"], 0)
            else:
                self.assertEqual(record["metrics"]["model_call"]["count"], 3)
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            connection = sqlite3.connect(db_path)
            try:
                self.assertEqual(connection.execute("SELECT count(*) FROM benchmark_runs").fetchone()[0], 1)
            finally:
                connection.close()

    def test_image_saved_once_across_reruns(self):
        self.exercise_app("image")

    def test_video_final_stats_include_last_nonpreview_frames(self):
        self.exercise_app("video")

    def test_failed_storage_shows_explicit_error(self):
        self.exercise_app("image", broken_storage=True)
