import hashlib
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark_core.metrics import summarize_samples, summarize_video
from benchmark_core.reporting import BenchmarkStore, new_record, sha256_file, utc_now


def completed_record(kind="image"):
    record = new_record(kind, {"sha256": "model-hash"}, {"name": "ảnh.jpg"}, {}, {})
    record["completed_at_utc"] = utc_now()
    record["metrics"] = {"model_call": summarize_samples([10, 20, 30])}
    return record


class MeasurementTests(unittest.TestCase):
    def test_fps_uses_total_time_not_average_of_reciprocals(self):
        stats = summarize_samples([10, 20, 30])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["mean_ms"], 20)
        self.assertEqual(stats["p50_ms"], 20)
        self.assertEqual(stats["p95_ms"], 29)
        self.assertEqual(stats["fps"], 50)

    def test_video_includes_encoder_flush_and_distinguishes_preview_count(self):
        stats = summarize_video([10, 10, 10], [20, 20, 20], 2, 30, 1)
        self.assertEqual(stats["model_call"]["fps"], 100)
        self.assertAlmostEqual(stats["active_processing_fps"], 1000 / 30)
        self.assertEqual(stats["server_wall_fps"], 1.5)
        self.assertEqual(stats["preview_submission_fps"], 0.5)
        self.assertEqual(stats["frame_count"], 3)

    def test_rejects_invalid_samples(self):
        for samples in ([], [-1], [float("nan")], [float("inf")]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                summarize_samples(samples)
        self.assertIsNone(summarize_samples([0])["fps"])
        with self.assertRaises(ValueError):
            summarize_video([1], [1, 2], 1, 0, 0)


class ReportingTests(unittest.TestCase):
    def test_reopen_and_duplicate_save_preserve_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data" / "benchmarks.db"
            record = completed_record()
            store = BenchmarkStore(path)
            store.save_completed(record)
            store.save_completed(record)
            reopened = BenchmarkStore(path)
            self.assertEqual(reopened.get(record["run_id"]), record)
            self.assertIsNone(reopened.get("missing"))
            connection = reopened._connect()
            try:
                self.assertEqual(connection.execute("SELECT count(*) FROM benchmark_runs").fetchone()[0], 1)
            finally:
                connection.close()

    def test_concurrent_sessions_do_not_lose_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            store = BenchmarkStore(Path(directory) / "benchmarks.db")
            records = [completed_record() for _ in range(8)]
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(store.save_completed, records))
            for record in records:
                self.assertEqual(store.get(record["run_id"]), record)

    def test_incomplete_and_nonfinite_results_are_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            store = BenchmarkStore(Path(directory) / "benchmarks.db")
            with self.assertRaises(ValueError):
                store.save_completed({})
            record = completed_record()
            record["metrics"]["bad"] = float("nan")
            with self.assertRaises(ValueError):
                store.save_completed(record)

    def test_model_hash_uses_file_contents(self):
        self.assertEqual(
            sha256_file(__file__), hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )


if __name__ == "__main__":
    unittest.main()
