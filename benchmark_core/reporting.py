"""Local benchmark records. No images, videos or per-frame predictions are stored."""

import hashlib
import json
import platform
import sqlite3
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from benchmark_core.metrics import MEASUREMENT_PROTOCOL, METRIC_DEFINITIONS


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions():
    packages = {}
    for package in ("streamlit", "ultralytics", "torch", "onnxruntime", "onnxruntime-gpu", "numpy", "imageio-ffmpeg"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            continue
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def new_record(kind, model, input_info, config, environment):
    if kind not in ("image", "video"):
        raise ValueError("Loại benchmark không hợp lệ")
    return {
        "schema_version": 1,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "run_id": str(uuid4()),
        "started_at_utc": utc_now(),
        "kind": kind,
        "model": model,
        "input": input_info,
        "config": config,
        "environment": environment,
    }


class BenchmarkStore:
    def __init__(self, path):
        self.path = Path(path)

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def save_completed(self, record):
        if not record.get("metrics") or not record.get("completed_at_utc"):
            raise ValueError("Chỉ lưu benchmark đã hoàn tất và có kết quả")
        payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
        connection = self._connect()
        try:
            with connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS benchmark_runs (
                        run_id TEXT PRIMARY KEY,
                        started_at_utc TEXT NOT NULL,
                        kind TEXT NOT NULL CHECK(kind IN ('image', 'video')),
                        payload_json TEXT NOT NULL
                    )
                """)
                # A repeated save of the same completed run cannot create duplicates.
                connection.execute(
                    "INSERT OR IGNORE INTO benchmark_runs VALUES (?, ?, ?, ?)",
                    (record["run_id"], record["started_at_utc"], record["kind"], payload),
                )
        finally:
            connection.close()
        return record["run_id"]

    def get(self, run_id):
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT payload_json FROM benchmark_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return json.loads(row[0]) if row else None
        finally:
            connection.close()
