"""Measurement protocol v1. All durations are milliseconds unless named otherwise."""

import math

import numpy as np


MEASUREMENT_PROTOCOL = "benchmark-v1"
METRIC_DEFINITIONS = {
    "model_fps": "Samples / total synchronized predict wall time; excludes UI, drawing and encoding.",
    "active_processing_fps": "Frames / (active frame processing + encoder finalization); excludes preview submission, UI and pacing. Not an unpaced benchmark.",
    "server_wall_fps": "Frames / server wall time through encoder finalization; includes UI and pacing; excludes upload, setup, report storage and browser rendering.",
    "preview_submission_fps": "Preview images submitted to Streamlit / server wall time. Not browser display FPS.",
    "encode_submit_ms": "Time submitting frames to the encoder; buffered encoding can finish later.",
}


def summarize_samples(samples_ms):
    values = np.asarray(samples_ms, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Cần ít nhất một mẫu thời gian")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Thời gian phải hữu hạn và không âm")
    total = float(values.sum())
    return {
        "count": int(values.size),
        "total_ms": total,
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "fps": values.size * 1000.0 / total if total > 0 else None,
    }


def summarize_video(model_ms, active_frame_ms, wall_seconds, finalize_ms, previews):
    model = summarize_samples(model_ms)
    active = summarize_samples(active_frame_ms)
    if model["count"] != active["count"]:
        raise ValueError("Số mẫu model và frame phải khớp")
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("Wall time phải hữu hạn và lớn hơn 0")
    if not math.isfinite(finalize_ms) or finalize_ms < 0:
        raise ValueError("Thời gian hoàn tất encoder không hợp lệ")
    if not isinstance(previews, int) or not 0 <= previews <= model["count"]:
        raise ValueError("Số preview không hợp lệ")
    active_total = active["total_ms"] + finalize_ms
    return {
        "model_call": model,
        "frame_count": model["count"],
        "active_frame": active,
        "encoder_finalize_ms": finalize_ms,
        "active_processing_fps": model["count"] * 1000.0 / active_total if active_total > 0 else None,
        "server_wall_seconds": wall_seconds,
        "server_wall_fps": model["count"] / wall_seconds,
        "preview_submissions": previews,
        "preview_submission_fps": previews / wall_seconds,
    }
