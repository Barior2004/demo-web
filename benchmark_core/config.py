import os

BENCHMARK_DB_PATH = os.getenv("BENCHMARK_DB_PATH", "data/benchmarks.db")


DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/yolov10n_final_coordattn.onnx",
)

WARMUP_RUNS = int(
    os.getenv("WARMUP_RUNS", "10")
)

IMAGE_BENCHMARK_RUNS = int(
    os.getenv("IMAGE_BENCHMARK_RUNS", "50")
)

PREDICT_CONF = float(
    os.getenv("PREDICT_CONF", "0.44")
)

PREDICT_IOU = float(
    os.getenv("PREDICT_IOU", "0.60")
)

MAX_DETECTIONS = int(
    os.getenv("MAX_DETECTIONS", "100")
)

INPUT_IMAGE_SIZE = int(
    os.getenv("INPUT_IMAGE_SIZE", "640")
)


def validate_config():
    if WARMUP_RUNS < 1:
        raise ValueError("WARMUP_RUNS phải lớn hơn 0")

    if IMAGE_BENCHMARK_RUNS < 1:
        raise ValueError(
            "IMAGE_BENCHMARK_RUNS phải lớn hơn 0"
        )

    if not 0.0 <= PREDICT_CONF <= 1.0:
        raise ValueError(
            "PREDICT_CONF phải nằm trong khoảng 0 đến 1"
        )

    if not 0.0 <= PREDICT_IOU <= 1.0:
        raise ValueError(
            "PREDICT_IOU phải nằm trong khoảng 0 đến 1"
        )

    if MAX_DETECTIONS < 1:
        raise ValueError(
            "MAX_DETECTIONS phải lớn hơn 0"
        )

    if INPUT_IMAGE_SIZE < 32:
        raise ValueError(
            "INPUT_IMAGE_SIZE phải từ 32 trở lên"
        )
