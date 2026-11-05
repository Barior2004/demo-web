#!/usr/bin/env bash

set -Eeuo pipefail

readonly COMPOSE_FILE="compose.gpu.yaml"
readonly IMAGE_NAME="demo-web:gpu-optimized"
readonly ROLLBACK_IMAGE="demo-web:gpu-rollback"
readonly MODEL_FILE="models/yolov10n_final_coordattn.onnx"
readonly HEALTH_URL="http://127.0.0.1:8501/_stcore/health"

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

if [[ "${EUID}" -eq 0 ]]; then
    DOCKER=(docker)
elif docker info >/dev/null 2>&1; then
    DOCKER=(docker)
elif sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
else
    echo "ERROR: Không thể truy cập Docker. Hãy chạy bằng user ec2-user có quyền sudo."
    exit 1
fi

for command_name in git curl nvidia-smi; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: Thiếu lệnh ${command_name}."
        exit 1
    fi
done

if [[ ! -f "${MODEL_FILE}" ]]; then
    echo "ERROR: Không tìm thấy ${REPO_DIR}/${MODEL_FILE}."
    echo "Model không nằm trên GitHub nên phải được giữ lại trên EC2."
    exit 1
fi

if [[ "$(git branch --show-current)" != "main" ]]; then
    echo "ERROR: Repo EC2 phải đang ở nhánh main."
    exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "ERROR: EC2 có thay đổi trên các file đã được Git theo dõi."
    echo "Chạy 'git status' và xử lý thay đổi trước khi triển khai."
    exit 1
fi

echo "[1/7] Kiểm tra GPU và Docker Compose"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${DOCKER[@]}" compose version
"${DOCKER[@]}" compose -f "${COMPOSE_FILE}" config --quiet

available_kb="$(df -Pk "${REPO_DIR}" | awk 'NR == 2 {print $4}')"
if (( available_kb < 8 * 1024 * 1024 )); then
    echo "WARNING: Ổ đĩa còn dưới 8 GiB; quá trình build có thể hết dung lượng."
fi

echo "[2/7] Tải phiên bản mới nhất từ GitHub"
git fetch origin main
git merge --ff-only origin/main

echo "[3/7] Kiểm tra lại cấu hình sau khi cập nhật"
"${DOCKER[@]}" compose -f "${COMPOSE_FILE}" config --quiet

echo "[4/7] Lưu image hiện tại để có thể rollback"
if "${DOCKER[@]}" image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    "${DOCKER[@]}" image tag "${IMAGE_NAME}" "${ROLLBACK_IMAGE}"
else
    echo "Chưa có image ${IMAGE_NAME}; bỏ qua bước tạo bản rollback."
fi

echo "[5/7] Build image GPU mới (container hiện tại vẫn tiếp tục chạy)"
"${DOCKER[@]}" compose -f "${COMPOSE_FILE}" build app

echo "[6/7] Kiểm tra CUDA và suy luận YOLOv10 trong image mới"
"${DOCKER[@]}" run --rm --gpus all \
    -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
    -e MODEL_PATH="/app/${MODEL_FILE}" \
    --mount "type=bind,source=${REPO_DIR}/${MODEL_FILE},target=/app/${MODEL_FILE},readonly" \
    "${IMAGE_NAME}" \
    python -c "import numpy as np, torch, onnxruntime as ort; from ultralytics import YOLO; from benchmark_core.config import DEFAULT_MODEL_PATH, INPUT_IMAGE_SIZE, MAX_DETECTIONS, PREDICT_CONF, PREDICT_IOU; assert torch.cuda.is_available(), 'CUDA không khả dụng'; providers = ort.get_available_providers(); assert 'CUDAExecutionProvider' in providers, providers; model = YOLO(DEFAULT_MODEL_PATH, task='detect'); result = model.predict(np.zeros((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3), dtype=np.uint8), imgsz=INPUT_IMAGE_SIZE, device=0, conf=PREDICT_CONF, iou=PREDICT_IOU, max_det=MAX_DETECTIONS, verbose=False); print('GPU:', torch.cuda.get_device_name(0)); print('ORT providers:', providers); print('YOLOv10 inference OK:', result[0].speed)"

echo "[7/7] Thay container và đợi health check"
"${DOCKER[@]}" compose -f "${COMPOSE_FILE}" up -d --no-build --force-recreate

for attempt in $(seq 1 45); do
    if curl --fail --silent --show-error "${HEALTH_URL}" >/dev/null 2>&1; then
        echo "DEPLOY THÀNH CÔNG: ${HEALTH_URL} trả về OK."
        "${DOCKER[@]}" compose -f "${COMPOSE_FILE}" ps
        exit 0
    fi

    sleep 2
done

echo "ERROR: Website chưa healthy sau 90 giây. Log gần nhất:"
"${DOCKER[@]}" compose -f "${COMPOSE_FILE}" logs --tail 100 app
echo "Image trước đó (nếu có) được giữ tại ${ROLLBACK_IMAGE}."
exit 1
