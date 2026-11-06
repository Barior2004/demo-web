import streamlit as st
import cv2
import numpy as np
import time
import os
import tempfile
import hashlib
import threading
import sqlite3
from PIL import Image
from ultralytics import YOLO
import onnxruntime as ort
import torch
from benchmark_core.config import (
    BENCHMARK_DB_PATH,
    DEFAULT_MODEL_PATH,
    IMAGE_BENCHMARK_RUNS,
    INPUT_IMAGE_SIZE,
    MAX_DETECTIONS,
    PREDICT_CONF,
    PREDICT_IOU,
    WARMUP_RUNS,
    validate_config,
)
from benchmark_core.video_output import H264VideoWriter
from benchmark_core.metrics import summarize_samples, summarize_video
from benchmark_core.reporting import (
    BenchmarkStore, new_record, runtime_versions, sha256_file, utc_now,
)
validate_config()
# ==========================================
# CẤU HÌNH TRANG VÀ TIÊU ĐỀ
# ==========================================
st.set_page_config(page_title="Dashboard Giám Sát An Ninh", layout="wide", page_icon="🛡️")

st.title("🛡️ Dashboard Giám Sát An Ninh Thông Minh & Đánh Giá Hiệu Năng")
st.markdown("Hệ thống nhận diện phương tiện/người sử dụng mô hình YOLO và đo lường hiệu năng thực thi (Benchmark).")

# ==========================================
# GIAO DIỆN BÊN TRÁI (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Cấu hình hệ thống")

model_path = DEFAULT_MODEL_PATH
model_exists = os.path.isfile(model_path)

# Kiểm tra cả PyTorch CUDA và ONNX Runtime CUDA.
available_providers = ort.get_available_providers()

gpu_ready = (
    torch.cuda.is_available()
    and "CUDAExecutionProvider" in available_providers
)

# Mặc định bật để tránh vô tình đo CPU trên EC2 GPU.
require_gpu = st.sidebar.checkbox(
    "Bắt buộc sử dụng NVIDIA GPU",
    value=True,
    help=(
        "Khi test EC2 g4dn.xlarge nên luôn bật. "
        "Chỉ tắt nếu muốn chạy thử trên máy không có GPU."
    )
)

if gpu_ready:
    INFERENCE_DEVICE = 0
    device_name = torch.cuda.get_device_name(0)

    st.sidebar.success(
        f"✅ GPU đã sẵn sàng: {device_name}"
    )
else:
    INFERENCE_DEVICE = "cpu"
    device_name = "CPU"

    st.sidebar.error(
        "❌ Chưa phát hiện đầy đủ NVIDIA CUDA."
    )

    st.sidebar.code(
        "\n".join(available_providers),
        language="text"
    )

if require_gpu and not gpu_ready:
    st.error(
        "Ứng dụng đang yêu cầu GPU nhưng CUDA chưa sẵn sàng. "
        "Trên EC2, hãy kiểm tra NVIDIA driver, PyTorch CUDA và "
        "onnxruntime-gpu. Nếu chỉ muốn thử bằng CPU trên máy cá "
        "nhân, hãy bỏ chọn 'Bắt buộc sử dụng NVIDIA GPU'."
    )
    st.stop()

if model_exists:
    model_size_mb = (
        os.path.getsize(model_path) / (1024 * 1024)
    )

    st.sidebar.info(
        f"""
**Thông tin benchmark**

- Model: `{os.path.basename(model_path)}`
- Kích thước: `{model_size_mb:.2f} MB`
- Thiết bị: `{device_name}`
- Ultralytics device: `{INFERENCE_DEVICE}`
- Warm-up: `{WARMUP_RUNS} lần`
- Benchmark ảnh: `{IMAGE_BENCHMARK_RUNS} lần`
"""
    )
else:
    st.error(
        f"❌ Không tìm thấy model `{model_path}`. "
        "Không tìm thấy model YOLOv10. "
        "Hãy kiểm tra file "
        "`models/yolov10n_final_coordattn.onnx`."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("📥 2. Dữ liệu đầu vào (Input Data)")
input_file = st.sidebar.file_uploader(
    "Tải lên Ảnh (.jpg, .png) hoặc Video (.mp4)",
    type=["jpg", "jpeg", "png", "mp4"]
)

DISPLAY_EVERY = st.sidebar.slider(
    "Hiển thị mỗi N frame",
    min_value=1,
    max_value=10,
    value=1,
    help=(
        "Giá trị 1 phát preview detection theo thời gian thực. "
        "Giá trị lớn hơn giảm số lần cập nhật giao diện. Video "
        "kết quả vẫn chứa detection trên mọi frame."
    )
)
# ==========================================
# PHẦN XỬ LÝ CHÍNH (MAIN CONTENT)
# ==========================================

# Hàm load model (Sử dụng cache để không phải load lại mỗi khi tương tác UI)
@st.cache_resource
def load_model(path, inference_device, model_sha256):
    """
    Load model một lần và warm-up trước khi benchmark.
    Warm-up loại bỏ thời gian khởi tạo CUDA khỏi kết quả.
    """
    loaded_model = YOLO(path, task="detect")

    dummy_frame = np.zeros(
        (
            INPUT_IMAGE_SIZE,
            INPUT_IMAGE_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    for _ in range(WARMUP_RUNS):
        loaded_model.predict(
            source=dummy_frame,
            device=inference_device,
            imgsz=INPUT_IMAGE_SIZE,
            conf=PREDICT_CONF,
            iou=PREDICT_IOU,
            max_det=MAX_DETECTIONS,
            verbose=False,
        )

    # Streamlit shares cached resources between sessions. Serialize predict
    # calls so another session cannot overlap or mutate the same predictor.
    return loaded_model, threading.RLock()


def measured_predict(frame):
    with model_lock:
        if INFERENCE_DEVICE != "cpu":
            torch.cuda.synchronize(INFERENCE_DEVICE)
        started = time.perf_counter_ns()
        predictions = model.predict(
            source=frame,
            device=INFERENCE_DEVICE,
            imgsz=INPUT_IMAGE_SIZE,
            conf=PREDICT_CONF,
            iou=PREDICT_IOU,
            max_det=MAX_DETECTIONS,
            verbose=False,
        )
        if INFERENCE_DEVICE != "cpu":
            torch.cuda.synchronize(INFERENCE_DEVICE)
        duration_ms = (time.perf_counter_ns() - started) / 1_000_000
    return predictions, duration_ms


def save_run(record, metrics):
    record["metrics"] = metrics
    record["completed_at_utc"] = utc_now()
    backend = getattr(getattr(model, "predictor", None), "model", None)
    session = getattr(backend, "session", None)
    record["environment"]["session_providers"] = (
        session.get_providers() if session is not None else []
    )
    try:
        run_id = BenchmarkStore(BENCHMARK_DB_PATH).save_completed(record)
        st.session_state["last_benchmark_run_id"] = run_id
        st.success(f"Đã lưu kết quả benchmark — mã lần chạy: {run_id}")
    except (OSError, ValueError, sqlite3.Error) as error:
        st.error(f"Phân tích hoàn tất nhưng KHÔNG lưu được kết quả: {error}")


def show_video_metrics(metrics, stages, source_fps):
    model_stats = metrics["model_call"]
    with metrics_placeholder.container():
        st.caption(
            f"Video gốc: {source_fps:.2f} FPS | "
            f"Đã xử lý: {metrics['frame_count']} frame | "
            f"Preview mỗi {DISPLAY_EVERY} frame"
        )
        columns = st.columns(3)
        for column, label, key in zip(columns, (
            "Decode trung bình", "Inference nội bộ", "Vẽ kết quả trung bình"
        ), ("decode", "inference_internal", "render")):
            column.metric(label, f"{stages[key]['mean_ms']:.2f} ms")
        columns = st.columns(3)
        for column, label, key in zip(columns, (
            "Model call trung bình", "Model call p50", "Model call p95"
        ), ("mean_ms", "p50_ms", "p95_ms")):
            column.metric(label, f"{model_stats[key]:.2f} ms")
        columns = st.columns(3)
        for column, label, key in zip(columns, (
            "Xử lý chủ động phía server", "Thông lượng vòng lặp server", "Gửi preview từ server"
        ), ("active_processing_fps", "server_wall_fps", "preview_submission_fps")):
            value = metrics[key]
            column.metric(label, f"{value:.2f} FPS" if value is not None else "N/A")
        columns = st.columns(3)
        model_fps = model_stats["fps"]
        columns[0].metric("Model FPS", f"{model_fps:.2f} FPS" if model_fps is not None else "N/A")
        if "encode_submit" in stages:
            columns[1].metric("Gửi frame tới encoder", f"{stages['encode_submit']['mean_ms']:.2f} ms")
        columns[2].metric("Hoàn tất MP4", f"{metrics['encoder_finalize_ms']:.2f} ms")
        st.caption(
            "Xử lý chủ động loại thời gian UI/chờ nhịp video; "
            "vòng lặp server có tính UI/chờ. Gửi preview là số ảnh gửi "
            "cho Streamlit, không phải FPS hiển thị trên trình duyệt. "
            "Số liệu trong lúc chạy là tạm thời; kết quả cuối tính cả hoàn tất MP4."
        )


model = None

if model_exists:
    try:
        with st.spinner(
            f"Đang load và warm-up model "
            f"{WARMUP_RUNS} lần..."
        ):
            model_sha256 = sha256_file(model_path)
            model, model_lock = load_model(
                model_path,
                INFERENCE_DEVICE,
                model_sha256,
            )

        st.success("✅ Model đã load và warm-up thành công.")

    except Exception as e:
        st.error(f"❌ Không thể load model: {e}")
        st.stop()

if input_file is not None and model is not None:
    # Xác định loại file tải lên (ảnh hay video)
    file_type = input_file.name.split('.')[-1].lower()
    
    # Chia giao diện làm 2 cột
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Dữ liệu gốc")
        if file_type in ['jpg', 'jpeg', 'png']:
            # Đọc và hiển thị ảnh
            image = Image.open(input_file).convert("RGB")
            st.image(image, use_container_width=True)
        elif file_type == 'mp4':
            # Hiển thị video
            st.video(input_file)
            
    with col2:
        st.subheader("🎯 Kết quả Nhận diện & Benchmark")
        # Nút bấm chạy phân tích
        start_btn = st.button("🚀 Bắt đầu Phân tích", use_container_width=True)
        
        # Khởi tạo các vùng trống (placeholder) để có thể cập nhật nội dung tự động sau này
        metrics_placeholder = st.empty()
        result_placeholder = st.empty()
        
        if start_btn:
            record = new_record(
                "video" if file_type == "mp4" else "image",
                model={"name": os.path.basename(model_path), "sha256": model_sha256},
                input_info={
                    "name": os.path.basename(input_file.name),
                    "sha256": hashlib.sha256(input_file.getbuffer()).hexdigest(),
                    "size_bytes": input_file.size,
                },
                config={
                    "imgsz": INPUT_IMAGE_SIZE, "conf": PREDICT_CONF,
                    "iou": PREDICT_IOU, "max_det": MAX_DETECTIONS,
                    "warmup_runs": WARMUP_RUNS, "warmup": "cached_model_synthetic_frames",
                    "image_runs": IMAGE_BENCHMARK_RUNS,
                    "display_every": DISPLAY_EVERY, "pacing": file_type == "mp4",
                    "gpu_synchronized": INFERENCE_DEVICE != "cpu",
                },
                environment={
                    **runtime_versions(), "device": device_name,
                    "available_providers": available_providers,
                    "torch_cuda_version": torch.version.cuda,
                    "app_revision": os.getenv("APP_REVISION", "unknown"),
                },
            )
            if file_type in ['jpg', 'jpeg', 'png']:
                image_wall_start = time.perf_counter_ns()
                # ==============================
                # XỬ LÝ ẢNH (IMAGE PROCESSING)
                # ==============================
                
                # 1. Pre-processing (Tiền xử lý)
                # Chuyển ảnh sang định dạng OpenCV.
                convert_start = time.perf_counter_ns()

                image_array = np.array(image)
                image_bgr = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR
                )

                convert_ms = (
                    time.perf_counter_ns() - convert_start
                ) / 1_000_000

                preprocess_times = []
                inference_times = []
                postprocess_times = []
                model_call_times = []

                progress = st.progress(
                    0,
                    text="Đang chạy benchmark ảnh..."
                )

                results = None

                # Chạy nhiều lần để kết quả không bị phụ thuộc
                # vào một lần inference ngẫu nhiên.
                for run_index in range(IMAGE_BENCHMARK_RUNS):
                    results, model_call_ms = measured_predict(image_bgr)

                    speed = results[0].speed or {}

                    preprocess_times.append(
                        float(speed.get("preprocess", 0.0))
                    )
                    inference_times.append(
                        float(speed.get("inference", 0.0))
                    )
                    postprocess_times.append(
                        float(speed.get("postprocess", 0.0))
                    )
                    model_call_times.append(model_call_ms)

                    progress.progress(
                        (run_index + 1) / IMAGE_BENCHMARK_RUNS,
                        text=(
                            "Đang chạy benchmark ảnh: "
                            f"{run_index + 1}/"
                            f"{IMAGE_BENCHMARK_RUNS}"
                        )
                    )

                progress.empty()

                # Chỉ vẽ kết quả một lần sau khi benchmark xong.
                render_start = time.perf_counter_ns()

                plotted_result = results[0].plot()
                result_rgb = cv2.cvtColor(
                    plotted_result,
                    cv2.COLOR_BGR2RGB
                )

                render_ms = (
                    time.perf_counter_ns() - render_start
                ) / 1_000_000

                image_wall_seconds = (
                    time.perf_counter_ns() - image_wall_start
                ) / 1_000_000_000
                model_stats = summarize_samples(model_call_times)

                avg_preprocess = float(
                    np.mean(preprocess_times)
                )
                avg_inference = float(
                    np.mean(inference_times)
                )
                avg_postprocess = float(
                    np.mean(postprocess_times)
                )
                avg_model_call = model_stats["mean_ms"]

                p50_model = model_stats["p50_ms"]
                p95_model = model_stats["p95_ms"]

                model_fps = model_stats["fps"]

                with metrics_placeholder.container():
                    st.caption(
                        f"Kết quả sau "
                        f"{IMAGE_BENCHMARK_RUNS} lần chạy"
                    )

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric(
                        "Preprocess trung bình",
                        f"{avg_preprocess:.2f} ms"
                    )
                    mc2.metric(
                        "Inference trung bình",
                        f"{avg_inference:.2f} ms"
                    )
                    mc3.metric(
                        "Postprocess trung bình",
                        f"{avg_postprocess:.2f} ms"
                    )

                    mc4, mc5, mc6 = st.columns(3)
                    mc4.metric(
                        "Model call trung bình",
                        f"{avg_model_call:.2f} ms"
                    )
                    mc5.metric(
                        "Model call p50",
                        f"{p50_model:.2f} ms"
                    )
                    mc6.metric(
                        "Model call p95",
                        f"{p95_model:.2f} ms"
                    )

                    mc7, mc8, mc9 = st.columns(3)
                    mc7.metric(
                        "Model FPS",
                        f"{model_fps:.2f} FPS" if model_fps is not None else "N/A"
                    )
                    mc8.metric(
                        "RGB → BGR",
                        f"{convert_ms:.2f} ms"
                    )
                    mc9.metric(
                        "Vẽ kết quả",
                        f"{render_ms:.2f} ms"
                    )

                result_placeholder.image(
                    result_rgb,
                    use_container_width=True
                )
                st.caption("Model FPS đo lời gọi predict đã đồng bộ GPU; không bao gồm vẽ và hiển thị.")
                record["input"].update({"width": image.width, "height": image.height})
                save_run(record, {
                    "model_call": model_stats,
                    "preprocess_internal": summarize_samples(preprocess_times),
                    "inference_internal": summarize_samples(inference_times),
                    "postprocess_internal": summarize_samples(postprocess_times),
                    "convert_ms_once": convert_ms, "render_ms_once": render_ms,
                    "benchmark_wall_seconds": image_wall_seconds,
                })
                
            elif file_type == 'mp4':
                # ==============================
                # XỬ LÝ VIDEO (VIDEO PROCESSING)
                # ==============================
                
                # Lưu file video ra bộ nhớ tạm trên ổ cứng để OpenCV có thể đọc theo từng frame
                temp_video_path = None
                output_video_path = None
                cap = None
                video_writer = None

                try:
                    # Lưu xong và đóng file trước khi OpenCV mở.
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4"
                    ) as temp_video:
                        temp_video.write(input_file.getbuffer())
                        temp_video_path = temp_video.name

                    cap = cv2.VideoCapture(temp_video_path)

                    if not cap.isOpened():
                        st.error("❌ Không thể mở file video.")
                        st.stop()

                    source_fps = cap.get(cv2.CAP_PROP_FPS)
                    output_fps = (
                        float(source_fps)
                        if np.isfinite(source_fps) and source_fps > 0
                        else 30.0
                    )
                    frame_width = int(
                        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    )
                    frame_height = int(
                        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    )

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4",
                    ) as output_video:
                        output_video_path = output_video.name

                    video_writer = H264VideoWriter(
                        output_video_path,
                        width=frame_width,
                        height=frame_height,
                        fps=output_fps,
                    ).open()

                    stframe = result_placeholder.empty()

                    # Danh sách lưu kết quả của từng frame.
                    decode_times = []
                    preprocess_times = []
                    inference_times = []
                    postprocess_times = []
                    model_call_times = []
                    render_times = []
                    encode_times = []
                    compute_times = []

                    total_frames = 0
                    preview_submissions = 0
                    last_metrics_update = float("-inf")
                    benchmark_start = time.perf_counter_ns()
                    preview_interval_seconds = (
                        DISPLAY_EVERY / output_fps
                    )
                    next_preview_deadline = time.perf_counter()

                    while cap.isOpened():
                        # 1. Đọc frame từ video.
                        active_frame_start = time.perf_counter_ns()
                        decode_start = active_frame_start
                        ret, frame = cap.read()
                        decode_ms = (
                            time.perf_counter_ns() - decode_start
                        ) / 1_000_000

                        if not ret:
                            break

                        # 2. Chạy model và bắt buộc sử dụng thiết bị đã chọn.
                        results, model_call_ms = measured_predict(frame)

                        # Ultralytics cung cấp thời gian nội bộ.
                        speed = results[0].speed or {}

                        preprocess_ms = float(
                            speed.get("preprocess", 0.0)
                        )
                        inference_ms = float(
                            speed.get("inference", 0.0)
                        )
                        postprocess_ms = float(
                            speed.get("postprocess", 0.0)
                        )

                        # 3. Vẽ detection trên mọi frame để tạo video output.
                        render_start = time.perf_counter_ns()

                        plotted_frame = results[0].plot()

                        render_ms = (
                            time.perf_counter_ns() - render_start
                        ) / 1_000_000

                        # 4. Mã hóa frame đã vẽ thành MP4 H.264.
                        encode_start = time.perf_counter_ns()

                        video_writer.write(plotted_frame)

                        encode_ms = (
                            time.perf_counter_ns() - encode_start
                        ) / 1_000_000

                        # DISPLAY_EVERY chỉ giảm số lần cập nhật preview.
                        should_display = (
                            total_frames == 0
                            or (total_frames + 1) % DISPLAY_EVERY == 0
                        )

                        result_rgb = None

                        if should_display:
                            result_rgb = cv2.cvtColor(
                                plotted_frame,
                                cv2.COLOR_BGR2RGB,
                            )

                        compute_ms = (
                            time.perf_counter_ns() - active_frame_start
                        ) / 1_000_000

                        decode_times.append(decode_ms)
                        preprocess_times.append(preprocess_ms)
                        inference_times.append(inference_ms)
                        postprocess_times.append(postprocess_ms)
                        model_call_times.append(model_call_ms)
                        render_times.append(render_ms)
                        encode_times.append(encode_ms)
                        compute_times.append(compute_ms)

                        total_frames += 1

                        # Không gửi mọi frame lên trình duyệt.
                        # Việc này tránh Streamlit làm sai lệch FPS của model.
                        if should_display and result_rgb is not None:
                            stframe.image(
                                result_rgb,
                                use_container_width=True,
                            )
                            preview_submissions += 1

                            elapsed_seconds = (
                                time.perf_counter_ns()
                                - benchmark_start
                            ) / 1_000_000_000

                            if time.perf_counter() - last_metrics_update >= 0.5:
                                live_metrics = summarize_video(
                                    model_call_times, compute_times,
                                    elapsed_seconds, 0.0, preview_submissions,
                                )
                                show_video_metrics(live_metrics, {
                                    "decode": summarize_samples(decode_times),
                                    "inference_internal": summarize_samples(inference_times),
                                    "render": summarize_samples(render_times),
                                    "encode_submit": summarize_samples(encode_times),
                                }, output_fps)
                                last_metrics_update = time.perf_counter()

                            # Đồng bộ preview với FPS video gốc để
                            # trình duyệt có thời gian hiển thị từng frame.
                            next_preview_deadline += (
                                preview_interval_seconds
                            )
                            preview_wait_seconds = (
                                next_preview_deadline
                                - time.perf_counter()
                            )

                            if preview_wait_seconds > 0:
                                time.sleep(preview_wait_seconds)
                            else:
                                # Nếu xử lý chậm hơn realtime, không cố
                                # chạy bù vì sẽ làm preview giật theo cụm.
                                next_preview_deadline = time.perf_counter()

                    finalize_start = time.perf_counter_ns()
                    video_writer.close()
                    finalize_ms = (
                        time.perf_counter_ns() - finalize_start
                    ) / 1_000_000
                    video_writer = None
                    final_wall_seconds = (
                        time.perf_counter_ns() - benchmark_start
                    ) / 1_000_000_000

                    if total_frames == 0:
                        st.error("❌ Video không có frame hợp lệ.")
                    else:
                        if os.path.getsize(output_video_path) == 0:
                            raise RuntimeError("Encoder không tạo được video kết quả")
                        stages = {
                            "decode": summarize_samples(decode_times),
                            "preprocess_internal": summarize_samples(preprocess_times),
                            "inference_internal": summarize_samples(inference_times),
                            "postprocess_internal": summarize_samples(postprocess_times),
                            "render": summarize_samples(render_times),
                            "encode_submit": summarize_samples(encode_times),
                        }
                        final_metrics = summarize_video(
                            model_call_times, compute_times,
                            final_wall_seconds, finalize_ms, preview_submissions,
                        )
                        final_metrics["stages"] = stages
                        show_video_metrics(final_metrics, stages, output_fps)
                        record["input"].update({
                            "width": frame_width, "height": frame_height,
                            "source_fps": float(source_fps) if np.isfinite(source_fps) else None,
                            "output_fps": output_fps,
                        })
                        save_run(record, final_metrics)
                        stframe.video(
                            output_video_path,
                            format="video/mp4",
                            autoplay=True,
                            muted=True,
                        )
                        st.success(
                            f"✅ Đã xử lý xong {total_frames} frame. "
                            "Video kết quả chứa detection trên mọi frame."
                        )

                except Exception as e:
                    st.error(f"❌ Lỗi xử lý video: {e}")

                finally:
                    if cap is not None:
                        cap.release()

                    if video_writer is not None:
                        video_writer.close()

                    if (
                        temp_video_path is not None
                        and os.path.exists(temp_video_path)
                    ):
                        os.remove(temp_video_path)

                    if (
                        output_video_path is not None
                        and os.path.exists(output_video_path)
                    ):
                        os.remove(output_video_path)
