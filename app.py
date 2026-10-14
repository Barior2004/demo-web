import streamlit as st
import cv2
import numpy as np
import time
import os
import tempfile
from PIL import Image
from ultralytics import YOLO
import onnxruntime as ort
import torch

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

# Model cố định để kết quả benchmark giữa các máy
# có thể so sánh công bằng.
DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "best_v11.onnx"
)

WARMUP_RUNS = 10
IMAGE_BENCHMARK_RUNS = 50

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
    value=False,
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
        "Hãy đặt `best_v11.onnx` cùng thư mục với `app.py`."
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
    value=3,
    help=(
        "Giá trị 1 hiển thị mọi frame nhưng có thể làm Streamlit "
        "chậm. Giá trị 3 phù hợp để đo hiệu năng model."
    )
)
# ==========================================
# PHẦN XỬ LÝ CHÍNH (MAIN CONTENT)
# ==========================================

# Hàm load model (Sử dụng cache để không phải load lại mỗi khi tương tác UI)
@st.cache_resource
def load_model(path, inference_device):
    """
    Load model một lần và warm-up trước khi benchmark.
    Warm-up loại bỏ thời gian khởi tạo CUDA khỏi kết quả.
    """
    loaded_model = YOLO(path, task="detect")

    dummy_frame = np.zeros(
        (640, 640, 3),
        dtype=np.uint8
    )

    for _ in range(WARMUP_RUNS):
        loaded_model.predict(
            source=dummy_frame,
            device=inference_device,
            verbose=False
        )

    return loaded_model


model = None

if model_exists:
    try:
        with st.spinner(
            f"Đang load và warm-up model "
            f"{WARMUP_RUNS} lần..."
        ):
            model = load_model(
                model_path,
                INFERENCE_DEVICE
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
            if file_type in ['jpg', 'jpeg', 'png']:
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
                    model_start = time.perf_counter_ns()

                    results = model.predict(
                        source=image_bgr,
                        device=INFERENCE_DEVICE,
                        verbose=False
                    )

                    model_call_ms = (
                        time.perf_counter_ns() - model_start
                    ) / 1_000_000

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

                avg_preprocess = float(
                    np.mean(preprocess_times)
                )
                avg_inference = float(
                    np.mean(inference_times)
                )
                avg_postprocess = float(
                    np.mean(postprocess_times)
                )
                avg_model_call = float(
                    np.mean(model_call_times)
                )

                p50_model = float(
                    np.percentile(model_call_times, 50)
                )
                p95_model = float(
                    np.percentile(model_call_times, 95)
                )

                model_fps = (
                    1000 / avg_model_call
                    if avg_model_call > 0 else 0
                )

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
                        f"{model_fps:.2f} FPS"
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
                
            elif file_type == 'mp4':
                # ==============================
                # XỬ LÝ VIDEO (VIDEO PROCESSING)
                # ==============================
                
                # Lưu file video ra bộ nhớ tạm trên ổ cứng để OpenCV có thể đọc theo từng frame
                temp_video_path = None
                cap = None

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
                    stframe = result_placeholder.empty()

                    # Danh sách lưu kết quả của từng frame.
                    decode_times = []
                    preprocess_times = []
                    inference_times = []
                    postprocess_times = []
                    model_call_times = []
                    render_times = []
                    compute_times = []

                    total_frames = 0
                    benchmark_start = time.perf_counter_ns()

                    while cap.isOpened():
                        # 1. Đọc frame từ video.
                        decode_start = time.perf_counter_ns()
                        ret, frame = cap.read()
                        decode_ms = (
                            time.perf_counter_ns() - decode_start
                        ) / 1_000_000

                        if not ret:
                            break

                        # 2. Chạy model và bắt buộc sử dụng thiết bị đã chọn.
                        model_start = time.perf_counter_ns()

                        results = model.predict(
                            source=frame,
                            device=INFERENCE_DEVICE,
                            verbose=False
                        )

                        model_call_ms = (
                            time.perf_counter_ns() - model_start
                        ) / 1_000_000

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

                        # 3. Vẽ bounding box.
                        render_start = time.perf_counter_ns()

                        plotted_frame = results[0].plot()
                        result_rgb = cv2.cvtColor(
                            plotted_frame,
                            cv2.COLOR_BGR2RGB
                        )

                        render_ms = (
                            time.perf_counter_ns() - render_start
                        ) / 1_000_000

                        compute_ms = (
                            decode_ms
                            + model_call_ms
                            + render_ms
                        )

                        decode_times.append(decode_ms)
                        preprocess_times.append(preprocess_ms)
                        inference_times.append(inference_ms)
                        postprocess_times.append(postprocess_ms)
                        model_call_times.append(model_call_ms)
                        render_times.append(render_ms)
                        compute_times.append(compute_ms)

                        total_frames += 1

                        # Không gửi mọi frame lên trình duyệt.
                        # Việc này tránh Streamlit làm sai lệch FPS của model.
                        if (
                            total_frames == 1
                            or total_frames % DISPLAY_EVERY == 0
                        ):
                            stframe.image(
                                result_rgb,
                                use_container_width=True
                            )

                            elapsed_seconds = (
                                time.perf_counter_ns()
                                - benchmark_start
                            ) / 1_000_000_000

                            avg_decode = float(np.mean(decode_times))
                            avg_preprocess = float(
                                np.mean(preprocess_times)
                            )
                            avg_inference = float(
                                np.mean(inference_times)
                            )
                            avg_postprocess = float(
                                np.mean(postprocess_times)
                            )
                            avg_model_call = float(
                                np.mean(model_call_times)
                            )
                            avg_render = float(np.mean(render_times))
                            avg_compute = float(np.mean(compute_times))

                            p50_model = float(
                                np.percentile(model_call_times, 50)
                            )
                            p95_model = float(
                                np.percentile(model_call_times, 95)
                            )

                            compute_fps = (
                                1000 / avg_compute
                                if avg_compute > 0 else 0
                            )

                            end_to_end_fps = (
                                total_frames / elapsed_seconds
                                if elapsed_seconds > 0 else 0
                            )

                            with metrics_placeholder.container():
                                st.caption(
                                    f"Video gốc: {source_fps:.2f} FPS | "
                                    f"Đã xử lý: {total_frames} frame | "
                                    f"Hiển thị mỗi {DISPLAY_EVERY} frame"
                                )

                                mc1, mc2, mc3 = st.columns(3)
                                mc1.metric(
                                    "Decode trung bình",
                                    f"{avg_decode:.2f} ms"
                                )
                                mc2.metric(
                                    "Inference trung bình",
                                    f"{avg_inference:.2f} ms"
                                )
                                mc3.metric(
                                    "Vẽ kết quả trung bình",
                                    f"{avg_render:.2f} ms"
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

                                mc7, mc8 = st.columns(2)
                                mc7.metric(
                                    "Compute FPS",
                                    f"{compute_fps:.2f} FPS"
                                )
                                mc8.metric(
                                    "Streamlit FPS",
                                    f"{end_to_end_fps:.2f} FPS"
                                )

                    if total_frames == 0:
                        st.error("❌ Video không có frame hợp lệ.")
                    else:
                        st.success(
                            f"✅ Đã xử lý xong {total_frames} frame."
                        )

                except Exception as e:
                    st.error(f"❌ Lỗi xử lý video: {e}")

                finally:
                    if cap is not None:
                        cap.release()

                    if (
                        temp_video_path is not None
                        and os.path.exists(temp_video_path)
                    ):
                        os.remove(temp_video_path)
