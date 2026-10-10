import streamlit as st
import cv2
import numpy as np
import time
import os
import tempfile
from PIL import Image
from ultralytics import YOLO
import onnxruntime as ort

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

# Mặc định file mô hình nằm trong cùng thư mục chạy code
DEFAULT_MODEL_PATH = "yolov8n-vehicle.onnx"

st.sidebar.subheader("📂 1. Tải lên mô hình (Model)")
uploaded_model = st.sidebar.file_uploader("Chọn file mô hình YOLO định dạng (.onnx)", type=['onnx'])

# Xử lý lấy đường dẫn mô hình (từ upload hoặc mặc định)
model_path = DEFAULT_MODEL_PATH
if uploaded_model is not None:
    # Lưu file mô hình do người dùng upload ra file tạm để chạy
    with tempfile.NamedTemporaryFile(delete=False, suffix='.onnx') as tmp_model:
        tmp_model.write(uploaded_model.read())
        model_path = tmp_model.name

# Bắt lỗi (try-except) kiểm tra xem file mô hình có tồn tại không
model_exists = os.path.exists(model_path)

if model_exists:
    try:
        # Đo kích thước file mô hình (MB)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Lấy thông tin thiết bị đang chạy từ onnxruntime (kiểm tra có hỗ trợ GPU/CUDA không)
        available_providers = ort.get_available_providers()
        device = "GPU" if "CUDAExecutionProvider" in available_providers else "CPU"
        
        st.sidebar.success("✅ Mô hình đã sẵn sàng!")
        st.sidebar.info(f"""
        **Thông tin mô hình:**
        - **Kích thước file:** {model_size_mb:.2f} MB
        - **Định dạng:** ONNX
        - **Thiết bị chạy:** {device}
        """)
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi đọc thông tin mô hình: {e}")
        model_exists = False
else:
    st.sidebar.error(f"⚠️ Không tìm thấy mô hình mặc định: `{DEFAULT_MODEL_PATH}`. Vui lòng tải lên mô hình ONNX của bạn.")

st.sidebar.markdown("---")
st.sidebar.subheader("📥 2. Dữ liệu đầu vào (Input Data)")
input_file = st.sidebar.file_uploader("Tải lên Ảnh (.jpg, .png) hoặc Video (.mp4)", type=['jpg', 'jpeg', 'png', 'mp4'])

# ==========================================
# PHẦN XỬ LÝ CHÍNH (MAIN CONTENT)
# ==========================================

# Hàm load model (Sử dụng cache để không phải load lại mỗi khi tương tác UI)
@st.cache_resource
def load_model(path):
    try:
        # Sử dụng ultralytics (được hỗ trợ ngầm bởi onnxruntime) để load ONNX model
        model = YOLO(path, task='detect')
        return model
    except Exception as e:
        st.error(f"❌ Lỗi ngoại lệ khi load mô hình: {e}")
        return None

# Tiến hành load mô hình nếu file tồn tại
model = load_model(model_path) if model_exists else None

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
                t0 = time.time()
                img_array = np.array(image)
                # Chuyển hệ màu từ RGB sang BGR để phù hợp với OpenCV & YOLO
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                t1 = time.time()
                
                # 2. Inference (Chạy nhận diện với mô hình)
                results = model(img_bgr, verbose=False)
                t2 = time.time()
                
                # 3. Post-processing (Hậu xử lý - Vẽ Bounding Box)
                # Plotting bounding box và label lên ảnh
                res_plotted = results[0].plot()
                # Chuyển lại hệ màu BGR sang RGB để Streamlit hiển thị đúng màu
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                t3 = time.time()
                
                # Tính toán thời gian (đơn vị: ms)
                pre_process_time = (t1 - t0) * 1000
                inference_time = (t2 - t1) * 1000
                post_process_time = (t3 - t2) * 1000
                total_latency = pre_process_time + inference_time + post_process_time
                fps = 1000 / total_latency if total_latency > 0 else 0
                
                # Hiển thị kết quả bằng bảng đồng hồ (Metrics)
                with metrics_placeholder.container():
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Pre-processing Time", f"{pre_process_time:.2f} ms")
                    mc2.metric("Inference Time", f"{inference_time:.2f} ms")
                    mc3.metric("Post-processing Time", f"{post_process_time:.2f} ms")
                    
                    mc4, mc5 = st.columns(2)
                    mc4.metric("Total Latency (Độ trễ)", f"{total_latency:.2f} ms")
                    mc5.metric("FPS (Tốc độ khung hình)", f"{fps:.2f} FPS")
                
                # Trực quan hóa ảnh kết quả
                result_placeholder.image(res_rgb, use_container_width=True)
                
            elif file_type == 'mp4':
                # ==============================
                # XỬ LÝ VIDEO (VIDEO PROCESSING)
                # ==============================
                
                # Lưu file video ra bộ nhớ tạm trên ổ cứng để OpenCV có thể đọc theo từng frame
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(input_file.read())
                tfile_name = tfile.name
                
                cap = cv2.VideoCapture(tfile_name)
                
                # Gọi các vùng trống để sẵn sàng hiển thị stream
                stframe = result_placeholder.empty()
                metrics_container = metrics_placeholder.container()
                
                # Khởi tạo các biến tích lũy để tính trung bình thời gian thực thi (Average Metrics)
                total_pre, total_inf, total_post, total_frames = 0, 0, 0, 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break # Video kết thúc
                    
                    # 1. Pre-processing
                    t0 = time.time()
                    # (Frame từ OpenCV đã là BGR, nếu cần resize thêm thì làm ở đây)
                    t1 = time.time()
                    
                    # 2. Inference (Nhận diện đối tượng trên frame)
                    results = model(frame, verbose=False)
                    t2 = time.time()
                    
                    # 3. Post-processing (Vẽ Bounding Box)
                    res_plotted = results[0].plot()
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    t3 = time.time()
                    
                    # Tính toán độ trễ của frame hiện tại (ms)
                    pre_time = (t1 - t0) * 1000
                    inf_time = (t2 - t1) * 1000
                    post_time = (t3 - t2) * 1000
                    
                    # Cập nhật tổng thời gian
                    total_pre += pre_time
                    total_inf += inf_time
                    total_post += post_time
                    total_frames += 1
                    
                    # Tính trung bình thời gian cho đến frame hiện tại
                    avg_pre = total_pre / total_frames
                    avg_inf = total_inf / total_frames
                    avg_post = total_post / total_frames
                    avg_latency = avg_pre + avg_inf + avg_post
                    current_fps = 1000 / avg_latency if avg_latency > 0 else 0
                    
                    # Cập nhật hình ảnh lên màn hình (Real-time)
                    stframe.image(res_rgb, use_container_width=True)
                    
                    # Cập nhật thông số đồng hồ đo liên tục
                    with metrics_container:
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Avg Pre-processing", f"{avg_pre:.2f} ms")
                        mc2.metric("Avg Inference", f"{avg_inf:.2f} ms")
                        mc3.metric("Avg Post-processing", f"{avg_post:.2f} ms")
                        
                        mc4, mc5 = st.columns(2)
                        mc4.metric("Avg Total Latency", f"{avg_latency:.2f} ms")
                        mc5.metric("Real-time FPS", f"{current_fps:.2f} FPS")
                        
                cap.release()
                st.success("✅ Hoàn thành phân tích video!")
