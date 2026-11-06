# Triển khai Demo Web AI lên AWS EC2 GPU

## Lưu benchmark (measurement protocol v1)

Mỗi lần phân tích ảnh/video hoàn tất được lưu vào `data/benchmarks.db` (SQLite),
kèm mã lần chạy, thời gian UTC, SHA256 model/input, cấu hình và phiên bản runtime.
Có thể đổi đường dẫn bằng `BENCHMARK_DB_PATH`. Compose giữ `./data` ngoài container.
Không lưu ảnh/video vào database; trang lịch sử và xuất báo cáo sẽ bổ sung sau.

Các chỉ số video phân biệt xử lý chủ động, thông lượng vòng lặp server và số
preview gửi cho Streamlit; chúng không đo FPS hiển thị thực tế trên trình duyệt.
Xem [phạm vi phép đo và cách kiểm thử](docs/benchmark-measurement.md).


## 1. Kiến trúc hiện tại

```text
Trình duyệt
    ↓ HTTP cổng 8501
AWS EC2 g4dn.xlarge
    ↓
Docker Compose → Streamlit → Ultralytics → ONNX Runtime CUDA
                                             ↓
                                      NVIDIA Tesla T4
```

Quy trình cập nhật:

```text
Push code lên GitHub → GitHub Actions kiểm tra CI
                     → SSH vào EC2 → chạy deploy-ec2.sh
```

Trạng thái hiện tại:

- Website và mô hình đã chạy thành công trên AWS bằng GPU Tesla T4.
- Docker container có healthcheck và tự khởi động lại.
- GitHub Actions tự động kiểm tra code và cấu hình Docker.
- Việc deploy lên EC2 vẫn được kích hoạt thủ công qua `deploy-ec2.sh`, chưa phải CD tự động hoàn toàn.
- Đây là bản benchmark bằng ảnh/video upload, chưa có RTSP camera, tracking, cảnh báo và lịch sử sự kiện.

## 2. Cấu hình đã sử dụng

| Thành phần | Cấu hình |
|---|---|
| AWS region | Sydney `ap-southeast-2` |
| EC2 | `g4dn.xlarge`: 4 vCPU, 16 GiB RAM |
| GPU | NVIDIA Tesla T4, khoảng 16 GB VRAM |
| AMI | Deep Learning OSS NVIDIA Driver AMI GPU PyTorch, Amazon Linux 2023, x86_64 |
| EBS | 50 GiB gp3 |
| Docker image | `demo-web:gpu-optimized` |
| GPU base image | `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime` |
| AI runtime | PyTorch CUDA, Ultralytics, ONNX Runtime GPU |
| Website | Streamlit, cổng 8501 |
| Model | `best_v11.onnx`, input `1 x 3 x 640 x 640` |

## 3. Tạo EC2 trong giao diện AWS Project

Nhóm đăng nhập bằng AWS Builder ID và tạo EC2 trong giao diện **Project**, không dùng giao diện tài khoản root.

1. Chọn region Sydney `ap-southeast-2`.
2. Tạo EC2 `g4dn.xlarge`.
3. Chọn AMI Deep Learning có NVIDIA Driver, Amazon Linux 2023, x86_64.
4. Chọn root EBS 50 GiB gp3.
5. Tạo key pair và tải file `.pem` về Windows.
6. Bật Public IPv4 cho giai đoạn test.
7. Cấu hình inbound Security Group:
   - SSH TCP 22: chỉ `My IP`.
   - Custom TCP 8501: chỉ `My IP`.

Không nên mở cổng 22 hoặc 8501 cho `0.0.0.0/0` vì bản test chưa có HTTPS và đăng nhập.

## 4. Kết nối và kiểm tra EC2

Nếu Windows báo private key có quyền quá mở:

```powershell
$key = "C:\Users\ACER\Downloads\demo-project.pem"
icacls $key /inheritance:r
icacls $key /grant:r "$($env:USERNAME):(R)"
```

Kết nối SSH bằng Public IPv4 hiển thị trong AWS Project:

```powershell
ssh -i "C:\Users\ACER\Downloads\demo-project.pem" ec2-user@<PUBLIC_IPV4>
```

Amazon Linux dùng user `ec2-user`, không dùng `ubuntu`.

Kiểm tra môi trường:

```bash
cat /etc/os-release
nvidia-smi
docker --version
nvidia-ctk --version
sudo docker compose version
sudo docker buildx version
sudo docker run --rm --gpus all ubuntu:latest nvidia-smi
```

Kết quả đúng khi host và container đều nhận GPU `Tesla T4`.

## 5. Triển khai lần đầu

Clone source code:

```bash
cd ~
git clone https://github.com/Barior2004/demo-web.git
cd demo-web
```

File `best_v11.onnx` không nằm trên GitHub. Chép model từ PowerShell local lên EC2:

```powershell
scp -i "C:\Users\ACER\Downloads\demo-project.pem" "C:\Users\ACER\Downloads\demo-web\best_v11.onnx" ec2-user@<PUBLIC_IPV4>:/home/ec2-user/demo-web/best_v11.onnx
```

Trên EC2, kiểm tra model và build image:

```bash
cd ~/demo-web
ls -lh best_v11.onnx
sudo docker compose -f compose.gpu.yaml config --quiet
sudo docker compose -f compose.gpu.yaml build app
```

Kiểm tra CUDA trong image:

```bash
sudo docker run --rm --gpus all demo-web:gpu-optimized \
  python -c "import torch, onnxruntime as ort; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('Providers:', ort.get_available_providers())"
```

Kết quả cần có:

```text
CUDA: True
GPU: Tesla T4
CUDAExecutionProvider
```

Khởi động website:

```bash
sudo docker compose -f compose.gpu.yaml up -d --no-build
sudo docker compose -f compose.gpu.yaml ps
curl --fail http://localhost:8501/_stcore/health
```

Khi health trả về `ok`, mở:

```text
http://<PUBLIC_IPV4>:8501
```

## 6. Cập nhật khi có code mới

Trên máy local:

```powershell
git status
git add <file-da-sua>
git commit -m "Mo ta thay doi"
git push origin main
```

Chờ workflow `CI checks` trong tab GitHub Actions có dấu xanh. Sau đó SSH vào EC2 và chạy:

```bash
cd ~/demo-web
./deploy-ec2.sh
```

Script sẽ pull nhánh `main`, build image mới, kiểm tra CUDA, giữ image cũ để rollback, thay container và chờ healthcheck.

Nếu script báo EC2 có thay đổi trên file Git, kiểm tra trước bằng:

```bash
git status
git diff
```

Không dùng `git reset --hard` khi chưa xác định các thay đổi có cần giữ lại hay không.

## 7. Mở lại website sau khi Stop và Start EC2

1. Trong AWS Project, chọn `Start instance`.
2. Chờ instance `Running` và status checks hoàn tất.
3. Lấy Public IPv4 mới vì IP có thể thay đổi sau khi stop/start.
4. SSH vào EC2.
5. Chạy:

```bash
cd ~/demo-web
sudo systemctl start docker
sudo docker compose -f compose.gpu.yaml up -d --no-build
sudo docker compose -f compose.gpu.yaml ps
curl --fail http://localhost:8501/_stcore/health
```

Nếu không có code mới thì không cần build và không cần chạy `deploy-ec2.sh`.

Sau khi test xong, chọn `Stop instance` trong AWS Project để ngừng phí compute GPU. Không chọn `Terminate` nếu còn muốn giữ EC2 và dữ liệu EBS.

## 8. Theo dõi và xử lý lỗi nhanh

Xem log:

```bash
sudo docker compose -f compose.gpu.yaml logs --tail 100 app
```

Theo dõi GPU và tài nguyên container:

```bash
watch -n 1 nvidia-smi
sudo docker stats demo-web-gpu
```

Kiểm tra dung lượng:

```bash
df -h /
sudo docker system df
```

Nếu website không truy cập được nhưng health local trả về `ok`, kiểm tra Public IPv4 mới và inbound TCP 8501 trong Security Group.

Nếu thiếu model:

```bash
ls -lh ~/demo-web/best_v11.onnx
```

Nếu container còn `health: starting`, chờ tối đa khoảng 90 giây rồi kiểm tra lại `ps`, health endpoint và log.

## 9. Các file chính

| File | Vai trò |
|---|---|
| `app.py` | Website và benchmark |
| `Dockerfile.gpu` | Xây dựng GPU image |
| `compose.gpu.yaml` | Chạy container, cấp GPU, mount model và healthcheck |
| `deploy-ec2.sh` | Cập nhật và triển khai an toàn trên EC2 |
| `.github/workflows/ci.yml` | GitHub Actions CI |
| `requirements-aws-gpu.txt` | Phiên bản thư viện AWS GPU |

