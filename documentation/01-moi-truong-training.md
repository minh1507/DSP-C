# Môi Trường Training

## Tổng Quan

Tài liệu này trình bày chi tiết về môi trường training cho hệ thống phát hiện lỗ hổng bảo mật trong mã nguồn sử dụng các kỹ thuật Deep Learning.

## Python Version

**Python 3.9+**

- Theo cấu hình trong Dockerfile (`docker/Dockerfile`), hệ thống được thiết kế để hoạt động với Python 3.9
- Khuyến nghị sử dụng Python 3.9 hoặc phiên bản cao hơn để đảm bảo tính tương thích với toàn bộ các thư viện phụ thuộc

## PyTorch

**PyTorch >= 2.0.0**

- Hệ thống yêu cầu PyTorch phiên bản 2.0.0 trở lên
- PyTorch đóng vai trò là framework chính trong việc xây dựng và huấn luyện các mô hình deep learning

### Cài đặt PyTorch với hỗ trợ CUDA (khuyến nghị khi có GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision torchaudio
```

## Các Thư Viện Chính

Dựa trên file cấu hình `requirements.txt`, các thư viện chính được sử dụng trong hệ thống bao gồm:

### Deep Learning & Neural Networks

1. **torch >= 2.0.0**
   - Framework deep learning chính của hệ thống
   - Được sử dụng cho toàn bộ các kiến trúc mô hình (MLP, GCN, GAT)

2. **torch-geometric >= 2.3.0**
   - Thư viện chuyên dụng cho Graph Neural Networks
   - Bắt buộc phải có để triển khai các mô hình GCN và GAT
   - Cung cấp các lớp GCNConv và GATConv cho xử lý đồ thị

### Data Processing

3. **numpy >= 1.24.0**
   - Thư viện xử lý dữ liệu dạng mảng đa chiều
   - Được sử dụng để load và xử lý các file định dạng .npy

4. **pandas >= 2.0.0**
   - Thư viện xử lý và phân tích dữ liệu dạng bảng
   - Được tích hợp vào quy trình tiền xử lý dataset

5. **scikit-learn >= 1.3.0**
   - Bộ công cụ machine learning
   - Sử dụng hàm train_test_split trong quá trình phân chia dữ liệu training

### Training Utilities

6. **tqdm >= 4.65.0**
   - Thư viện hiển thị thanh tiến trình trong quá trình training
   - Được tích hợp vào các vòng lặp training epochs

### Visualization

7. **matplotlib >= 3.7.0**
   - Thư viện vẽ biểu đồ và trực quan hóa dữ liệu
   - Sử dụng để tạo các biểu đồ ROC curve, confusion matrix, và training history

8. **seaborn >= 0.12.0**
   - Thư viện trực quan hóa dữ liệu nâng cao
   - Được sử dụng kết hợp với matplotlib để tăng tính thẩm mỹ của biểu đồ

### API & Web Services (cho inference)

9. **fastapi >= 0.104.0**
   - Framework web để tạo API
   - Sử dụng cho API inference server

10. **uvicorn[standard] >= 0.24.0**
    - ASGI server để chạy FastAPI
    - Dùng để deploy API server

11. **pydantic >= 2.0.0**
    - Data validation cho FastAPI
    - Đảm bảo type safety cho API requests

12. **python-multipart >= 0.0.6**
    - Hỗ trợ multipart/form-data trong FastAPI

## Cài Đặt Môi Trường

### Phương pháp 1: Cài đặt thủ công

```bash
pip install -r requirements.txt
pip install torch-geometric
```

### Phương pháp 2: Sử dụng Docker (Khuyến nghị)

```bash
cd docker
docker-compose up --build
```

Docker sẽ tự động thực hiện các bước sau:
- Cài đặt Python 3.9
- Cài đặt các công cụ biên dịch gcc, g++ (bắt buộc cho torch-geometric)
- Cài đặt toàn bộ dependencies từ requirements.txt
- Cài đặt torch-geometric

## Yêu Cầu Phần Cứng

### GPU (Khuyến nghị)
- GPU hỗ trợ CUDA để tăng tốc độ quá trình training
- PyTorch sẽ tự động phát hiện và sử dụng GPU nếu có sẵn
- Kiểm tra tính khả dụng của CUDA bằng lệnh: `torch.cuda.is_available()`

### CPU
- Hệ thống có thể hoạt động trên CPU nhưng hiệu suất sẽ giảm đáng kể
- Sử dụng tham số `--device cpu` khi training nếu không có GPU

### RAM
- Yêu cầu tối thiểu: 8GB RAM
- Đối với dataset có kích thước lớn, khuyến nghị sử dụng 16GB RAM trở lên

### Dung Lượng Ổ Đĩa
- Lưu trữ models: khoảng 100MB - 500MB
- Lưu trữ dữ liệu đã xử lý: phụ thuộc vào kích thước dataset
- Lưu trữ kết quả và các visualization: khoảng 50MB - 200MB

## Kiểm Tra Môi Trường

### Kiểm tra phiên bản Python

```bash
python --version
```

### Kiểm tra PyTorch và CUDA

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

### Kiểm tra torch-geometric

```python
import torch_geometric
print(f"torch-geometric version: {torch_geometric.__version__}")
```

### Kiểm tra toàn bộ các thư viện phụ thuộc

```bash
pip list | grep -E "torch|numpy|pandas|scikit-learn|tqdm|matplotlib|seaborn|fastapi"
```

## Biến Môi Trường

Hệ thống sử dụng các biến môi trường sau (theo cấu hình trong Dockerfile):

- `PYTHONPATH=/app`: Đảm bảo Python có thể import các module từ thư mục gốc của project
- `PYTHONUNBUFFERED=1`: Vô hiệu hóa buffer output để xem log theo thời gian thực

## Khắc Phục Sự Cố

### Lỗi cài đặt torch-geometric

Trong trường hợp gặp lỗi khi cài đặt torch-geometric, cần cài đặt các công cụ biên dịch:

**Windows:**
- Cài đặt Visual Studio Build Tools
- Hoặc sử dụng conda: `conda install pytorch-geometric -c pyg`

**Linux:**
```bash
apt-get update && apt-get install -y gcc g++ make
```

### Lỗi CUDA không khả dụng

- Kiểm tra driver NVIDIA bằng lệnh: `nvidia-smi`
- Cài đặt PyTorch với hỗ trợ CUDA đúng phiên bản
- Hoặc sử dụng chế độ CPU bằng tham số: `--device cpu`

