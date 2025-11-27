# AI-based Source Code Vulnerability Detection and Classification

Hệ thống phát hiện và phân loại lỗ hổng bảo mật trong mã nguồn sử dụng Deep Learning.

## Tổng quan

Dự án triển khai ba mô hình deep learning để phát hiện lỗ hổng bảo mật trong mã nguồn:
- **MLP** (Multi-Layer Perceptron)
- **GCN** (Graph Convolutional Network)
- **GAT** (Graph Attention Network)

## Yêu cầu

- Python 3.9+
- Windows 10/11
- GPU (khuyến nghị)

## Cài đặt nhanh

```bash
pip install -r requirements.txt
```

Xem chi tiết trong [docs/TECHNICAL.md](docs/TECHNICAL.md)

## Sử dụng

### 1. Xử lý dữ liệu

```bash
python scripts/preprocess_data.py
```

### 2. Huấn luyện mô hình

```bash
python scripts/train.py --model all --epochs 50 --device cuda
```

Nếu không có GPU, bỏ `--device cuda` hoặc đặt `--device cpu`.

### 3. Chạy API Server

**Docker:**
```bash
cd docker
docker-compose up --build
```

**Local:**
```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Sử dụng API

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"code": "your code here", "model_type": "mlp"}'
```

## Tài liệu

### Tài liệu Kỹ thuật
- [Tài liệu kỹ thuật](docs/TECHNICAL.md) - Chi tiết kỹ thuật và kiến trúc
- [Kiến trúc hệ thống](docs/ARCHITECTURE.md) - Mô tả kiến trúc và design patterns
- [Hướng dẫn mở rộng](docs/EXTENDING.md) - Cách thêm model/metric mới

### Tài liệu API
- [Tài liệu API](docs/API.adoc) - Tài liệu API cho đối tác (AsciiDoc format)

## Cấu trúc dự án

```
DSP-C/
├── src/              # Source code
│   ├── config/       # Configuration
│   ├── data/         # Data preprocessing
│   ├── models/       # Model definitions
│   ├── training/     # Training logic
│   ├── inference/    # Inference logic
│   └── utils/        # Utilities
├── api/              # API server
├── scripts/          # Executable scripts
├── docker/           # Docker configuration
├── docs/             # Documentation
└── data/             # Processed data (generated)
```

## Metrics

Hệ thống hỗ trợ các metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Visualization plots

## Mở rộng

Kiến trúc được thiết kế để dễ mở rộng:
- Thêm model mới: Tạo file trong `src/models/` và đăng ký
- Thêm metric mới: Tạo function trong `src/utils/metrics.py`
- Xem chi tiết trong [docs/EXTENDING.md](docs/EXTENDING.md)

## License

MIT License
