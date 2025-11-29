# Tài Liệu Training - DSP-C

Tài liệu này trình bày một cách hệ thống về môi trường training, quy trình tiền xử lý dữ liệu (preprocessing), cấu hình huấn luyện, và phương thức thực thi quá trình training cho hệ thống phát hiện lỗ hổng bảo mật trong mã nguồn.

## Cấu Trúc Tài Liệu

### 1. [Môi Trường Training](01-moi-truong-training.md)
- Python version và các yêu cầu
- PyTorch và các thư viện deep learning
- Cài đặt môi trường
- Hardware requirements
- Troubleshooting

### 2. [Preprocess Dataset](02-preprocess-dataset.md)
- Cách thức preprocessing dataset
- Cấu hình preprocessing
- Input/Output files
- Cách thực hiện preprocessing
- Lưu ý quan trọng

### 3. [Training Configuration](03-training-config.md)
- Cấu hình training (epochs, batch size, learning rate)
- Cấu hình cho từng model (MLP, GCN, GAT)
- Optimizer và learning rate scheduler
- Metrics và evaluation
- Override configuration

### 4. [Cách Execute](04-cach-execute.md)
- Hướng dẫn chi tiết cách chạy training
- Command line arguments
- Các trường hợp sử dụng
- Workflow hoàn chỉnh
- Troubleshooting

## Hướng Dẫn Khởi Động Nhanh

### Bước 1: Thiết lập môi trường

```bash
pip install -r requirements.txt
pip install torch-geometric
```

### Bước 2: Tiền xử lý dữ liệu

```bash
python scripts/preprocess_data.py
```

### Bước 3: Huấn luyện mô hình

```bash
python scripts/train.py --model all --device auto
```

## Các Mô Hình

Hệ thống hỗ trợ ba kiến trúc mô hình:

1. **MLP (Multi-Layer Perceptron)**: Mô hình mạng neural truyền thống dựa trên fully connected layers
2. **GCN (Graph Convolutional Network)**: Mô hình mạng neural đồ thị sử dụng phép tích chập đồ thị
3. **GAT (Graph Attention Network)**: Mô hình mạng neural đồ thị tích hợp cơ chế attention

## Thông Tin Quan Trọng

- **Python Version**: 3.9+
- **PyTorch Version**: >= 2.0.0
- **Dataset Location**: `dataset/dataset_final_sorted/`
- **Processed Data Location**: `data/`
- **Models Location**: `saved_models/`
- **Results Location**: `results/`

## Tài Liệu Liên Quan

- [README chính](../README.md): Tổng quan về dự án
- [Tài liệu kỹ thuật](../docs/TECHNICAL.md): Chi tiết kỹ thuật và kiến trúc
- [Kiến trúc hệ thống](../docs/ARCHITECTURE.md): Mô tả kiến trúc
- [API Documentation](../docs/API.adoc): Tài liệu API

## Khắc Phục Sự Cố

Đối với các vấn đề có thể phát sinh, vui lòng tham khảo:
1. Phần Troubleshooting trong từng tài liệu chi tiết
2. Các log output trong quá trình thực thi script
3. Đảm bảo đã thực hiện đúng các bước trong phần Hướng Dẫn Khởi Động Nhanh

