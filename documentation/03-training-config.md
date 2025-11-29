# Training Configuration

## Tổng Quan

Tài liệu này trình bày chi tiết về cấu hình huấn luyện cho các mô hình MLP, GCN, và GAT trong hệ thống phát hiện lỗ hổng bảo mật.

## File Cấu Hình

Toàn bộ cấu hình được định nghĩa trong file: `src/config/settings.py`

## Cấu Hình Training

Cấu hình chung cho quá trình training được định nghĩa trong `TRAINING_CONFIG`:

```python
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'train_test_split': 0.2,
    'random_state': 42
}
```

### Chi Tiết Các Tham Số

#### 1. Epochs: 50
- Số lần duyệt qua toàn bộ dataset trong quá trình training
- Có thể ghi đè (override) bằng tham số `--epochs` khi thực thi training
- Early stopping được thực hiện thông qua việc lưu mô hình tốt nhất dựa trên validation loss

#### 2. Batch Size: 32
- Số lượng samples được xử lý trong mỗi batch
- Có thể ghi đè bằng tham số `--batch_size`
- Batch size lớn hơn có thể tăng tốc độ xử lý nhưng yêu cầu nhiều bộ nhớ hơn

#### 3. Learning Rate: 0.001
- Tốc độ học của mô hình trong quá trình tối ưu hóa
- Có thể ghi đè bằng tham số `--lr`
- Hệ thống sử dụng Adam optimizer với learning rate này

#### 4. Train/Test Split: 0.2
- 80% dữ liệu dành cho training, 20% dành cho test
- Phân chia được thực hiện với stratification (duy trì tỷ lệ các classes)
- Random seed = 42 để đảm bảo tính reproducibility

#### 5. Random State: 42
- Seed cho bộ tạo số ngẫu nhiên (random number generator)
- Đảm bảo kết quả có thể tái tạo được (reproducible)

## Cấu Hình Mô Hình

Cấu hình cho từng kiến trúc mô hình được định nghĩa trong `MODEL_CONFIG`:

### MLP Configuration

```python
'mlp': {
    'input_dim': 512,
    'hidden_dims': [256, 128, 64],
    'num_classes': 2,
    'dropout': 0.3
}
```

**Kiến trúc MLP:**
- Lớp đầu vào (Input layer): 512 neurons
- Lớp ẩn 1 (Hidden layer 1): 256 neurons + BatchNorm + ReLU + Dropout(0.3)
- Lớp ẩn 2 (Hidden layer 2): 128 neurons + BatchNorm + ReLU + Dropout(0.3)
- Lớp ẩn 3 (Hidden layer 3): 64 neurons + BatchNorm + ReLU + Dropout(0.3)
- Lớp đầu ra (Output layer): 2 neurons (phân loại nhị phân)

### GCN Configuration

```python
'gcn': {
    'num_features': 100,
    'hidden_dim': 64,
    'num_layers': 3,
    'num_classes': 2,
    'dropout': 0.3
}
```

**Kiến trúc GCN:**
- Chiếu đầu vào (Input projection): Linear(100 → 64)
- 3 lớp GCN, mỗi lớp bao gồm:
  - GCNConv với hidden_dim = 64
  - BatchNorm
  - Hàm kích hoạt ReLU
  - Dropout(0.3)
- Gộp đồ thị (Graph pooling): Global mean pooling
- Lớp fully connected 1: Linear(64 → 32) + ReLU + Dropout
- Lớp đầu ra: Linear(32 → 2)

### GAT Configuration

```python
'gat': {
    'num_features': 100,
    'hidden_dim': 64,
    'num_heads': 8,
    'num_layers': 3,
    'num_classes': 2,
    'dropout': 0.3
}
```

**Kiến trúc GAT:**
- Chiếu đầu vào (Input projection): Linear(100 → 64)
- 3 lớp GAT, mỗi lớp bao gồm:
  - GATConv với hidden_dim = 64, num_heads = 8
  - Multi-head attention với concatenation
  - BatchNorm
  - Hàm kích hoạt ReLU
  - Dropout(0.3)
- Gộp đồ thị (Graph pooling): Global mean pooling
- Lớp fully connected 1: Linear(64 → 32) + ReLU + Dropout
- Lớp đầu ra: Linear(32 → 2)

## Cấu Hình Quá Trình Training

### Optimizer

**Adam Optimizer** được sử dụng cho toàn bộ các mô hình:
- Learning rate: được lấy từ `TRAINING_CONFIG['learning_rate']` hoặc tham số `--lr`
- Learning rate mặc định: 0.001

### Learning Rate Scheduler

**ReduceLROnPlateau**:
- Mode: `'min'` (giảm learning rate khi validation loss không giảm)
- Factor: 0.5 (giảm learning rate đi một nửa)
- Patience: 5 epochs (chờ 5 epochs không có cải thiện trước khi giảm learning rate)

### Hàm Loss

**CrossEntropyLoss**:
- Phù hợp cho bài toán phân loại đa lớp (multi-class classification)
- Đối với phân loại nhị phân (2 classes), tương đương với softmax + negative log likelihood

### Chiến Lược Validation

- **Tỷ lệ phân chia validation**: 20% từ dữ liệu training
- **Stratification**: Duy trì tỷ lệ các classes không đổi
- **Random seed**: 42
- **Phân chia cuối cùng**: 80% train, 20% validation từ tập training gốc → 64% train, 16% validation, 20% test

### Lưu Trữ Mô Hình

- **Lưu mô hình tốt nhất**: Mô hình được lưu khi validation loss giảm
- **Vị trí lưu trữ**: `saved_models/best_{model_type}_model.pth`
- **Định dạng**: PyTorch state_dict
- Chỉ lưu trữ weights của mô hình, không lưu toàn bộ đối tượng mô hình

## DataLoader Configuration

### Pin Memory

- **CUDA**: `pin_memory=True` khi sử dụng GPU để tăng tốc độ truyền dữ liệu
- **CPU**: `pin_memory=False` khi sử dụng CPU

### Xáo Trộn Dữ Liệu

- **Training loader**: `shuffle=True` - xáo trộn dữ liệu ở mỗi epoch
- **Validation loader**: `shuffle=False` - không xáo trộn dữ liệu
- **Test loader**: `shuffle=False` - không xáo trộn dữ liệu

## Metrics và Đánh Giá

### Các metrics được tính toán:

1. **Accuracy**: Tỷ lệ dự đoán đúng
2. **Precision**: Độ chính xác cho class vulnerable
3. **Recall**: Độ nhạy (recall) cho class vulnerable
4. **F1-Score**: F1-score cho class vulnerable
5. **ROC-AUC**: Diện tích dưới đường cong ROC

### Các biểu đồ trực quan hóa được tạo:

1. **ROC Curve**: `results/{model}_roc_curve.png`
2. **Confusion Matrix**: `results/{model}_confusion_matrix.png`
3. **Training History**: `results/{model}_training_history.png`
4. **Metrics Comparison**: `results/metrics_comparison.png` (khi huấn luyện nhiều mô hình)

### Results Storage

- **Metrics JSON**: `results/{model}_metrics.json`
- **Model weights**: `saved_models/best_{model}_model.pth`
- **Visualizations**: Các file PNG trong `results/`

## Ghi Đè Cấu Hình

Tất cả các tham số training có thể được ghi đè (override) thông qua các đối số dòng lệnh khi thực thi `scripts/train.py`:

### Arguments

```bash
--model {mlp,gcn,gat,all}
--epochs INT
--batch_size INT
--lr FLOAT
--device {cuda,cpu,auto}
```

### Ví dụ Override

```bash
python scripts/train.py --model mlp --epochs 100
python scripts/train.py --model gcn --batch_size 64
python scripts/train.py --model gat --lr 0.0001
python scripts/train.py --model all --device cuda
```

## Directory Structure

Các thư mục được sử dụng (tự động tạo nếu chưa có):

- `data/`: Dữ liệu đã preprocess
- `saved_models/`: Model weights đã train
- `results/`: Metrics, visualizations, và training history

## Logging Configuration

Cấu hình logging trong `LOGGING_CONFIG`:

```python
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}
```

Logs sẽ hiển thị:
- Epoch number
- Train loss và accuracy
- Validation loss và accuracy
- Model saving events
- Errors và warnings

## Thực Hành Tốt Nhất

1. **Bắt đầu với cấu hình mặc định**: Nên bắt đầu với cấu hình mặc định trước khi điều chỉnh hyperparameters
2. **Giám sát validation loss**: Mô hình được lưu dựa trên validation loss, không phải training loss
3. **Sử dụng GPU nếu có sẵn**: GPU sẽ tăng tốc độ training đáng kể
4. **Lưu trữ kết quả**: Tất cả metrics và visualizations được tự động lưu trữ
5. **Tính reproducibility**: Random seed = 42 đảm bảo kết quả có thể tái tạo được

## Khắc Phục Sự Cố

### Hết bộ nhớ (Out of Memory)
- Giảm giá trị `batch_size`
- Giảm giá trị `max_nodes` trong preprocessing cho các mô hình đồ thị

### Training quá chậm
- Kiểm tra xem hệ thống có đang sử dụng GPU không
- Tăng `batch_size` nếu có thể
- Giảm số lượng `epochs` để test nhanh

### Mô hình không hội tụ
- Thử giảm giá trị `learning_rate`
- Kiểm tra dữ liệu có được preprocess đúng cách không
- Kiểm tra nhãn có bị mất cân bằng (imbalanced) không

