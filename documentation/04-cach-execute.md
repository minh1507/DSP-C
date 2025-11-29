# Cách Execute Training

## Tổng Quan

Tài liệu này trình bày chi tiết phương thức thực thi quá trình training cho hệ thống phát hiện lỗ hổng bảo mật. Hệ thống hỗ trợ huấn luyện ba kiến trúc mô hình: MLP, GCN, và GAT.

## Điều Kiện Tiên Quyết

Trước khi bắt đầu quá trình training, cần đảm bảo:

1. ✅ **Môi trường đã được thiết lập**: Python 3.9+, PyTorch, và toàn bộ các thư viện phụ thuộc
2. ✅ **Dataset đã được tiền xử lý**: Đã thực thi thành công `scripts/preprocess_data.py`
3. ✅ **Các file đã được preprocess tồn tại**: Kiểm tra thư mục `data/` có đầy đủ các file định dạng .npy

Xem chi tiết trong:
- [Môi trường Training](01-moi-truong-training.md)
- [Preprocess Dataset](02-preprocess-dataset.md)

## Hướng Dẫn Khởi Động Nhanh

### Bước 1: Tiền Xử Lý Dataset

```bash
python scripts/preprocess_data.py
```

Chờ quá trình preprocessing hoàn tất. Kết quả đầu ra sẽ được lưu trong thư mục `data/`.

### Bước 2: Huấn Luyện Mô Hình

```bash
python scripts/train.py --model all
```

Hoặc huấn luyện từng mô hình riêng biệt:

```bash
python scripts/train.py --model mlp
python scripts/train.py --model gcn
python scripts/train.py --model gat
```

## Script Training

**File:** `scripts/train.py`

Script này đóng vai trò là điểm khởi đầu chính cho quá trình training. Nó thực hiện các chức năng sau:
- Tải dữ liệu đã được tiền xử lý
- Tạo các data loaders
- Huấn luyện các mô hình được chỉ định
- Đánh giá (evaluate) trên test set
- Lưu trữ các mô hình và kết quả

## Command Line Arguments

### Cú pháp cơ bản

```bash
python scripts/train.py [OPTIONS]
```

### Các Options

#### `--model`

Lựa chọn mô hình (hoặc các mô hình) để huấn luyện:

- `mlp`: Chỉ huấn luyện mô hình MLP
- `gcn`: Chỉ huấn luyện mô hình GCN
- `gat`: Chỉ huấn luyện mô hình GAT
- `all`: Huấn luyện cả ba mô hình (mặc định)

**Ví dụ:**
```bash
python scripts/train.py --model mlp
python scripts/train.py --model all
```

#### `--epochs`

Số lượng epochs để huấn luyện (ghi đè giá trị mặc định 50):

```bash
python scripts/train.py --model mlp --epochs 100
python scripts/train.py --model all --epochs 30
```

**Mặc định:** 50 epochs (theo `TRAINING_CONFIG`)

#### `--batch_size`

Kích thước batch cho quá trình training (ghi đè giá trị mặc định 32):

```bash
python scripts/train.py --model gcn --batch_size 64
python scripts/train.py --model all --batch_size 16
```

**Mặc định:** 32 (theo `TRAINING_CONFIG`)

**Lưu ý:** Batch size lớn hơn yêu cầu nhiều bộ nhớ hơn, nhưng có thể tăng tốc độ training.

#### `--lr`

Learning rate (ghi đè giá trị mặc định 0.001):

```bash
python scripts/train.py --model gat --lr 0.0001
python scripts/train.py --model mlp --lr 0.01
```

**Mặc định:** 0.001 (theo `TRAINING_CONFIG`)

#### `--device`

Lựa chọn thiết bị để huấn luyện:

- `cuda`: Sử dụng GPU (nếu có sẵn)
- `cpu`: Sử dụng CPU
- `auto`: Tự động lựa chọn (cuda nếu có, ngược lại cpu) - **mặc định**

```bash
python scripts/train.py --model all --device cuda
python scripts/train.py --model all --device cpu
python scripts/train.py --model all --device auto
```

**Lưu ý:** Nếu lựa chọn `cuda` nhưng không có GPU, script sẽ báo lỗi và dừng quá trình.

## Các Trường Hợp Sử Dụng

### 1. Huấn Luyện Nhanh để Kiểm Thử

```bash
python scripts/train.py --model mlp --epochs 5 --device cpu
```

### 2. Huấn Luyện Production với GPU

```bash
python scripts/train.py --model all --epochs 100 --device cuda
```

### 3. Fine-tuning với Learning Rate Thấp

```bash
python scripts/train.py --model gcn --lr 0.0001 --epochs 80
```

### 4. Huấn Luyện với Batch Size Lớn (nếu có nhiều RAM/GPU memory)

```bash
python scripts/train.py --model all --batch_size 64 --device cuda
```

### 5. Huấn Luyện Từng Mô Hình Riêng Lẻ

```bash
python scripts/train.py --model mlp --epochs 50
python scripts/train.py --model gcn --epochs 50
python scripts/train.py --model gat --epochs 50
```

## Quá Trình Training

### 1. Kiểm Tra Dữ Liệu

Script sẽ kiểm tra sự tồn tại của các file đã được preprocess:

```python
if not (DATA_DIR / "X_train.npy").exists():
    logger.error("Processed data not found. Please run data preprocessing first.")
    return
```

Nếu không tìm thấy, quá trình training sẽ dừng lại với thông báo lỗi.

### 2. Tải Dữ Liệu

#### Cho mô hình MLP:
- Tải `X_train.npy`, `X_test.npy`
- Tải `y_train.npy`, `y_test.npy`
- Phân chia dữ liệu training thành train/validation (80/20)

#### Cho mô hình GCN và GAT:
- Tải `graph_X_train.npy`, `graph_X_test.npy`
- Tải `graph_adj_train.npy`, `graph_adj_test.npy`
- Tải `graph_y_train.npy`, `graph_y_test.npy`
- Phân chia dữ liệu training thành train/validation (80/20)

### 3. Tạo Data Loaders

- **Training loader**: `shuffle=True`, `pin_memory=True` (nếu sử dụng GPU)
- **Validation loader**: `shuffle=False`
- **Test loader**: `shuffle=False`

### 4. Vòng Lặp Training

Đối với mỗi epoch:
- Huấn luyện trên training set
- Đánh giá trên validation set
- Ghi log các metrics (loss, accuracy)
- Lưu mô hình nếu validation loss cải thiện
- Giảm learning rate nếu validation loss không cải thiện (sau 5 epochs)

### 5. Đánh Giá

Sau khi hoàn tất training:
- Tải mô hình tốt nhất từ `saved_models/`
- Đánh giá trên test set
- Tính toán các metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- In các metrics ra console
- Lưu metrics vào file JSON

### 6. Trực Quan Hóa

Tự động tạo các biểu đồ trực quan hóa:
- ROC curve
- Confusion matrix
- Training history (loss và accuracy qua các epochs)
- So sánh metrics (nếu huấn luyện nhiều mô hình)

### 7. Lưu Trữ Kết Quả

Tất cả kết quả được lưu trong thư mục `results/`:
- `{model}_metrics.json`: Metrics dưới dạng JSON
- `{model}_roc_curve.png`: ROC curve
- `{model}_confusion_matrix.png`: Confusion matrix
- `{model}_training_history.png`: Training history
- `metrics_comparison.png`: So sánh metrics giữa các mô hình (nếu có)

Các mô hình được lưu trong `saved_models/`:
- `best_mlp_model.pth`
- `best_gcn_model.pth`
- `best_gat_model.pth`

## Output và Logs

### Console Output

Trong quá trình training, bạn sẽ thấy:

```
INFO - Using device: CUDA
INFO - Loading data...
INFO - Loading MLP data...
INFO - Training MLP Model
Epoch 1/50 [Train]: 100%|████████████| 125/125 [00:15<00:00,  8.12it/s]
INFO - Epoch 1: Train Loss: 0.6234, Train Acc: 0.6500, Val Loss: 0.5891, Val Acc: 0.6900
INFO - Saved best MLP model at epoch 1
...
```

### Log Format

Logs theo format:
```
YYYY-MM-DD HH:MM:SS - module_name - LEVEL - message
```

## Quy Trình Hoàn Chỉnh

### Bước 1: Thiết Lập

```bash
pip install -r requirements.txt
pip install torch-geometric
```

### Bước 2: Tiền Xử Lý

```bash
python scripts/preprocess_data.py
```

### Bước 3: Huấn Luyện

```bash
python scripts/train.py --model all --device auto
python scripts/train.py --model all --epochs 100 --batch_size 64 --lr 0.001 --device cuda
```

### Bước 4: Kiểm Tra Kết Quả

```bash
cat results/mlp_metrics.json
```

## Khắc Phục Sự Cố

### Lỗi: "Processed data not found"

**Nguyên nhân:** Chưa thực thi preprocessing hoặc các file .npy không tồn tại.

**Giải pháp:**
```bash
python scripts/preprocess_data.py
```

### Lỗi: "CUDA requested but not available"

**Nguyên nhân:** Yêu cầu GPU nhưng không có sẵn hoặc CUDA chưa được cài đặt.

**Giải pháp:**
```bash
python scripts/train.py --model all --device cpu
python scripts/train.py --model all --device auto
```

### Lỗi: Hết Bộ Nhớ (Out of Memory - OOM)

**Nguyên nhân:** Batch size quá lớn hoặc mô hình quá lớn so với GPU/RAM hiện tại.

**Giải pháp:**
```bash
python scripts/train.py --model all --batch_size 16
python scripts/train.py --model mlp --batch_size 32
```

### Training quá chậm

**Nguyên nhân:** Đang sử dụng CPU hoặc batch size nhỏ.

**Giải pháp:**
- Đảm bảo đang sử dụng GPU: `--device cuda`
- Tăng batch size nếu có thể: `--batch_size 64`
- Kiểm tra mức độ sử dụng GPU: `nvidia-smi`

### Mô hình không được lưu

**Kiểm tra:**
- Thư mục `saved_models/` có tồn tại không
- Quyền ghi file trong thư mục
- Dung lượng ổ đĩa có đủ không

### Không có biểu đồ trực quan hóa

**Kiểm tra:**
- Thư mục `results/` có được tạo không
- Các thư viện matplotlib, seaborn đã được cài đặt chưa
- Xem logs để tìm lỗi cụ thể

## Thực Hành Tốt Nhất

1. **Luôn kiểm tra dữ liệu trước**: Thực thi preprocessing và kiểm tra các file đầu ra
2. **Bắt đầu với ít epochs**: Kiểm thử với 5-10 epochs trước khi huấn luyện đầy đủ
3. **Giám sát logs**: Theo dõi training loss và validation loss
4. **Sử dụng GPU nếu có sẵn**: GPU sẽ tăng tốc độ đáng kể
5. **Lưu trữ thường xuyên**: Các mô hình được tự động lưu khi validation loss cải thiện
6. **Sao lưu kết quả**: Sao lưu thư mục `saved_models/` và `results/` sau khi training

## Ước Tính Thời Gian

**Thời gian training ước tính (với GPU):**
- MLP (50 epochs): khoảng 10-30 phút (tùy thuộc vào kích thước dataset)
- GCN (50 epochs): khoảng 30-60 phút
- GAT (50 epochs): khoảng 45-90 phút (chậm hơn GCN do cơ chế attention)

**Với CPU:** Thời gian có thể tăng gấp 5-10 lần.

## Các Bước Tiếp Theo

Sau khi hoàn tất quá trình training:
1. Xem các metrics trong `results/{model}_metrics.json`
2. Xem các biểu đồ trực quan hóa để đánh giá hiệu suất
3. Sử dụng các mô hình đã được huấn luyện cho inference (API hoặc script)
4. Điều chỉnh hyperparameters nếu cần thiết

Xem thêm:
- [Training Configuration](03-training-config.md) để hiểu chi tiết về config
- Tài liệu về API inference trong `api/README.md`

