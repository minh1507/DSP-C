# Preprocess Dataset

## Tổng Quan

Tài liệu này trình bày phương pháp tiền xử lý (preprocessing) dataset cho hệ thống phát hiện lỗ hổng bảo mật. Quá trình preprocessing chuyển đổi dữ liệu mã nguồn thô thành định dạng phù hợp để huấn luyện các mô hình MLP, GCN, và GAT.

## Script Preprocessing

**File:** `scripts/preprocess_data.py`

Script này đóng vai trò là điểm khởi đầu (entry point) để thực thi quy trình preprocessing:

```python
from src.data.preprocessing import process_and_save_dataset
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

if __name__ == "__main__":
    process_and_save_dataset()
```

## Cấu Hình Preprocessing

Cấu hình được định nghĩa trong `src/config/settings.py`:

```python
PREPROCESSING_CONFIG = {
    'max_length': 512,
    'vocab_size': 10000,
    'max_nodes': 100,
    'feature_dim': 100
}
```

## Thư Mục Dữ Liệu

Các thư mục liên quan được định nghĩa trong file cấu hình `src/config/settings.py`:

- **DATASET_DIR**: `dataset/dataset_final_sorted/`
  - Thư mục chứa dataset gốc (raw data)
  - Dataset cần được đặt tại vị trí này trước khi thực thi quy trình preprocessing

- **DATA_DIR**: `data/`
  - Thư mục chứa dữ liệu đã được tiền xử lý
  - Tất cả các file output sẽ được lưu trữ tại đây

## Quá Trình Preprocessing

### 1. Module Preprocessing

Module chính: `src.data.preprocessing`

Module này chứa:
- `CodePreprocessor`: Class để preprocess mã nguồn
- `create_graph_structure`: Function để tạo cấu trúc graph từ mã nguồn
- `process_and_save_dataset`: Function chính để xử lý toàn bộ dataset

### 2. Các Bước Xử Lý

#### A. Tokenization

- **Phương thức**: `CodePreprocessor.tokenize_code(code)`
- Phân tách mã nguồn thành các tokens
- Xử lý các ký tự đặc biệt, khoảng trắng (whitespace), và comments

#### B. Encoding

- **Phương thức**: `CodePreprocessor.encode_tokens(tokens)`
- Chuyển đổi các tokens thành biểu diễn số
- Sử dụng vocabulary với kích thước `vocab_size = 10000`
- Thực hiện padding hoặc truncate để đạt độ dài `max_length = 512`

#### C. Tạo Cấu Trúc Đồ Thị

- **Cho mô hình GCN và GAT**: `create_graph_structure(encoded, max_nodes=100, feature_dim=100)`
- Chuyển đổi sequence tokens thành cấu trúc đồ thị (graph structure)
- Tạo ma trận kề (adjacency matrix) để biểu diễn quan hệ giữa các nodes
- Giới hạn số lượng nodes tối đa là `max_nodes = 100`
- Đặc trưng của nodes có chiều `feature_dim = 100`

### 3. Các File Đầu Ra

Sau khi hoàn tất quá trình preprocessing, các file sau sẽ được tạo trong thư mục `data/`:

#### Cho mô hình MLP:
- `X_train.npy`: Đặc trưng training (sequence embeddings)
- `X_test.npy`: Đặc trưng test (sequence embeddings)
- `y_train.npy`: Nhãn training (0: Non-Vulnerable, 1: Vulnerable)
- `y_test.npy`: Nhãn test

#### Cho mô hình GCN và GAT:
- `graph_X_train.npy`: Đặc trưng nodes cho training graphs
- `graph_X_test.npy`: Đặc trưng nodes cho test graphs
- `graph_adj_train.npy`: Ma trận kề cho training graphs
- `graph_adj_test.npy`: Ma trận kề cho test graphs
- `graph_y_train.npy`: Nhãn training cho graphs
- `graph_y_test.npy`: Nhãn test cho graphs

#### File quan trọng khác:
- `preprocessor.pkl`: Đối tượng preprocessor đã được serialize
  - Bắt buộc cần có cho quá trình inference sau này
  - Chứa vocabulary và các tham số đã được học (fit) trên dữ liệu training

## Phương Thức Thực Hiện Preprocessing

### Bước 1: Chuẩn Bị Dataset

Đảm bảo dataset được đặt tại vị trí sau:
```
dataset/dataset_final_sorted/
```

Dataset cần có định dạng phù hợp với cấu trúc mà module preprocessing yêu cầu.

### Bước 2: Thực Thi Preprocessing Script

```bash
python scripts/preprocess_data.py
```

### Bước 3: Kiểm Tra Kết Quả

Sau khi hoàn tất quá trình xử lý, kiểm tra các file đã được tạo trong thư mục `data/`:

```bash
dir data
ls -la data/
```

Các file cần có:
- `X_train.npy`, `X_test.npy`
- `y_train.npy`, `y_test.npy`
- `graph_X_train.npy`, `graph_X_test.npy`
- `graph_adj_train.npy`, `graph_adj_test.npy`
- `graph_y_train.npy`, `graph_y_test.npy`
- `preprocessor.pkl`

## Cấu Trúc Dữ Liệu

### Dữ Liệu Dạng Sequence (MLP)

- **Kích thước của X_train/X_test**: `(n_samples, max_length)` hoặc `(n_samples, feature_dim)`
- **Kích thước của y_train/y_test**: `(n_samples,)` - phân loại nhị phân (0 hoặc 1)

### Dữ Liệu Dạng Đồ Thị (GCN/GAT)

- **graph_X**: `(n_samples, max_nodes, feature_dim)`
  - Mỗi sample đại diện cho một đồ thị với tối đa `max_nodes` nodes
  - Mỗi node được đặc trưng bởi `feature_dim` chiều

- **graph_adj**: `(n_samples, max_nodes, max_nodes)`
  - Ma trận kề cho mỗi đồ thị
  - Giá trị lớn hơn 0 biểu thị sự tồn tại của cạnh (edge) giữa hai nodes

- **graph_y**: `(n_samples,)` - nhãn phân loại nhị phân

## Lưu Ý Quan Trọng

1. **Preprocessing phải được thực thi trước training**: Script training (`scripts/train.py`) sẽ kiểm tra sự tồn tại của file `X_train.npy`. Nếu không tìm thấy, hệ thống sẽ báo lỗi và dừng quá trình.

2. **Đối tượng Preprocessor**: File `preprocessor.pkl` có vai trò quan trọng đối với quá trình inference. Không nên xóa file này sau khi preprocessing.

3. **Tính nhất quán**: Preprocessing chỉ nên được thực thi một lần cho mỗi dataset. Trong trường hợp thay đổi dataset, cần thực hiện lại quy trình preprocessing.

4. **Bộ nhớ**: Dataset có kích thước lớn có thể yêu cầu dung lượng RAM lớn. Cần đảm bảo hệ thống có đủ bộ nhớ khi thực thi preprocessing.

5. **Phân chia Train/Test**: Tỷ lệ phân chia được định nghĩa trong `TRAINING_CONFIG['train_test_split'] = 0.2` (20% dành cho test set).

## Sử Dụng Preprocessor trong Quá Trình Inference

Trong quá trình inference (dự đoán), hệ thống sẽ tải `preprocessor.pkl`:

```python
with open(preprocessor_path, 'rb') as f:
    self.preprocessor = pickle.load(f)

tokens = self.preprocessor.tokenize_code(code)
encoded = self.preprocessor.encode_tokens(tokens)
```

Cần đảm bảo cùng một preprocessor được sử dụng cho cả quá trình training và inference để duy trì tính nhất quán về mặt biểu diễn dữ liệu.

## Khắc Phục Sự Cố

### Lỗi: Dataset không tìm thấy
- Kiểm tra đường dẫn `dataset/dataset_final_sorted/`
- Đảm bảo dataset được đặt đúng vị trí quy định

### Lỗi: Hết bộ nhớ (Out of memory)
- Giảm kích thước batch nếu có thể
- Hoặc xử lý dataset theo từng phần (nếu module hỗ trợ tính năng này)

### Lỗi: Module không tìm thấy
- Đảm bảo biến môi trường `PYTHONPATH` được thiết lập đúng
- Thực thi script từ thư mục gốc của project: `python scripts/preprocess_data.py`

