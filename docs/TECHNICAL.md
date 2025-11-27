# Tài liệu Kỹ thuật

## Kiến trúc Hệ thống

### Cấu trúc Thư mục

```
DSP-C/
├── src/
│   ├── config/          # Configuration management
│   ├── data/            # Data preprocessing
│   ├── models/          # Model definitions
│   ├── training/        # Training logic
│   ├── inference/       # Inference logic
│   └── utils/           # Utilities (metrics, visualization, logging)
├── api/                 # FastAPI server
├── scripts/             # Executable scripts
├── docker/              # Docker configuration
└── docs/                # Documentation
```

### Design Patterns

#### 1. Factory Pattern
- Model creation: `create_mlp_model()`, `create_gcn_model()`, `create_gat_model()`
- Predictor creation trong API layer

#### 2. Strategy Pattern
- Model selection tại runtime
- Metric calculation strategies

#### 3. Registry Pattern
- Model registry để dễ mở rộng
- Metric registry để thêm metrics mới

#### 4. Singleton Pattern
- Predictor instances cached trong API
- Logger instances

#### 5. Template Method Pattern
- Training workflow template
- Evaluation workflow template

## Mở rộng Hệ thống

### Thêm Model Mới

1. Tạo file model trong `src/models/`:
```python
# src/models/new_model.py
import torch
import torch.nn as nn

class NewModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Model definition
    
    def forward(self, x):
        # Forward pass
        return output

def create_new_model(**kwargs):
    return NewModel(**kwargs)
```

2. Đăng ký trong `src/models/__init__.py`:
```python
from .new_model import NewModel, create_new_model
```

3. Thêm config trong `src/config/settings.py`:
```python
MODEL_CONFIG = {
    'new_model': {
        'param1': value1,
        'param2': value2,
    }
}
```

4. Cập nhật trainer nếu cần logic đặc biệt

### Thêm Metric Mới

1. Thêm function trong `src/utils/metrics.py`:
```python
def calculate_custom_metric(y_true, y_pred):
    # Calculation logic
    return metric_value
```

2. Đăng ký trong `calculate_metrics()`:
```python
metrics['custom_metric'] = calculate_custom_metric(y_true, y_pred)
```

3. Cập nhật visualization nếu cần plot

### Thêm Visualization Mới

1. Tạo function trong `src/utils/visualization.py`:
```python
def plot_custom_visualization(data, save_path=None):
    # Plotting logic
    plt.savefig(save_path)
```

2. Import và sử dụng trong training script

## Configuration

Tất cả configuration được quản lý tập trung trong `src/config/settings.py`:

- `MODEL_CONFIG`: Cấu hình cho từng model
- `TRAINING_CONFIG`: Cấu hình training
- `PREPROCESSING_CONFIG`: Cấu hình preprocessing
- `API_CONFIG`: Cấu hình API server
- `LOGGING_CONFIG`: Cấu hình logging

## Data Flow

### Training Flow
1. Load raw data từ `dataset/`
2. Preprocess và tạo vocabulary
3. Split data (80/20)
4. Train model với validation
5. Evaluate trên test set
6. Save model và metrics

### Lệnh Huấn luyện
```bash
python scripts/train.py --model all --epochs 50 --device cuda
```
- Dùng `--device cuda` để khai thác GPU (dùng `cpu` nếu không có GPU)
- `--model` có thể là `mlp`, `gcn`, `gat` hoặc `all`
- `--epochs`, `--batch_size`, `--lr` có thể tùy chỉnh

### Inference Flow
1. Load preprocessor
2. Load trained model
3. Preprocess input code
4. Predict
5. Return results

## Logging

Hệ thống sử dụng Python logging module:
- Console output only
- Structured format
- Trace information
- Level: INFO by default

## Performance

### Optimization
- Model caching trong API
- Batch processing
- GPU support
- Memory efficient data loading

### Monitoring
- Training metrics tracking
- API request logging
- Error tracking với traceback

## Testing

### Unit Tests
- Model creation tests
- Preprocessing tests
- Metric calculation tests

### Integration Tests
- API endpoint tests
- End-to-end prediction tests

## Deployment

### Docker
- Single container setup
- Volume mounts cho models và data
- Health checks
- Auto-restart

### Production Considerations
- Model versioning
- API rate limiting
- Error handling
- Monitoring và alerting

