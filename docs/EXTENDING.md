# Hướng dẫn Mở rộng Hệ thống

## Thêm Model Mới

### Bước 1: Tạo Model File

Tạo file mới trong `src/models/`:

```python
# src/models/new_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NewModelVulnerabilityDetector(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_classes=2, dropout=0.3):
        super(NewModelVulnerabilityDetector, self).__init__()
        # Define your model architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def create_new_model(input_dim=512, hidden_dim=128, num_classes=2, dropout=0.3):
    return NewModelVulnerabilityDetector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout
    )
```

### Bước 2: Đăng ký Model

Cập nhật `src/models/__init__.py`:

```python
from .new_model import NewModelVulnerabilityDetector, create_new_model
from .registry import register_model

register_model('new_model', create_new_model)
```

### Bước 3: Thêm Configuration

Cập nhật `src/config/settings.py`:

```python
MODEL_CONFIG = {
    'new_model': {
        'input_dim': 512,
        'hidden_dim': 128,
        'num_classes': 2,
        'dropout': 0.3
    }
}
```

### Bước 4: Cập nhật Trainer (nếu cần)

Nếu model cần logic training đặc biệt, cập nhật `src/training/trainer.py`.

## Thêm Metric Mới

### Bước 1: Tạo Metric Function

Thêm vào `src/utils/metrics.py`:

```python
def _calculate_custom_metric(y_true, y_pred, y_pred_proba=None):
    # Your calculation logic
    return metric_value
```

### Bước 2: Đăng ký Metric

```python
from .metrics_registry import register_metric

register_metric('custom_metric', _calculate_custom_metric)
```

### Bước 3: Metric sẽ tự động được tính

Metric sẽ tự động được tính trong `calculate_metrics()`.

## Thêm Visualization Mới

### Bước 1: Tạo Plot Function

Thêm vào `src/utils/visualization.py`:

```python
def plot_custom_visualization(data, model_name, save_path=None):
    plt.figure(figsize=(10, 6))
    # Your plotting logic
    if save_path:
        plt.savefig(save_path)
    plt.close()
```

### Bước 2: Sử dụng trong Training

Import và gọi trong `scripts/train.py`:

```python
from src.utils.visualization import plot_custom_visualization

plot_custom_visualization(data, "ModelName", str(RESULTS_DIR / 'custom_plot.png'))
```

## Best Practices

1. **Naming Convention**: Sử dụng tên rõ ràng, mô tả chức năng
2. **Error Handling**: Luôn có try-catch cho các operations có thể fail
3. **Logging**: Log các operations quan trọng
4. **Documentation**: Thêm docstring (nếu cần, nhưng không comment)
5. **Testing**: Test các components mới trước khi tích hợp

## Ví dụ Hoàn chỉnh: Thêm Model Transformer

### 1. Tạo Model

```python
# src/models/transformer_model.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerVulnerabilityDetector(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

def create_transformer_model(vocab_size=10000, d_model=512, nhead=8, num_layers=6, num_classes=2):
    return TransformerVulnerabilityDetector(vocab_size, d_model, nhead, num_layers, num_classes)
```

### 2. Đăng ký

```python
# src/models/__init__.py
from .transformer_model import create_transformer_model
register_model('transformer', create_transformer_model)
```

### 3. Config

```python
# src/config/settings.py
MODEL_CONFIG['transformer'] = {
    'vocab_size': 10000,
    'd_model': 512,
    'nhead': 8,
    'num_layers': 6,
    'num_classes': 2
}
```

### 4. Sử dụng

Model mới sẽ tự động available trong API và training script!

