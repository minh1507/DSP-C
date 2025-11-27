import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"
DATASET_DIR = BASE_DIR / "dataset" / "dataset_final_sorted"

MODEL_CONFIG = {
    'mlp': {
        'input_dim': 512,
        'hidden_dims': [256, 128, 64],
        'num_classes': 2,
        'dropout': 0.3
    },
    'gcn': {
        'num_features': 100,
        'hidden_dim': 64,
        'num_layers': 3,
        'num_classes': 2,
        'dropout': 0.3
    },
    'gat': {
        'num_features': 100,
        'hidden_dim': 64,
        'num_heads': 8,
        'num_layers': 3,
        'num_classes': 2,
        'dropout': 0.3
    }
}

TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'train_test_split': 0.2,
    'random_state': 42
}

PREPROCESSING_CONFIG = {
    'max_length': 512,
    'vocab_size': 10000,
    'max_nodes': 100,
    'feature_dim': 100
}

API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False
}

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

