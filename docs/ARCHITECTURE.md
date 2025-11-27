# Architecture Documentation

## Project Structure

```
DSP-C/
├── src/                          # Source code (main application)
│   ├── config/                   # Configuration files
│   │   ├── __init__.py
│   │   └── settings.py          # All configuration constants
│   ├── data/                     # Data preprocessing
│   │   ├── __init__.py
│   │   └── preprocessing.py      # Data cleaning and preprocessing
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   ├── mlp_model.py          # MLP model
│   │   ├── gcn_model.py          # GCN model
│   │   └── gat_model.py          # GAT model
│   ├── training/                 # Training logic
│   │   ├── __init__.py
│   │   └── trainer.py            # Model training classes
│   ├── inference/                # Inference logic
│   │   ├── __init__.py
│   │   └── predictor.py          # Prediction classes
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── logger.py             # Logging setup
│       ├── metrics.py            # Metrics calculation
│       └── visualization.py      # Plotting functions
├── api/                          # API server
│   ├── app.py                    # FastAPI application
│   └── README.md                 # API documentation
├── scripts/                      # Executable scripts
│   ├── preprocess_data.py        # Data preprocessing script
│   └── train.py                  # Training script
├── docker/                       # Docker configuration
│   ├── Dockerfile                # Docker image definition
│   └── docker-compose.yml       # Docker compose configuration
├── data/                         # Processed data (generated)
├── saved_models/                 # Trained models (generated)
├── results/                      # Training results (generated)
├── requirements.txt              # Python dependencies
├── .dockerignore                 # Docker ignore file
└── .gitignore                    # Git ignore file
```

## Design Patterns

### 1. Separation of Concerns
- **Data Layer**: `src/data/` - Handles data loading and preprocessing
- **Model Layer**: `src/models/` - Model definitions
- **Training Layer**: `src/training/` - Training logic
- **Inference Layer**: `src/inference/` - Prediction logic
- **API Layer**: `api/` - REST API interface

### 2. Configuration Management
- Centralized configuration in `src/config/settings.py`
- Environment-based configuration support
- Easy to modify without changing code

### 3. Factory Pattern
- Model creation functions: `create_mlp_model()`, `create_gcn_model()`, `create_gat_model()`
- Predictor factory in API layer

### 4. Registry Pattern
- Model registry: `src/models/registry.py`
- Metric registry: `src/utils/metrics_registry.py`
- Enables dynamic extension without code modification

### 5. Strategy Pattern
- Different models can be selected at runtime
- Model selection via API parameter

### 6. Singleton Pattern
- Predictor instances cached in API server
- Prevents reloading models on every request

## Workflow

### 1. Data Preprocessing
```bash
python scripts/preprocess_data.py
```
- Loads raw dataset
- Creates vocabulary
- Processes and splits data (80/20)
- Saves to `data/` directory

### 2. Training
```bash
python scripts/train.py --model all --epochs 50
```
- Loads processed data
- Trains selected models
- Saves best models to `saved_models/`
- Generates metrics and plots to `results/`

### 3. API Server
```bash
# Using Docker
cd docker
docker-compose up

# Or locally
cd api
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Making Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"code": "your code", "model_type": "mlp"}'
```

## Logging

All components use centralized logging:
- Console output only (no file logging)
- Trace information for debugging
- Structured log format
- Log level: INFO by default

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict from code string
- `POST /predict/file` - Predict from file path

## Model Selection

Models can be selected via `model_type` parameter:
- `"mlp"` - Multi-Layer Perceptron
- `"gcn"` - Graph Convolutional Network
- `"gat"` - Graph Attention Network

## Docker Deployment

The application is containerized for easy deployment:
- Single container setup
- Volume mounts for models and data
- Health checks included
- Auto-restart on failure

