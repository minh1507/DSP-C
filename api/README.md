# API Server Documentation

## Overview

RESTful API server for Source Code Vulnerability Detection using FastAPI.

## Endpoints

### Health Check
```
GET /health
```

### Root
```
GET /
```

### Predict from Code
```
POST /predict
Content-Type: application/json

{
    "code": "your source code here",
    "model_type": "mlp"  // Options: "mlp", "gcn", "gat"
}
```

### Predict from File Path
```
POST /predict/file?file_path=/path/to/file&model_type=mlp
```

## Usage Examples

### Using curl

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "code": "void vulnerable_function(char *input) { strcpy(buffer, input); }",
       "model_type": "mlp"
     }'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "code": "void vulnerable_function(char *input) { strcpy(buffer, input); }",
        "model_type": "mlp"
    }
)

print(response.json())
```

## Response Format

```json
{
    "model_type": "mlp",
    "prediction": "Vulnerable",
    "confidence": 0.9234,
    "probabilities": {
        "Non-Vulnerable": 0.0766,
        "Vulnerable": 0.9234
    }
}
```

## Running with Docker

```bash
cd docker
docker-compose up --build
```

## Running Locally

```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Logging

All requests and predictions are logged to console with trace information.

