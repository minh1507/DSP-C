from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import traceback

from src.inference.predictor import VulnerabilityPredictor
from src.utils.logger import setup_logger
from src.config.settings import API_CONFIG

logger = setup_logger(__name__)

app = FastAPI(
    title="Source Code Vulnerability Detection API",
    description="API for detecting vulnerabilities in source code using MLP, GCN, and GAT models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictors = {}


class PredictionRequest(BaseModel):
    code: Optional[str] = None
    model_type: str = "mlp"


class PredictionResponse(BaseModel):
    model_type: str
    prediction: str
    confidence: float
    probabilities: dict


@app.on_event("startup")
async def startup_event():
    from src.models import list_models
    available_models = list_models()
    logger.info("API server starting up...")
    logger.info(f"Available models: {', '.join(available_models)}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API server shutting down...")
    predictors.clear()


def get_predictor(model_type: str):
    if model_type not in predictors:
        logger.info(f"Loading {model_type.upper()} model...")
        try:
            predictors[model_type] = VulnerabilityPredictor(model_type=model_type)
            logger.info(f"{model_type.upper()} model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading {model_type.upper()} model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    return predictors[model_type]


@app.get("/")
async def root():
    from src.models import list_models
    return {
        "message": "Source Code Vulnerability Detection API",
        "version": "1.0.0",
        "available_models": list_models()
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_vulnerability(request: PredictionRequest):
    logger.info(f"Received prediction request with model: {request.model_type}")
    
    if not request.code:
        logger.warning("No code provided in request")
        raise HTTPException(status_code=400, detail="Code is required")
    
    from src.models import list_models
    available_models = list_models()
    if request.model_type not in available_models:
        logger.warning(f"Invalid model type: {request.model_type}")
        raise HTTPException(status_code=400, detail=f"Model type must be one of: {', '.join(available_models)}")
    
    try:
        predictor = get_predictor(request.model_type)
        result = predictor.predict_from_code(request.code)
        
        if 'error' in result:
            logger.error(f"Prediction error: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        logger.info(f"Prediction successful: {result['prediction']}")
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict/file")
async def predict_from_file_path(file_path: str, model_type: str = "mlp"):
    logger.info(f"Received file prediction request: {file_path} with model: {model_type}")
    
    from src.models import list_models
    available_models = list_models()
    if model_type not in available_models:
        logger.warning(f"Invalid model type: {model_type}")
        raise HTTPException(status_code=400, detail=f"Model type must be one of: {', '.join(available_models)}")
    
    try:
        predictor = get_predictor(model_type)
        result = predictor.predict_from_file(file_path)
        
        if 'error' in result:
            logger.error(f"Prediction error: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        logger.info(f"Prediction successful: {result['prediction']}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        log_level="info"
    )

