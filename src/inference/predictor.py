import numpy as np
import torch
import torch.nn.functional as F
import pickle
import logging

from src.models import get_model_factory
from src.data.preprocessing import CodePreprocessor, create_graph_structure
from src.config.settings import DATA_DIR, MODELS_DIR, MODEL_CONFIG, PREPROCESSING_CONFIG

logger = logging.getLogger(__name__)


class VulnerabilityPredictor:
    
    def __init__(self, model_type='mlp', device='auto'):
        self.model_type = model_type.lower()
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.preprocessor = None
        self._load_preprocessor()
        self._load_model()
        
        logger.info(f"Predictor initialized with {model_type.upper()} model on {self.device}")
    
    def _load_preprocessor(self):
        preprocessor_path = DATA_DIR / "preprocessor.pkl"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Please run data preprocessing first.")
        
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        logger.info("Preprocessor loaded successfully")
    
    def _load_model(self):
        model_path = MODELS_DIR / f'best_{self.model_type}_model.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        if self.model_type not in MODEL_CONFIG:
            raise ValueError(f"Unknown model type: {self.model_type}. Available: {list(MODEL_CONFIG.keys())}")
        
        config = MODEL_CONFIG[self.model_type]
        factory_func = get_model_factory(self.model_type)
        self.model = factory_func(**config)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model {self.model_type.upper()} loaded successfully")
    
    def _preprocess_code(self, code):
        tokens = self.preprocessor.tokenize_code(code)
        encoded = self.preprocessor.encode_tokens(tokens)
        return np.array(encoded)
    
    def predict_from_code(self, code):
        logger.info(f"Predicting with {self.model_type.upper()} model")
        
        try:
            if self.model_type == 'mlp':
                X = self._preprocess_code(code)
                X = torch.FloatTensor(X).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(X)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                probs = probabilities[0].cpu().numpy()
            else:
                tokens = self.preprocessor.tokenize_code(code)
                encoded = self.preprocessor.encode_tokens(tokens)
                
                nodes, adj = create_graph_structure(
                    encoded,
                    max_nodes=PREPROCESSING_CONFIG['max_nodes'],
                    feature_dim=PREPROCESSING_CONFIG['feature_dim']
                )
                
                X = torch.FloatTensor(nodes).unsqueeze(0).to(self.device)
                adj = torch.FloatTensor(adj).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.forward_from_adj(X, adj)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                probs = probabilities[0].cpu().numpy()
            
            result = {
                'model_type': self.model_type,
                'prediction': 'Vulnerable' if predicted_class == 1 else 'Non-Vulnerable',
                'confidence': float(confidence),
                'probabilities': {
                    'Non-Vulnerable': float(probs[0]),
                    'Vulnerable': float(probs[1])
                }
            }
            
            logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            return {'error': str(e)}
    
    def predict_from_file(self, file_path):
        logger.info(f"Reading file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            result = self.predict_from_code(code)
            if 'error' not in result:
                result['file_path'] = file_path
            
            return result
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            return {'error': f"Error reading file: {e}"}

