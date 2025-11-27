import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import logging

from src.models import get_model_factory
from src.utils.metrics import calculate_metrics, print_metrics
from src.utils.visualization import (
    plot_confusion_matrix, plot_roc_curve, plot_training_history
)
from src.config.settings import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, MODEL_CONFIG, TRAINING_CONFIG
)

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GraphDataset(Dataset):
    def __init__(self, X, adj, y):
        self.X = torch.FloatTensor(X)
        self.adj = torch.FloatTensor(adj)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.adj[idx], self.y[idx]


class ModelTrainer:
    
    def __init__(self, model_type, device='auto'):
        self.model_type = model_type
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def train_mlp(self, train_loader, val_loader, epochs=None, lr=None):
        epochs = epochs or TRAINING_CONFIG['epochs']
        lr = lr or TRAINING_CONFIG['learning_rate']
        
        config = MODEL_CONFIG['mlp']
        factory_func = get_model_factory('mlp')
        model = factory_func(**config)
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                X, y = X.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += y.size(0)
                train_correct += (predicted == y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y.size(0)
                    val_correct += (predicted == y).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODELS_DIR / 'best_mlp_model.pth')
                logger.info(f"Saved best MLP model at epoch {epoch+1}")
        
        return history, model
    
    def train_gcn_gat(self, train_loader, val_loader, epochs=None, lr=None):
        epochs = epochs or TRAINING_CONFIG['epochs']
        lr = lr or TRAINING_CONFIG['learning_rate']
        
        config = MODEL_CONFIG[self.model_type]
        factory_func = get_model_factory(self.model_type)
        model = factory_func(**config)
        
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for X, adj, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                X, adj, y = X.to(self.device), adj.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model.forward_from_adj(X, adj)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += y.size(0)
                train_correct += (predicted == y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X, adj, y in val_loader:
                    X, adj, y = X.to(self.device), adj.to(self.device), y.to(self.device)
                    outputs = model.forward_from_adj(X, adj)
                    loss = criterion(outputs, y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y.size(0)
                    val_correct += (predicted == y).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_name = self.model_type
                torch.save(model.state_dict(), MODELS_DIR / f'best_{model_name}_model.pth')
                logger.info(f"Saved best {model_name.upper()} model at epoch {epoch+1}")
        
        return history, model
    
    def evaluate_model(self, model, test_loader):
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            if self.model_type == 'mlp':
                for X, y in test_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = model(X)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
            else:
                for X, adj, y in test_loader:
                    X, adj, y = X.to(self.device), adj.to(self.device), y.to(self.device)
                    outputs = model.forward_from_adj(X, adj)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def save_results(self, y_true, y_pred, metrics, history, model_name):
        from src.utils.visualization import plot_confusion_matrix
        plot_confusion_matrix(
            y_true, y_pred, model_name,
            str(RESULTS_DIR / f'{model_name.lower()}_confusion_matrix.png')
        )
        
        with open(RESULTS_DIR / f'{model_name.lower()}_metrics.json', 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.integer, np.floating)) else str(v) 
                     for k, v in metrics.items() if k != 'confusion_matrix'}, f, indent=2)
        
        plot_training_history(history, model_name, str(RESULTS_DIR / f'{model_name.lower()}_training_history.png'))
        
        logger.info(f"Results saved for {model_name}")

