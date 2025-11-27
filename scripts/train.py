import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.training.trainer import ModelTrainer, SequenceDataset, GraphDataset
from src.utils.metrics import calculate_metrics, print_metrics
from src.utils.visualization import plot_roc_curve, plot_metrics_comparison
from src.utils.logger import setup_logger
from src.config.settings import DATA_DIR, TRAINING_CONFIG

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train vulnerability detection models')
    parser.add_argument('--model', type=str, choices=['mlp', 'gcn', 'gat', 'all'],
                       default='all', help='Model to train')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    import torch
    from src.config.settings import MODELS_DIR, RESULTS_DIR
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.error("CUDA requested but not available. Please install CUDA-enabled PyTorch or use --device cpu.")
        return
    
    resolved_device = args.device
    if args.device == 'auto':
        resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    use_pin_memory = resolved_device.startswith('cuda')
    
    logger.info(f"Using device: {resolved_device.upper()}")
    
    if not (DATA_DIR / "X_train.npy").exists():
        logger.error("Processed data not found. Please run data preprocessing first.")
        return
    
    logger.info("Loading data...")
    
    all_metrics = {}
    
    if args.model in ['mlp', 'all']:
        logger.info("Loading MLP data...")
        X_train = np.load(DATA_DIR / 'X_train.npy')
        X_test = np.load(DATA_DIR / 'X_test.npy')
        y_train = np.load(DATA_DIR / 'y_train.npy')
        y_test = np.load(DATA_DIR / 'y_test.npy')
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        train_dataset = SequenceDataset(X_train, y_train)
        val_dataset = SequenceDataset(X_val, y_val)
        test_dataset = SequenceDataset(X_test, y_test)
        
        batch_size = args.batch_size or TRAINING_CONFIG['batch_size']
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=use_pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=use_pin_memory
        )
        
        logger.info("Training MLP Model")
        trainer = ModelTrainer('mlp', resolved_device)
        history, model = trainer.train_mlp(
            train_loader, val_loader,
            epochs=args.epochs, lr=args.lr
        )
        
        import torch
        from src.config.settings import MODELS_DIR, RESULTS_DIR
        model.load_state_dict(torch.load(MODELS_DIR / 'best_mlp_model.pth'))
        y_true, y_pred, y_proba = trainer.evaluate_model(model, test_loader)
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        print_metrics(metrics, "MLP")
        all_metrics['MLP'] = metrics
        
        plot_roc_curve(y_true, y_proba, "MLP", str(RESULTS_DIR / 'mlp_roc_curve.png'))
        trainer.save_results(y_true, y_pred, metrics, history, "MLP")
    
    if args.model in ['gcn', 'gat', 'all']:
        logger.info("Loading graph data...")
        graph_X_train = np.load(DATA_DIR / 'graph_X_train.npy')
        graph_X_test = np.load(DATA_DIR / 'graph_X_test.npy')
        graph_adj_train = np.load(DATA_DIR / 'graph_adj_train.npy')
        graph_adj_test = np.load(DATA_DIR / 'graph_adj_test.npy')
        graph_y_train = np.load(DATA_DIR / 'graph_y_train.npy')
        graph_y_test = np.load(DATA_DIR / 'graph_y_test.npy')
        
        graph_X_train, graph_X_val, graph_adj_train, graph_adj_val, graph_y_train, graph_y_val = train_test_split(
            graph_X_train, graph_adj_train, graph_y_train, test_size=0.2, random_state=42, stratify=graph_y_train
        )
        
        batch_size = args.batch_size or TRAINING_CONFIG['batch_size']
        
        if args.model in ['gcn', 'all']:
            graph_train_dataset = GraphDataset(graph_X_train, graph_adj_train, graph_y_train)
            graph_val_dataset = GraphDataset(graph_X_val, graph_adj_val, graph_y_val)
            graph_test_dataset = GraphDataset(graph_X_test, graph_adj_test, graph_y_test)
            
            graph_train_loader = DataLoader(
                graph_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=use_pin_memory
            )
            graph_val_loader = DataLoader(
                graph_val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=use_pin_memory
            )
            graph_test_loader = DataLoader(
                graph_test_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=use_pin_memory
            )
            
            logger.info("Training GCN Model")
            trainer = ModelTrainer('gcn', resolved_device)
            history, model = trainer.train_gcn_gat(
                graph_train_loader, graph_val_loader,
                epochs=args.epochs, lr=args.lr
            )
            
            model.load_state_dict(torch.load(MODELS_DIR / 'best_gcn_model.pth'))
            y_true, y_pred, y_proba = trainer.evaluate_model(model, graph_test_loader)
            
            metrics = calculate_metrics(y_true, y_pred, y_proba)
            print_metrics(metrics, "GCN")
            all_metrics['GCN'] = metrics
            
            plot_roc_curve(y_true, y_proba, "GCN", str(RESULTS_DIR / 'gcn_roc_curve.png'))
            trainer.save_results(y_true, y_pred, metrics, history, "GCN")
        
        if args.model in ['gat', 'all']:
            graph_train_dataset = GraphDataset(graph_X_train, graph_adj_train, graph_y_train)
            graph_val_dataset = GraphDataset(graph_X_val, graph_adj_val, graph_y_val)
            graph_test_dataset = GraphDataset(graph_X_test, graph_adj_test, graph_y_test)
            
            graph_train_loader = DataLoader(
                graph_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=use_pin_memory
            )
            graph_val_loader = DataLoader(
                graph_val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=use_pin_memory
            )
            graph_test_loader = DataLoader(
                graph_test_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=use_pin_memory
            )
            
            logger.info("Training GAT Model")
            trainer = ModelTrainer('gat', resolved_device)
            history, model = trainer.train_gcn_gat(
                graph_train_loader, graph_val_loader,
                epochs=args.epochs, lr=args.lr
            )
            
            model.load_state_dict(torch.load(MODELS_DIR / 'best_gat_model.pth'))
            y_true, y_pred, y_proba = trainer.evaluate_model(model, graph_test_loader)
            
            metrics = calculate_metrics(y_true, y_pred, y_proba)
            print_metrics(metrics, "GAT")
            all_metrics['GAT'] = metrics
            
            plot_roc_curve(y_true, y_proba, "GAT", str(RESULTS_DIR / 'gat_roc_curve.png'))
            trainer.save_results(y_true, y_pred, metrics, history, "GAT")
    
    if len(all_metrics) > 1:
        from src.utils.visualization import plot_metrics_comparison
        from src.config.settings import RESULTS_DIR
        plot_metrics_comparison(all_metrics, str(RESULTS_DIR / 'metrics_comparison.png'))
        logger.info("Model comparison plot saved")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

