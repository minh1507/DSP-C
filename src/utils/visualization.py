import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
import os
import math


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Vulnerable', 'Vulnerable'],
                yticklabels=['Non-Vulnerable', 'Vulnerable'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, model_name, save_path=None):
    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
        y_pred_proba = y_pred_proba[:, 1]
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path=None):
    models = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    if any('roc_auc' in m for m in metrics_dict.values()):
        metric_names.append('roc_auc')
    
    data = {metric: [metrics_dict[model].get(metric, 0) for model in models] 
            for metric in metric_names}
    
    total_metrics = len(metric_names)
    cols = 2 if total_metrics > 1 else 1
    rows = math.ceil(total_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = np.array(axes).reshape(-1)
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = data[metric]
        bars = ax.bar(models, values, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom')
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for extra_ax in axes[total_metrics:]:
        extra_ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_training_history(history_dict, model_name, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history_dict['train_loss']) + 1)
    
    ax1.plot(epochs, history_dict['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training and Validation Loss - {model_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history_dict['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history_dict['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Training and Validation Accuracy - {model_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_all_comparisons(metrics_dict, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    
    plot_metrics_comparison(metrics_dict, 
                           save_path=os.path.join(save_dir, 'metrics_comparison.png'))
    
    print(f"Visualizations saved to {save_dir}/")
