import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def _calculate_accuracy(y_true, y_pred, y_pred_proba=None):
    return accuracy_score(y_true, y_pred)

def _calculate_precision(y_true, y_pred, y_pred_proba=None):
    return precision_score(y_true, y_pred, average='binary', zero_division=0)

def _calculate_recall(y_true, y_pred, y_pred_proba=None):
    return recall_score(y_true, y_pred, average='binary', zero_division=0)

def _calculate_f1(y_true, y_pred, y_pred_proba=None):
    return f1_score(y_true, y_pred, average='binary', zero_division=0)

def _calculate_roc_auc(y_true, y_pred, y_pred_proba=None):
    if y_pred_proba is None:
        return None
    try:
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
            y_pred_proba = y_pred_proba[:, 1]
        return roc_auc_score(y_true, y_pred_proba)
    except Exception as e:
        print(f"Warning: Could not calculate ROC-AUC: {e}")
        return 0.0

from .metrics_registry import register_metric

register_metric('accuracy', _calculate_accuracy)
register_metric('precision', _calculate_precision)
register_metric('recall', _calculate_recall)
register_metric('f1_score', _calculate_f1)
register_metric('roc_auc', _calculate_roc_auc)

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    from .metrics_registry import calculate_all_metrics
    
    metrics = calculate_all_metrics(y_true, y_pred, y_pred_proba)
    
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    print(f"\n{'='*60}")
    print(f"Metrics for {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    
    if 'specificity' in metrics:
        print(f"Specificity: {metrics['specificity']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    if 'true_positives' in metrics:
        print(f"\nTrue Positives:  {metrics['true_positives']}")
        print(f"True Negatives:  {metrics['true_negatives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
    
    print(f"{'='*60}\n")


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=['Non-Vulnerable', 'Vulnerable'])
