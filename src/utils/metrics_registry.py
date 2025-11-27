from typing import Dict, Callable, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

_metric_registry: Dict[str, Callable] = {}


def register_metric(name: str, metric_func: Callable):
    _metric_registry[name] = metric_func
    logger.info(f"Registered metric: {name}")


def get_metric(name: str) -> Callable:
    if name not in _metric_registry:
        raise ValueError(f"Metric {name} not found in registry. Available: {list(_metric_registry.keys())}")
    return _metric_registry[name]


def list_metrics() -> list:
    return list(_metric_registry.keys())


def calculate_all_metrics(y_true, y_pred, y_pred_proba=None) -> Dict[str, Any]:
    results = {}
    for metric_name, metric_func in _metric_registry.items():
        try:
            if y_pred_proba is not None:
                results[metric_name] = metric_func(y_true, y_pred, y_pred_proba)
            else:
                results[metric_name] = metric_func(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Error calculating metric {metric_name}: {e}")
            results[metric_name] = None
    return results

