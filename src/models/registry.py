from typing import Dict, Callable, Any
import logging

logger = logging.getLogger(__name__)

_model_registry: Dict[str, Callable] = {}


def register_model(name: str, factory_func: Callable):
    _model_registry[name] = factory_func
    logger.info(f"Registered model: {name}")


def get_model_factory(name: str) -> Callable:
    if name not in _model_registry:
        raise ValueError(f"Model {name} not found in registry. Available: {list(_model_registry.keys())}")
    return _model_registry[name]


def list_models() -> list:
    return list(_model_registry.keys())


from .mlp_model import create_mlp_model
from .gcn_model import create_gcn_model
from .gat_model import create_gat_model

register_model('mlp', create_mlp_model)
register_model('gcn', create_gcn_model)
register_model('gat', create_gat_model)

