import logging
import sys
from src.config.settings import LOGGING_CONFIG

def setup_logger(name=None):
    logger = logging.getLogger(name or __name__)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, LOGGING_CONFIG['level']))
    
    formatter = logging.Formatter(
        LOGGING_CONFIG['format'],
        datefmt=LOGGING_CONFIG['date_format']
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger

