import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import process_and_save_dataset
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

if __name__ == "__main__":
    process_and_save_dataset()

