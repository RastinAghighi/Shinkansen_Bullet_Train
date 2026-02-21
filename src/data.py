"""
Data ingestion module.
Handles robust, OS-agnostic loading of datasets via configuration mapping.
"""
import logging
import sys
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw datasets using paths defined in the configuration dictionary.
    
    Args:
        config (dict): Runtime configuration dictionary containing data paths.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The training and testing matrices.
    """
    train_path = Path(config['data']['train_path'])
    test_path = Path(config['data']['test_path'])
    
    if not train_path.exists() or not test_path.exists():
        logger.error(
            "Data extraction failed. Verify the following paths exist:\n"
            "Train: %s\n"
            "Test: %s", train_path.resolve(), test_path.resolve()
        )
        sys.exit(1)
        
    logger.info("Ingesting training matrix from %s", train_path.name)
    train_df = pd.read_csv(train_path)
    
    logger.info("Ingesting testing matrix from %s", test_path.name)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df