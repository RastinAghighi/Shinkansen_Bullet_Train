"""
Data ingestion module.
Handles robust, OS-agnostic loading of datasets via configuration mapping.
"""
import logging
import sys
from pathlib import Path
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

def load_data(config_path: str = 'configs/base.yaml') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw datasets using paths defined in the configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration profile.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The training and testing matrices.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)
        
    # pathlib handles cross-OS slash routing automatically
    train_path = Path(config['data']['train_path'])
    test_path = Path(config['data']['test_path'])
    
    if not train_path.exists() or not test_path.exists():
        logger.error(
            f"Data extraction failed. Verify the following paths exist:\n"
            f"Train: {train_path.resolve()}\n"
            f"Test: {test_path.resolve()}"
        )
        sys.exit(1)
        
    logger.info(f"Ingesting training matrix from {train_path.name}")
    train_df = pd.read_csv(train_path)
    
    logger.info(f"Ingesting testing matrix from {test_path.name}")
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df