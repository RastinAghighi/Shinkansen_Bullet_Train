"""
Data ingestion module for archive extraction and dataframe merging.
"""
import pandas as pd
import zipfile
import os
import logging
from .config import ZIP_PATH, FILES

logger = logging.getLogger(__name__)

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts and merges travel and survey datasets from the compressed archive.
    """
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Archive missing at specified path: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        data = {k: pd.read_csv(z.open(v)) for k, v in FILES.items()}

    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')

    return train, test