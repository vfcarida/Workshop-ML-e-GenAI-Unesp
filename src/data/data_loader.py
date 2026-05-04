"""
Data loading and preprocessing utilities.

Provides functions to read datasets (CSV, etc.) cleanly with error handling.
"""

import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_csv_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        filepath (str): The path to the CSV file.
        **kwargs: Additional keyword arguments to pass to pandas.read_csv.

    Returns:
        pd.DataFrame: The loaded dataset.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"The file {filepath} does not exist.")
        
    try:
        df = pd.read_csv(path, **kwargs)
        logger.info(f"Successfully loaded {filepath} with shape {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"The file {filepath} is empty.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while reading {filepath}: {e}")
        raise

def preprocess_regression_data(df: pd.DataFrame, target_col: str):
    """
    Separates features and target variable for regression models.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        target_col (str): The name of the target column.
        
    Returns:
        tuple: (X, y) where X is the feature dataframe and y is the target series.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
        
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y
