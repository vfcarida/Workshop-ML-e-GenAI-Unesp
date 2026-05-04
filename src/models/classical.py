"""
Classical Machine Learning Models module.

Provides easy-to-use wrappers around scikit-learn models like
Linear Regression, Logistic Regression, KMeans, etc., 
with built-in error handling and type hints.
"""

import logging
from typing import Optional, Any
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import mean_squared_error, accuracy_score
except ImportError:
    logging.warning("scikit-learn is not installed. Classical ML models will not work.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Base class for training ML models.
    """
    def __init__(self, model: Any):
        self.model = model
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        try:
            if y is not None:
                self.model.fit(X, y)
            else:
                self.model.fit(X)
            logger.info(f"Successfully fitted {self.model.__class__.__name__}")
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error predicting: {e}")
            raise

class SimpleLinearRegression(ModelTrainer):
    """
    Wrapper for Linear Regression.
    """
    def __init__(self):
        super().__init__(LinearRegression())
        
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculates and returns the Mean Squared Error."""
        mse = mean_squared_error(y_true, y_pred)
        logger.info(f"Mean Squared Error: {mse:.4f}")
        return mse

class SimpleLogisticRegression(ModelTrainer):
    """
    Wrapper for Logistic Regression.
    """
    def __init__(self, C: float = 1.0):
        super().__init__(LogisticRegression(C=C))
        
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculates and returns the Accuracy Score."""
        acc = accuracy_score(y_true, y_pred)
        logger.info(f"Accuracy Score: {acc:.4f}")
        return acc

class SimpleKMeans(ModelTrainer):
    """
    Wrapper for KMeans clustering.
    """
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        super().__init__(KMeans(n_clusters=n_clusters, random_state=random_state))
