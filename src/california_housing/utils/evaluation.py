"""Model evaluation utilities."""

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
    """Evaluate model using cross-validation.
    
    Args:
        model: Fitted sklearn model
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with evaluation metrics
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    y_pred = model.predict(X)
    
    return {
        'r2_cv_mean': cv_scores.mean(),
        'r2_cv_std': cv_scores.std(),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'r2': r2_score(y, y_pred)
    }


def format_metrics(metrics: Dict[str, float]) -> pd.Series:
    """Format metrics for display.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Formatted pandas Series
    """
    return pd.Series({
        'R² (CV)': f"{metrics['r2_cv_mean']:.3f} ± {metrics['r2_cv_std']:.3f}",
        'RMSE': f"{metrics['rmse']:.3f}",
        'R²': f"{metrics['r2']:.3f}"
    })
