"""Data preprocessing utilities."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer


class HousingPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for California Housing dataset."""
    
    def __init__(self, handle_outliers: bool = True, handle_skewness: bool = True):
        """Initialize preprocessor.
        
        Args:
            handle_outliers: Whether to remove outliers using IQR method
            handle_skewness: Whether to transform skewed features
        """
        self.handle_outliers = handle_outliers
        self.handle_skewness = handle_skewness
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.skewed_features = None
        
    def _get_outlier_mask(self, X: pd.DataFrame) -> pd.Series:
        """Get boolean mask for outliers using IQR method.
        
        Returns:
            Boolean mask where True indicates an outlier
        """
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        return ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    
    def _identify_skewed_features(self, X: pd.DataFrame, threshold: float = 0.75) -> list:
        """Identify features with high skewness."""
        skewness = X.skew()
        return list(skewness[skewness > threshold].index)
    
    def fit(self, X: pd.DataFrame, y=None) -> 'HousingPreprocessor':
        """Fit preprocessor to data."""
        if self.handle_skewness:
            self.skewed_features = self._identify_skewed_features(X)
            if self.skewed_features:
                self.power_transformer.fit(X[self.skewed_features])
        
        self.scaler.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Transform data.
        
        Args:
            X: Features to transform
            y: Optional target variable to filter along with X
            
        Returns:
            Transformed features and optionally filtered target
        """
        X_copy = X.copy()
        
        if self.handle_outliers:
            mask = self._get_outlier_mask(X_copy)
            X_copy = X_copy[~mask]
            if y is not None:
                y = y[~mask]
        
        if self.handle_skewness and self.skewed_features:
            X_copy[self.skewed_features] = self.power_transformer.transform(X_copy[self.skewed_features])
        
        X_transformed = self.scaler.transform(X_copy)
        
        if y is not None:
            return X_transformed, y
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Fit and transform data.
        
        Args:
            X: Features to transform
            y: Optional target variable to filter along with X
            
        Returns:
            Transformed features and optionally filtered target
        """
        self.fit(X)
        return self.transform(X, y)


def prepare_data(df: pd.DataFrame, target_col: str = 'MedHouseVal') -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from raw dataframe.
    
    Args:
        df: Raw dataframe
        target_col: Name of target column
        
    Returns:
        Tuple of (features, target)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
