"""Tests for the HousingPreprocessor class."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing

from california_housing.preprocessing.preprocessor import HousingPreprocessor


@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    data = fetch_california_housing(as_frame=True)
    return data.frame.copy()


def test_preprocessor_initialization():
    """Test preprocessor initialization with default parameters."""
    preprocessor = HousingPreprocessor()
    assert preprocessor.handle_outliers is True
    assert preprocessor.handle_skewness is True


def test_fit_transform(sample_data):
    """Test fit_transform method."""
    preprocessor = HousingPreprocessor()
    X_transformed = preprocessor.fit_transform(sample_data)
    
    # Check output shape
    assert X_transformed.shape[0] == sample_data.shape[0]
    assert X_transformed.shape[1] == sample_data.shape[1]
    
    # Check if data is standardized
    for col in X_transformed.columns:
        assert -4 <= X_transformed[col].mean() <= 4
        assert 0.5 <= X_transformed[col].std() <= 2


def test_transform_without_fit():
    """Test transform without fit raises error."""
    preprocessor = HousingPreprocessor()
    with pytest.raises(ValueError):
        preprocessor.transform(pd.DataFrame({'col': [1, 2, 3]}))


def test_outlier_removal():
    """Test outlier removal functionality."""
    # Create data with obvious outliers
    data = pd.DataFrame({
        'col1': [1, 2, 3, 100],  # 100 is an outlier
        'col2': [1, 2, 3, 4]
    })
    
    preprocessor = HousingPreprocessor(handle_outliers=True)
    X_transformed = preprocessor.fit_transform(data)
    
    # Check if outlier was handled
    assert np.abs(X_transformed['col1']).max() < 10


def test_skewness_transformation():
    """Test skewness transformation functionality."""
    # Create highly skewed data
    data = pd.DataFrame({
        'col1': np.exp([1, 2, 3, 4]),  # exponentially distributed
        'col2': [1, 2, 3, 4]  # normally distributed
    })
    
    preprocessor = HousingPreprocessor(handle_skewness=True)
    X_transformed = preprocessor.fit_transform(data)
    
    # Check if skewness was reduced
    assert abs(pd.DataFrame(X_transformed).skew()['col1']) < 1
