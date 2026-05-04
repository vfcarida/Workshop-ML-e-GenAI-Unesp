import pytest
from PIL import Image
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.visual import image_grid
from src.data.data_loader import load_csv_data, preprocess_regression_data

def test_image_grid():
    # Create dummy images
    img1 = Image.new('RGB', (100, 100), color = 'red')
    img2 = Image.new('RGB', (100, 100), color = 'blue')
    
    grid = image_grid([img1, img2], rows=1, cols=2)
    
    assert grid.size == (200, 100)
    
def test_image_grid_error():
    img1 = Image.new('RGB', (100, 100), color = 'red')
    with pytest.raises(AssertionError):
        image_grid([img1], rows=2, cols=2)

@patch("src.data.data_loader.Path.exists")
@patch("src.data.data_loader.pd.read_csv")
def test_load_csv_data(mock_read_csv, mock_exists):
    mock_exists.return_value = True
    mock_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    mock_read_csv.return_value = mock_df
    
    df = load_csv_data("dummy_path.csv")
    assert df.equals(mock_df)

def test_preprocess_regression_data():
    df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'target': [5, 6]})
    X, y = preprocess_regression_data(df, 'target')
    
    assert list(X.columns) == ['feature1', 'feature2']
    assert y.name == 'target'
