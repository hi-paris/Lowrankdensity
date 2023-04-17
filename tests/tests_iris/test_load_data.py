import pytest
import numpy as np
from DensLowRank.dataset.load_data import load_data
from DensLowRank.dataset.load_data import load_data_path
import os


def test_load_data_path():
    # check the path exists
    iris_data = load_data_path()
    assert os.path.exists(iris_data)

def test_load_data():
    # check iris data 
    df = load_data()
    assert df.shape == (150, 2)
    # type assertion
    assert df.iloc[0].dtype == 'int64'
    assert df.iloc[1].dtype == 'int64'