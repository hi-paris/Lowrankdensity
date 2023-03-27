# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 14, 2023
"""

import numpy as np
import pandas as pd
from os.path import join
from scipy import sparse
import os

def load_iris():
    """Load the iris dataset containing only sepal length and petal length in a categorical format

    """
    path = join("", 'iris/iris_cat.csv')
   # path =' DensLowRank/dataset/iris/iris_cat.csv'
    print(path)
    df = pd.read_csv(path)
    return df


