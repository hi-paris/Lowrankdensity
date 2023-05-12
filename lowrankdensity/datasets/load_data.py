# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 14, 2023
"""

import numpy as np
import pandas as pd
import os
from os.path import join



def load_iris():
    """Load the iris dataset containing only sepal length and petal length in a categorical format

    """
    path = join("", 'datasets/iris/iris_cat.csv')
    print(path)
    df = pd.read_csv(path)
    print(df)
    return df



