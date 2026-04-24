# -*- coding: utf-8 -*-

"""
Pycaret
Author: Mavs
Description: Module containing all constants of pycaret.

"""

from typing import Union

import numpy as np
import pandas as pd
from scipy import sparse

# Group of variable types for isinstance
SEQUENCE = (list, tuple, np.ndarray, pd.Series)

# Variable types for type hinting
SEQUENCE_LIKE = Union[SEQUENCE]
DATAFRAME_LIKE = Union[dict, list, tuple, np.ndarray, sparse.spmatrix, pd.DataFrame]
TARGET_LIKE = Union[int, str, list, tuple, np.ndarray, pd.Series]

# Column name that contains the predicted label in predict_model's output
LABEL_COLUMN = "prediction_label"

# Column name that contains the predicted label in predict_model's output
SCORE_COLUMN = "prediction_score"
