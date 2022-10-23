# -*- coding: utf-8 -*-

import numpy as np


def rmse (y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the accuracy of the model on the given dataset
    
    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    return np.sqrt(np.sum((y_true - y_pred)**2) / y_true.shape[0])
