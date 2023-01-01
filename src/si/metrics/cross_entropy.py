# -*- coding: utf-8 -*-

import numpy as np


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
     It returns the cross entropy of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset s
    """
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)


def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    It returns the derivative of the cross entropy

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    return y_pred - y_true 


