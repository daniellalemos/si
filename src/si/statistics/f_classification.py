# -*- coding: utf-8 -*-

from typing import Tuple, Union

import sys
sys.path.insert(1, '/Users/danielalemos/si/src/si/data')
from dataset import Dataset

import numpy as np
from scipy import stats


def f_classification(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray],Tuple[float, float]]:
    """
    Scoring function for classification problems. It computes one-way ANOVA F-value for the
    provided dataset. The F-value scores allows analyzing if the mean between two or more groups (factors)
    are significantly different. Samples are grouped by the labels of the dataset.

    Parameters
    ----------
    dataset: Dataset
        A labeled dataset
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    F, p = stats.f_oneway(*groups)
    return F, p
