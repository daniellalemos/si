# -*- coding: utf-8 -*-

from typing import Callable

import sys
sys.path.insert(1, '/Users/danielalemos/si/src/si/data')
sys.path.insert(1, '/Users/danielalemos/si/src/si/statistics')
from dataset import Dataset
from f_classification import f_classification

import numpy as np


class SelectPercentile:
    """
    Select features according to a given percentile
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks
    """
    def __init__(self,  percentile: float, score_func: Callable = f_classification):
        """
        Select features according to a given percentile

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: float
            Percentage of features to select
        
        Attributes
        ----------
        F: array, shape (n_features,)
            F scores of features
        p: array, shape (n_features,)
            p-values of F-scores
        
        """
        #parameters
        self.percentile = percentile
        self.score_func = score_func

        #atributes
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentil to compute the F scores and p-values

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the highest scoring features according to a given percentile
        Returns a labeled dataset with the selected features

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        features_number = round(len(dataset.features) * self.percentile) 
        idxs = np.argsort(self.F)[-features_number:]
        new_features = np.array(dataset.features)[idxs] 
        return Dataset(X = dataset.X[:, idxs], y = dataset.y, features = list(new_features), label = dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectPercentil and transforms the dataset by selecting the highest scoring features 
        according to the a given percentile
        Returns a labeled dataset with the selected features

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":
    x = np.array([[1,2,3,4],[0,1,3,4],[5,6,0,1],[1,3,7,0]])
    y = np.array([1,1,0,0])
    data = Dataset(x,y)
    selector = SelectPercentile(0.5, f_classification)
    new_data = selector.fit_transform(data)
    print(new_data.X)