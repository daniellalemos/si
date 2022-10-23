# -*- coding: utf-8 -*-

from types import NoneType
from typing import Callable

import sys
sys.path.insert(1, '/Users/danielalemos/si/src/si/data')
sys.path.insert(1, '/Users/danielalemos/si/src/si/statistics')
from dataset import Dataset
from f_classification import f_classification

import numpy as np


class SelectKBest:
    """
    Select features according to the k highest scores
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks
    """
    def __init__(self, score_func: Callable = f_classification, k: int = 10):
        """
        Select features according to the k highest scores.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        k: int, default=10
            Number of top features to select.

        Attributes
        ----------
        F: array, shape (n_features,)
            F scores of features
        p: array, shape (n_features,)
            p-values of F-scores
        
        """
        #parameters
        self.k = k  
        self.score_func = score_func  

        #atributes
        self.F = None 
        self.p = NoneType

    def fit(self, dataset: Dataset) -> 'SelectKBest': 
        """
        It fits SelectKBest to compute the F scores and p-values

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        self.F, self.p = self.score_func(dataset)  
        return self

    def transform(self, dataset: Dataset) -> Dataset: 
        """
        It transforms the dataset by selecting the k highest scoring features
        Returns a labeled dataset with the k highest scoring features

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        idxs = np.argsort(self.F)[-self.k:]  
        features = np.array(dataset.features)[idxs] 
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label) 

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectKBest and transforms the dataset by selecting the k highest scoring features
        Returns a labeled dataset with the k highest scoring features

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

    
if __name__ == '__main__':
    X = np.array([[1,2,3,4],[0,1,3,4],[5,6,0,1],[1,3,7,0]])
    y = np.array([1,1,0,0])
    data = Dataset(X,y)
    selector = SelectKBest(f_classification, 2)
    new_data = selector.fit_transform(data)
    print(new_data.X)