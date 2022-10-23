# -*- coding: utf-8 -*-

from typing import Callable, Union

import numpy as np

import sys
sys.path.insert(1, '/Users/danielalemos/si/src/si/data')
sys.path.insert(1, '/Users/danielalemos/si/src/si/metrics')
sys.path.insert(1, '/Users/danielalemos/si/src/si/statistics')

from dataset import Dataset
from rmse import rmse
from euclidean_distance import euclidean_distance


class KNNRegressor:
    """
    KNN Regressor 
    k-Nearst Neighbors classifier is a machine learning model that approximates the association between independent variables 
    and the continuous outcome by averaging the observations in the same neighbourhood
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN Regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        
        Attributes
        ----------
        dataset: np.ndarray
            The training data
        """
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        self.dataset = dataset
        return self

    def _get_closest_values(self, sample: np.ndarray) -> float:
        """
        Returns the mean of the closest values
        """
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors indexes
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the values of the k nearest neighbors
        k_nearest_neighbors_values = self.dataset.y[k_nearest_neighbors]

        # get the mean of the values 
        return np.mean(k_nearest_neighbors_values)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of
        """
        return np.apply_along_axis(self._get_closest_values, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        It returns the root mean squared error (rmse) of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    sys.path.insert(1, '/Users/danielalemos/si/src/si/io')
    from csv_file import read_csv
    sys.path.insert(1, '/Users/danielalemos/si/src/si/model_selection')
    from split import train_test_split
    
    # load and split the dataset
    path = '/Users/danielalemos/si/datasets/cpu.csv'
    data = read_csv(path, sep = ",", features = True, label = True)
    dataset_train, dataset_test = train_test_split(data, test_size=0.2)

    # initialize the KNN regressor
    knn = KNNRegressor(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')