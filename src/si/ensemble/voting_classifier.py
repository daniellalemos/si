# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '/Users/danielalemos/si/src/si/data')
sys.path.insert(1, '/Users/danielalemos/si/src/si/metrics')

from dataset import Dataset
from accuracy import accuracy

import numpy as np


class VotingClassifier:
    """
    Ensemble classifier that uses the majority vote to predict the class labels
    """
    def __init__(self, models: list):
        """
        Initialize the ensemble classifier

        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble
        """
        # parameters
        self.models = models

    def fit(self, dataset: Dataset) -> 'VotingClassifier':
        """
        Fit the models according to the given training data

        Parameters
        ----------
        dataset : Dataset
            The training data
        """
        for model in self.models:
            model.fit(dataset)

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X

        Parameters
        ----------
        dataset : Dataset
            The test data
        """

        # helper function
        def _get_majority_vote(pred: np.ndarray) -> int:
            """
            It returns the majority vote of the given predictions

            Parameters
            ----------
            pred: np.ndarray
                The predictions to get the majority vote of
            """
            # get the most common label
            labels, counts = np.unique(pred, return_counts=True)
            return labels[np.argmax(counts)]

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the mean accuracy on the given test data and labels

        Parameters
        ----------
        dataset : Dataset
            The test data
        """
        return accuracy(dataset.y, self.predict(dataset))


if __name__ == '__main__':
    # import dataset
    sys.path.insert(1, '/Users/danielalemos/si/src/si/model_selection')
    sys.path.insert(1, '/Users/danielalemos/si/src/si/neighbors')
    sys.path.insert(1, '/Users/danielalemos/si/src/si/linear_model')

    from split import train_test_split
    from knn_classifier import KNNClassifier
    from logistic_regression import LogisticRegression
 
    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN and Logistic classifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    # initialize the Voting classifier
    voting = VotingClassifier([knn, lg])

    voting.fit(dataset_train)

    # compute the score
    score = voting.score(dataset_test)
    print(f"Score: {score}")

    print(voting.predict(dataset_test))