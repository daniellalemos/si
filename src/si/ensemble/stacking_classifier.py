# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '/Users/danielalemos/si/src/si/data')
sys.path.insert(1, '/Users/danielalemos/si/src/si/metrics')

from dataset import Dataset
from accuracy import accuracy

import numpy as np


class StackingClassifier:
    """
    Ensemble classifier that uses a set of models to generate predictions
    """
    def __init__(self, models: list, final_model):
        """
        Initialize the ensemble classifier

        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble
        final_model:
            Final model
        """
        # parameters
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models according to the given training data

        Parameters
        ----------
        dataset : Dataset
            The training data
        """
        for model in self.models:
            model.fit(dataset)

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        self.final_model.fit(Dataset(predictions,dataset.y))
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Estimates the output variable using the trained models and the final model

        Parameters
        ----------
        dataset : Dataset
            The test data
        """
        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        return self.final_model.predict(Dataset(predictions,dataset.y))

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

    # initialize the KNN, Logistic classifier and final_model
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    fmodel = KNNClassifier()

    # initialize the Stacking classifier
    stacking = StackingClassifier([knn, lg], fmodel)

    stacking.fit(dataset_train)

    # compute the score
    score = stacking.score(dataset_test)
    print(f"Score: {score}")

    print(stacking.predict(dataset_test))