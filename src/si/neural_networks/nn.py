# -*- coding: utf-8 -*-

from typing import Callable

import sys
sys.path.insert(1, '/Users/danielalemos/si/src/si/data')
sys.path.insert(1, '/Users/danielalemos/si/src/si/metrics')
from dataset import Dataset
from accuracy import accuracy
from mse import mse, mse_derivative

import numpy as np


class NN:
    """
    The NN is the Neural Network model.
    It comprehends the model topology including several neural network layers.
    The algorithm for fitting the model is based on backpropagation.

    Parameters
    ----------
    layers: list
        List of layers in the neural network
    epochs: int
        Number of epochs to train the model
    learning_rate: float
        The learning rate of the model
    loss: Callable
        The loss function to use
    loss_derivative: Callable
        The derivative of the loss function to use
    verbose: bool
        Whether to print the loss at each epoch

    Attributes
    ----------
    history: dict
        The history of the model training
    """
    def __init__(self, layers: list, epochs: int = 1000, learning_rate: float = 0.01, loss: Callable = mse, loss_derivative: Callable = mse_derivative, verbose: bool = False):
        """
        Initialize the neural network model

        Parameters
        ----------
        layers: list
            List of layers in the neural network
        epochs: int
            Number of epochs to train the model
        learning_rate: float
            The learning rate of the model
        loss: Callable
            The loss function to use
        loss_derivative: Callable
            The derivative of the loss function to use
        verbose: bool
            Whether to print the loss at each epoch
        """
        # parameters
        self.layers = layers
        self.epochs = int(epochs)
        self.learning_rate = learning_rate
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.verbose = verbose

        # attributes
        self.history = {}

    def fit(self, dataset: Dataset) -> 'NN':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """

        for epoch in range(1, self.epochs + 1):

            y_pred = dataset.X.copy()
            y_true = np.reshape(dataset.y,(-1,1))

            # forward propagation
            for layer in self.layers:
                y_pred = layer.forward(y_pred)

            # backward propagation
            error = self.loss_derivative(y_true, y_pred)
            for layer in self.layers[::-1]:
                error = layer.backward(error, self.learning_rate)

            # save history
            cost = self.loss(y_true, y_pred)
            self.history[epoch] = cost

            # print loss
            if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} - cost: {cost}')

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the output of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of
        """
        X = dataset.X.copy()

        # forward propagation
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def cost(self, dataset: Dataset) -> float:
        """
        It computes the cost of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost on
        """
        y_pred = self.predict(dataset)
        return self.loss(dataset.y, y_pred)

    def score(self, dataset: Dataset, scoring_func: Callable = accuracy) -> float:
        """
        It computes the score of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score on
        scoring_func: Callable
            The scoring function to use
        """
        y_pred = self.predict(dataset)
        return scoring_func(dataset.y, y_pred)


if __name__ == '__main__':
    pass