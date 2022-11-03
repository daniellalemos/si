# -*- coding: utf-8 -*-

import sys

sys.path.insert(1, '/Users/danielalemos/si/src/si/data')
sys.path.insert(1, '/Users/danielalemos/si/src/si/metrics')
sys.path.insert(1, '/Users/danielalemos/si/src/si/statistics')
from dataset import Dataset
from accuracy import accuracy
from sigmoid_function import sigmoid_function

import numpy as np


class LogisticRegression:
    """
    The LogisticRegression is a logistic model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique
    """
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000, use_adaptive_alpha: bool = True):
        """
        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations

        Attributes
        ----------
        theta: np.array
            The model parameters, namely the coefficients of the logistic model.
            For example, sigmoid(x0 * theta[0] + x1 * theta[1] + ...)
        theta_zero: float
            The intercept of the logistic model
        cost_history: dict
            Contains the results of the cost function at each iteration
        """

        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.use_adaptive_alpha = use_adaptive_alpha

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}

    def gradient_descent (self, dataset: Dataset, m: int) -> None:
        '''Algorithm based on a convex function and tweaks its parameters iteratively to minimize 
        a given function to its local minimum

        Parameters
        ----------
        dataset: Dataset
            The dataset object
        m: int
            Number of examples in the dataset
        '''
        # predicted y
        y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

        # apply sigmoid function
        y_pred = sigmoid_function(y_pred)

        # compute the gradient using the learning rate
        gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

        # compute the penalty
        penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

        # update the model parameters
        self.theta = self.theta - gradient - penalization_term
        self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

    def _regular_fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset but not update the learning rate (alpha).
        The Gradient Descent should stop when the difference between the cost of the previous and the 
        current iteration is less than 0.0001

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        for i in range(self.max_iter):

             #gradient descent
            self.gradient_descent(dataset,m)
            
            #stores the cost in the cost_history
            self.cost_history[i] = self.cost(dataset)

            if i!=0 and self.cost_history[i-1]-self.cost_history[i] < 0.0001:
                break
            
        return self

    def _adpative_fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset and updates the learning rate (alpha).
        The parameter alpha should be decreased by half when the difference between the cost of the previous and the 
        current iteration is less than 0.0001

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        for i in range(self.max_iter):

             #gradient descent
            self.gradient_descent(dataset,m)
            
            #stores the cost in the cost_history
            self.cost_history[i] = self.cost(dataset)

            if i!=0 and self.cost_history[i-1]-self.cost_history[i] < 0.0001:
                self.alpha /= 2

        return self

    def fit (self, dataset: Dataset) -> 'LogisticRegression':
        ''' Fit the model to the dataset
        If the use_adaptive_alfa is True, fits the model updating the alpha else fits the model not updating the alpha

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        '''
        return self._adpative_fit(dataset) if self.use_adaptive_alpha else self._regular_fit(dataset)

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of
        """
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        # convert the predictions to 0 or 1 (binarization)
        mask = predictions >= 0.5
        predictions[mask] = 1
        predictions[~mask] = 0
        return predictions

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on
        """
    
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (-dataset.y * np.log(predictions)) - ((1 - dataset.y) * np.log(1 - predictions)) #tem o - porque na formula tem a multiplicar por -1/m
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * dataset.shape()[0]))
        return cost

  
if __name__ == '__main__':
    # import dataset
    sys.path.insert(1, '/Users/danielalemos/si/src/si/model_selection')
    from split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # fit the model
    model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=2000, use_adaptive_alpha = True)
    model.fit(dataset_train)

    # compute the score
    score = model.score(dataset_test)
    print(f"Score: {score}")