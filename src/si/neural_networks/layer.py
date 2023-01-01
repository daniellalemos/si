# -*- coding: utf-8 -*-


import sys
sys.path.insert(1, '/Users/danielalemos/si/src/si/statistics')
from sigmoid_function import sigmoid_function


import numpy as np


class Dense:
    """
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer

    Parameters
    ----------
    input_size: int
        The number of inputs the layer will receive
    output_size: int
        The number of outputs the layer will produce

    Attributes
    ----------
    weights: np.ndarray
        The weights of the layer
    bias: np.ndarray
        The bias of the layer
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer

        Parameters
        ----------
        input_size: int
            The number of inputs the layer will receive
        output_size: int
            The number of outputs the layer will produce
        """
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        self.X = None
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer 
        Returns a 2d numpy array with shape (1, output_size)

        Parameters
        ----------
        X: np.ndarray
            The input to the layer
        """
        self.X = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Performs a backward pass of the layer
        Returns the error of the previous layer

        Parameters
        ----------
        error: np.ndarray
            Error value of the loss function
        learning_rate: float
            Value of the learning rate
        """
        error_to_propagate = np.dot(error, self.weights.T)

        self.weights = self.weights - learning_rate * np.dot(self.X.T, error)
        self.bias = self.bias - learning_rate * np.sum(error, axis = 0)

        return error_to_propagate


class SigmoidActivation: 
    """
    A sigmoid activation layer
    """

    def __init__(self):
        """
        Initialize the sigmoid activation layer
        """
        self.X = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input
        Returns a 2d numpy array with shape (1, output_size)
    
        Parameters
        ----------
        input_data: np.ndarray
            The input to the layer
        learning_rate: float
            Value of the learning rate
        """
        self.X = input_data
        return sigmoid_function(input_data)

    def backward(self, error: np.ndarray,learning_rate: float) -> np.ndarray:
        """
        Performs a backward pass of the layer
        Returns the error of the previous layer

        Parameters
        ----------
        error: np.ndarray
            Error value of the loss function
        learning_rate: float
            Value of the learning rate
        """
        deriv_sig = sigmoid_function(self.X) * (1-sigmoid_function(self.X))
        return error * deriv_sig


class SoftMaxActivation: 
    """
    A softmax activation layer
    """

    def __init__(self):
        """
        Initialize the softmax activation layer
        """
        self.X = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Computes the probability of each class

        Parameters
        ----------
        input_data: np.ndarray
            The input to the layer
        
        """
        self.X = input_data
        exp = np.exp(input_data - np.max(input_data))
        return exp / np.sum(exp, axis = 1, keepdims = True)

    def backward(self, error: np.ndarray,learning_rate: float) -> np.ndarray:
        """
        Performs a backward pass of the layer
        Returns the error of the previous layer

        Parameters
        ----------
        error: np.ndarray
            Error value of the loss function
        learning_rate: float
            Value of the learning rate
        """
        return error

class ReLUActivation: 
    """
    A ReLu activation layer
    """

    def __init__(self):
        """
        Initialize the ReLu activation layer
        """
        self.X = None
        
    def forward(self, input_data: np.ndarray) -> np.ndarray: 
        """
        Calculates the rectified linear relashioship

        Parameters
        ----------
        input_data: np.ndarray
            The input to the layer
        """
        self.X = input_data
        return  np.maximum(0, input_data)
    
    def backward(self, error: np.ndarray,learning_rate: float) -> np.ndarray:
        """
        Performs a backward pass of the layer
        Returns the error of the previous layer

        Parameters
        ----------
        error: np.ndarray
            Error value of the loss function
        learning_rate: float
            Value of the learning rate
        """
        self.X = np.where(self.X > 0, 1 , 0)
        return error * self.X

class LinearActivation: 
    """
    The Linear activation layer
    """

    def __init__(self):
        """
        Initialize the linear activation layer
        """
        self.X = None
        
    def forward(self, input_data: np.ndarray) -> np.ndarray: # static method
        """
        Calculates the linear activation algorithm

        Parameters
        ----------
        input_data: np.ndarray
            The input values to be activated
        """
        self.X = input_data
        return input_data

    def backward(self, error: np.ndarray,learning_rate: float) -> np.ndarray:
        """
        Performs a backward pass of the layer
        Returns the error of the previous layer

        Parameters
        ----------
        error: np.ndarray
            Error value of the loss function
        learning_rate: float
            Value of the learning rate
        """
        return error 