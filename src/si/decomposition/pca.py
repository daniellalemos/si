# -*- coding: utf-8 -*-

from typing import Callable

import sys
sys.path.insert(1, '/Users/danielalemos/si/src/si/data')
sys.path.insert(1, '/Users/danielalemos/si/src/si/io')
from dataset import Dataset
from csv_file import read_csv

import numpy as np

class PCA: 
    """
    It performs principal component analysis on the dataset 
    It uses linear algebra principles to transform possibly correlated variables into a smaller number of variables capable of representing the data
    It uses SVD (Singular Value Decomposition)
    """
    def __init__(self, n_components: int):
        """
        PCA algorithm

	Parameters
	----------
	n_components: int
		Number of components

	Attributes
	----------
	mean: np.ndarray
		Mean of each feature of the dataset
	components: np.ndarray
		The principal components 
	explained_variance: np.ndarray
		The explained variance
        """

        #parameters
        self.n_components = n_components

        #attributes
        self.mean = None
        self.components = None
        self.explained_variance = None


    def fit(self, dataset: Dataset) -> 'PCA':
        '''
        It fits PCA on the dataset
        The PCA algorithm estimates the mean, the principal components and the explained variance

	Parameters
	----------
	dataset: Dataset
		A Dataset object
        '''
        #calculates the means of the samples
        self.mean = np.mean(dataset.X, axis = 0) 

        #centers the data deducing the meab from the dataset
        centered = dataset.X - self.mean 

        #calculates the SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices = False) 

        #get the first n components of vt
        self.components = Vt[:self.n_components] 

        #get the explained variance that corresponds to the first n components of EV
        EV = (S**2)/(dataset.shape()[0] - 1)  
        self.explained_variance = EV[:self.n_components] 

        return self

    def transform(self, dataset: Dataset) -> np.ndarray:
        '''
	It transforms the dataset
        Calculates the reduced dataset using the principal components

	Parameters
	----------
	dataset: Dataset
		A Dataset object
        '''
        centered = dataset.X - self.mean 
        V = np.transpose(self.components) 
        return np.dot(centered, V)      
    
    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        '''
	It fits and transforms the dataset
		
	Parameters
	----------
	dataset: Dataset
		A Dataset object
	'''
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    path = '/Users/danielalemos/si/datasets/iris.csv'
    data = read_csv(filename = path, sep = ',', features = True, label = True)
    pca = PCA(n_components = 5)
    reduced = pca.fit_transform(data)
    print(reduced)
