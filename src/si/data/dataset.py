# -*- coding: utf-8 -*-

from cProfile import label
from typing import Tuple,Sequence

import numpy as np
import pandas as pd


class Dataset:
    '''
    Creates a tabular dataset for machine learning
    '''

    def __init__ (self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None):
        '''
        Stores the values for X, y, feautures and label from dataset

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            matrix/feature table 
        y: numpy.ndarray (n_samples, 1)
            label vector
        features: list of str (n_features)
            feature name vector
        label: str (1)
            label vector name
        '''

        if X is None:
            raise ValueError("X can't be None")

        if features is None:
            features = [str(i) for i in range(X.shape[1])]
        else:
            features = list(features)

        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int,int]:
        ''' 
        Returns the shape of the dataset (n_samples, n_features)
        '''
        return self.X.shape

    def has_label(self) -> bool:
        ''' 
        Returns True if the dataset has a label
        '''
        if self.y is not None:
            return True
        return False

    def get_classes(self) -> np.ndarray:
        '''
        Returns the unique classes of the dataset 
        '''
        if self.y is None:
            raise ValueError ('The dataset does not have a label')
        return np.unique(self.y)
    
    def get_mean(self) -> np.ndarray:
        '''
        Returns the mean of each feature
        '''
        return np.nanmean(self.X, axis = 0)
    
    def get_variance(self) -> np.ndarray:
        '''
        Returns the variance of each feature
        '''
        return np.nanvar(self.X, axis = 0)
    
    def get_median(self) -> np.ndarray:
        ''' 
        Returns the median of each feature
        '''
        return np.nanmedian(self.X, axis = 0)
    
    def get_min(self) -> np.ndarray:
        ''' 
        Returns the minimum value of each feature
        '''
        return np.nanmin(self.X, axis = 0)

    def get_max(self) -> np.ndarray:
        ''' 
        Returns the maximum value of each feature
        '''
        return np.nanmax(self.X, axis = 0)

    def summary(self) -> pd.DataFrame:
        ''' 
        Returns a DataFrame containing descriptive metrics (mean, median, varaince, minimum 
        and maximum values of each feature)
        '''
        data = pd.DataFrame (
            {'Mean': self.get_mean(),
            'Median': self.get_median(),
            'Variance': self.get_variance(),
            'Min': self.get_min(),
            'Max': self.get_max()}
        )
        return data

    def dropna (self) -> None:
        '''
        Remove the samples with at least one missing value (NaN)
        '''
        mask = np.isnan(self.X).any(axis = 1)
        self.X = self.X[~mask] 
        if self.has_label():
            self.y = self.y[~mask] 

    def fillna (self,new_value: int) -> np.ndarray:
        '''
        Replaces the missing values (NaN) with a given value

        Parameters
        ----------
        new_value: int
            Given value to replace the missing values
        '''
        return np.nan_to_num(self.X, nan = new_value, copy = False)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df
    
    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data
        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)


if __name__ == '__main__':
    x1 = np.array([[1,2,3],[1,2,3]])
    y1 = np.array([1,2])
    features = ['A','B','C']
    label ='y'
    dataset = Dataset(X = x1,y = y1, features = features, label = label) 

    print('Testing the metrics\n')
    print('Shape:',dataset.shape())
    print('Label:',dataset.has_label())
    print('Classes:',dataset.get_classes())
    print('Mean:',dataset.get_mean())
    print(f'Summary:\n',dataset.summary())

    print(f'\nTesting the drop Nan\n')
    X2 = np.array([[1,2,3],[1,np.nan,np.nan],[4,5,6]])
    y2 = np.array([7,8,9])
    ds2 = Dataset(X2, y2, features = features, label = label)
    print(f'Inicial dataframe\n')
    print(ds2.to_dataframe())
    ds2.dropna()
    print(f'\nDataframe after dropna\n')
    print(ds2.to_dataframe())

    print(f'\nTesting the fill Nan\n')
    X3 = np.array([[1,2,3],[1,np.nan,np.nan],[4,5,6]])
    y3 = np.array([7,8,9])
    ds3 = Dataset(X3, y3, features = features, label = label)
    print(f'Inicial dataframe\n')
    print(ds3.to_dataframe())
    ds3.fillna(8)
    print(f'\nDataframe after fillna\n')
    print(ds3.to_dataframe())
