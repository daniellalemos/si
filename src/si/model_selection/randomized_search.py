# -*- coding: utf-8 -*-

import itertools
from typing import Callable, Tuple, Dict, List, Any


import sys
sys.path.insert(1, '/Users/danielalemos/si/src/si/data')
sys.path.insert(1, '/Users/danielalemos/si/src/si/model_selection')

from dataset import Dataset
from cross_validate import cross_validate

import numpy as np

def randomized_search_cv(model, dataset: Dataset, parameter_distribution: Dict[str, Tuple], scoring: Callable = None, cv: int = 3, n_iter: int = 10, test_size: float = 0.2) -> List[Dict[str, Any]]:
    """
    Performs a randomized search cross validation on a model.
    Returns a list of dictionaries with the combination of parameters and the scores

    Parameters
    ----------
    model:
        The model to cross validate
    dataset: Dataset
        The dataset to cross validate on
    parameter_distribution: Dict[str, Tuple]
        The parameter to use
    scoring: Callable
        The scoring function to use
    cv: int
        The cross validation folds
    n__iter: int
        Number of random combinations of parameters
    test_size: float
        The test size
    """
    # validate the parameters
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    scores = []

    # for each combination
    for i in range(n_iter):

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter in parameter_distribution: 
            #take a random value from the value distribution of each parameter
            value = np.random.choice(parameter_distribution[parameter])
            setattr(model, parameter, value)
            parameters[parameter] = value

        # cross validate the model
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        # add the parameter configuration
        score['parameters'] = parameters

        # add the score
        scores.append(score)

    return scores


if __name__ == '__main__':
    # import dataset
    sys.path.insert(1, '/Users/danielalemos/si/src/si/linear_model')
    from logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the Logistic Regression model
    knn = LogisticRegression()

    # parameter distribution
    parameter_distribution_ = {
        'l2_penalty': np.linspace(1,10,10),
        'alpha':  np.linspace(0.001,0.0001,100),
        'max_iter': np.linspace(1000, 2000, 200, dtype=np.int64)
    }
    
    # cross validate the model
    scores_ = randomized_search_cv(knn,dataset_,parameter_distribution=parameter_distribution_, cv=3, n_iter=10)

    # print the scores
    print(scores_)