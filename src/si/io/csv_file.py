# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '/Users/danielalemos/si/src/si/data')
from dataset import Dataset
import numpy as np


def main():
    x = np.array([[1,2,3],[1,2,3]])
    y = np.array([1,2])
    features = ['A','B','C']
    label ='y'
    ds = Dataset(x,y, features = features, label = label)
    print(ds.shape())
    print(ds.has_label())
    print(ds.get_classes())
    print(ds.get_mean())
    print(ds.get_variance())
    print(ds.get_median())
    print(ds.get_min())
    print(ds.get_max())
    print(ds.summary())

if __name__ == "__main__":
    main()