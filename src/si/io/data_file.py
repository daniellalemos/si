# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '/Users/danielalemos/si/src/si/data')
from dataset import Dataset

import numpy as np

def read_data_file(filename: str, sep: str = None, label: bool = False) -> Dataset:
    """
    Reads a data file into a Dataset object

    Parameters
    ----------
    filename : str
        Path to the file
    sep : str, optional
        The separator used in the file, by default None
    label : bool, optional
        Whether the file has a label, by default False
    """

    raw_data = np.genfromtxt(filename, delimiter=sep)

    if label:
        X = raw_data[:, :-1]
        y = raw_data[:, -1]

    else:
        X = raw_data
        y = None

    return Dataset(X, y)


def write_data_file(filename: str, dataset: Dataset, sep: str = None, label: bool = False) -> None:
    """
    Writes a Dataset object to a data file

    Parameters
    ----------
    filename : str
        Path to the file
    dataset : Dataset
        The dataset object
    sep : str, optional
        The separator used in the file, by default None
    label : bool, optional
        Whether to write the file with label, by default False
    """
    if not sep:
        sep = " "

    if label:
        data = np.hstack((dataset.X, dataset.y.reshape(-1, 1)))
    else:
        data = dataset.X

    return np.savetxt(filename, data, delimiter=sep)


if __name__ == "__main__":
    path = "/Users/danielalemos/si/datasets/breast-bin.data"
    data = read_data_file(path, ",", True)
    print(f'Testing the read_data_file\n')
    print(data.summary())
    #Testing the write_data
    path1 = "/Users/danielalemos/si/datasets/dataset2.txt"
    write_data_file(path1,data, ",", True)