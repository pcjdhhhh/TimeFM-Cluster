# -*- coding: utf-8 -*-

import pandas
import random
from scipy import stats
from scipy.io import loadmat
import math
import numpy as np
from matplotlib.image import imread
from matplotlib import pyplot as plt




def get_UCR_datasets(file_name):
    test_file_path = 'UCR2018/' + file_name + '/' + file_name + '_TEST'
    train_file_path = 'UCR2018/' + file_name + '/' + file_name + '_TRAIN'
    test_data = np.loadtxt(test_file_path,delimiter=",")
    train_data = np.loadtxt(train_file_path,delimiter=",")
    
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0].astype(int)
    
    X_test = test_data[:,1:]
    y_test = test_data[:,0].astype(int)
    
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])
    
    return X,y
    
    





