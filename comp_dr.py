# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 03:28:56 2019

@author: Shaurya Gupta
"""


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("input/train.csv", header=None)
dataset = dataframe.values
X = dataset[1:-1,1:785].astype(float)
Y = dataset[1:-1,0]