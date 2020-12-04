# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:45:24 2019

@author: HP
"""

#import quandl
import numpy as np
import pandas as pd
#preprocessing - Scaling
#cross_validation - splitting data set into Train and Test
from sklearn import preprocessing, cross_validation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import load_boston

boston = load_boston()
print(boston)


df_x=pd.DataFrame(boston.data,columns=boston.feature_names)
df_y=pd.DataFrame(boston.target)
df_x.describe()