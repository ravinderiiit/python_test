# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:48:46 2018

@author: Hp First
"""

import quandl
import pandas as pd
import numpy as np
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

df = quandl.get("WIKI/anydata_set")

 #we consider only  the Adj. Close column for our predictions.



df = df[['Adj. Close']]

forecast_out = int(20)  # predicting 20 days into future

# To fill our output data with data to be trained upon, we will set our  prediction column equal to our Adj. Close column, but shifted 20 units up.
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out) #  label column with data shifted 20 units up

X = np.array(df.drop(['Prediction'],1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(df['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)

confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)