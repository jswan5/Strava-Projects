# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 11:04:18 2021

@author: Jordan Swanson

Using Strava data from personal runs, exported via Golden Cheetah
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Grab file list
fileList = glob.glob("*.csv")
# Stick all the dataframes together
data = pd.concat((pd.read_csv(f) for f in fileList)).reset_index()
print(data.describe())
corr = data.corr()

# plt.figure(1)
# plt.ylabel("Cadence (SPM)")
# plt.hist(data["cad"], bins=100)
# plt.figure(2)
# plt.ylabel("Heart Rate (RPM)")
# plt.boxplot(data["hr"])
# plt.figure(3)
# plt.ylabel("Speed (kph)")
# plt.boxplot(data["kph"])


y = data["kph"]
X = data[["cad", "hr", "alt"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101)

linear_regression = LinearRegression()
linear_regression.fit(X,y)
y_pred = linear_regression.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

coeff_df = pd.DataFrame(linear_regression.coef_, X.columns, columns=['Coefficient'])
coeff_df.loc["Intercept"] = linear_regression.intercept_

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))