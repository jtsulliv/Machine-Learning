# -*- coding: utf-8 -*-
"""
Created on Tue Nov 03 16:55:11 2015

@author: jsullivan
"""

# Lesson 6:  Data science pipeline with pandas, seaborn, scikit-learn
# Regression problems

# AGENDA
# How do I use the pandas library to read data into Python?
# How do I use the seaborn library to visualize data?
# What is linear regression, and how does it work?
# How do I train and interpret a linear regression model in scikit-learn?
# What are some evaluation metrics for regression problems?
# How do I choose which features to include in my model?

# Types of Supervised Learning:
# Classification: Predict a categorical response
# Regression: Predict a continuous response

# So far we've focused on classification...now we'll focus on regression

import pandas as pd
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# display the first 5 rows
print data.head() # This is a pandas dataframe, a single column is a 'series'

# display the last 5 rows
print data.tail()

print data.shape

# What are the features of our data set?
# TV:  Advertising dollars spent on TV for a product in a given market (in thousands of dollars)
# Radio: Advertising dollars on Radio
# Newspaper: Advertising dollars on newspaper

# What is the response?
# Sales: Sales of a single product in a given market (in thousands of items)

# Our response variable is continuous, thus this is a regression problem

# seaborn is a python library for statistical data visualization built on top of 
# matplotlib

import seaborn as sns

# visualize the relationship between features and response using scatterplots 
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales',size=7,aspect=0.7, kind='reg')
# kind='reg' adds a line of best fit, along with a 95% confidence band

# Linear Regression
# Pros: fast, no tuning required, highly interpretable, well-understood
# Linear regression is a type of machine learning model, regression is a type 
# of supervised learning 

# Cons: unlikely to produce the best predictive accuracy (presumes a linear relationship)

# y = B_o + B_1*x_1 + B_2*x_2+...B_n*x_n
# B terms are learned during the machine learning process using "least squares"
# x terms are the features 

# create a list of feature names
X = data[['TV','Radio','Newspaper']]

#print the first five rows
print X.head()

# check the type and shape of X
print type(X)
print X.shape

# select a series from the dataframe
y = data['Sales']

# Equivalent command that works if there are no spaces in the column name
y = data.Sales
print y.head()
print type(y)
print y.shape

# splitting X and y into training and testing sets 
from sklearn.cross_validation import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

# default split is 75% train, 25% test
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape

# import model
from sklearn.linear_model import LinearRegression as LR

# instantiate 
linreg = LR()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# interpreting model coefficients 
print 'These are the original model coefficients'
print linreg.intercept_
print linreg.coef_

# pair the feature names with the coefficients 
feature_cols=['TV','Radio','Newspaper']
zip(feature_cols, linreg.coef_)

# How do we interpret the TV coefficient (0.0466)?
# For a given amount of Radio and Newspaper ad spending, a "unit" increase in TV ad spending 
# is associated with a 0.0466 "unit" increase in sales
# $1000 spent on TV adds results in an increase in sales of 46.6 items 

# Note: This is 'association' not 'causation'
y_pred = linreg.predict(X_test)

# Model evaluation for linear regression
# root-mean-square-error
from sklearn import metrics
import numpy as np
print 'This is the original RMSE'
print np.sqrt(metrics.mean_squared_error(y_test, y_pred)) 

# Try without the 'Newspaper' feature
# prep the features
X = data[['TV','Radio']]
y = data['Sales']
# split into train/test set
X_train, X_test, y_train, y_test = tts(X, y, random_state=1)
# instantiate the model
linreg = LR()
# fit the model
linreg.fit(X_train, y_train)
# interpreting model coefficients 
print 'These are the new model coefficients'
print linreg.intercept_
print linreg.coef_
# make predictions
y_pred = linreg.predict(X_test)
# evaluate
print 'This is the new RMSE'
print np.sqrt(metrics.mean_squared_error(y_pred, y_test))








