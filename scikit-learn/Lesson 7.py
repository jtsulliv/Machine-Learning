# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 14:50:51 2015

@author: JSULLIVAN
"""

# Lesson 7: Optimizing your model with cross-validation

# Agenda:
# What is the drawback of the train/test split procedure for model validation?
# How does K-fold cross-validation overcome this limit?
# How can cross-validation be used for selecting tuning parameters, choosing between models,
# and selecting features?
# What are some possible improvements to cross-validation?

# Review
# Motivation:  Need a way to choose between machine learning models
#   -Goal is to estimate likely performance of a model on out-of-sample data
# Initial Idea:  Train and test on the same data
#   -But, maximizing training accuracy rewards overly complex models that overfit the training data
# Alternative idea: Train/test split 
#   -split the dataset into two pieces, so that the model can be trained and tested on different data
#   -testing accuracy is a better estimate than training accuracy of out-of-sample performance
#   -but, it provides a high variance estimate since changing which observations happen to be in the testing
#    set can significantly change testing accuracy

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier as knn_
from sklearn import metrics 

# read in the iris data 
iris = load_iris()
X = iris.data
y = iris.target

# train/test split 
X_train, X_test, y_train, y_test = tts(X, y, random_state=4)

# check classification of KNN with K=5
knn = knn_(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred)

# What if we created a bunch of train/test splits, calculated the accuracy for each,
# then averaged the results together?
# That's the essence of cross-validation

# Steps for K-fold cross-validation
# 1)  Split the data into K equal partitions (or "folds")
# 2)  Use fold 1 as the testing set and the union of the other folds as the training set
# 3)  Calculate testing accuracy
# 4)  Repeat steps 2 and 3 K times, using a different fold as the testing set each time 
# 5)  Use the average testing accuracy as the estimate of out-of-sample accuracy

# simulate splitting a dataset of 25 observations into 5 folds 
from sklearn.cross_validation import KFold
kf = KFold(25, n_folds=5, shuffle=False)

# print the contents of each training and testing set 
print '{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations')
for iteration, data in enumerate(kf, start=1):
    print '{:^9} {} {:^25}'.format(iteration, data[0], data[1])
    
# Comparing cross-validation to train/test split 
# Advantages of cross-validation:
    # More accurate estimate of out-of-sample accuracy
    # More "efficient" use of data (every observation is used for both training and testing)
# Advantages of train/test split:
    # Runs K times faster than K-fold cross-validation
    # Simpler to examine the detailed results of the testing process
    
# Cross-validation recommendations
# 1) K can be any number, but K=10 is generally recommended 
# 2) For classification problems, stratified sampling is recommended for creating folds
    # Each response class should be represented with equal proportions in each of the K folds
    # scikit-learn does this by default
    
# Cross-validation example: parameter tuning 
    # Goal: Select the best tuning parameters (aka "hyperparameters") for KNN on the iris dataset

from sklearn.cross_validation import cross_val_score

# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = knn_(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print scores

# use average accuracy as an estimate of out-of-sample accuracy
print scores.mean()

# search for an optimal value of k for KNN
k_range = range(1,31)
k_scores=[]
for k in k_range:
    knn = knn_(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    mean = scores.mean()
    k_scores.append(mean)
print k_scores

import matplotlib.pyplot as plt

# plot the relationship between K and testing accuracy
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

# Which value of K do we choose?
# Generally, you should choose the value of K that produces the simplest model
# In the case of KNN, higher values of K produce lower complexity models 
# Thus, choose K=20

# Compare KNN with logistic regression on the iris dataset 

# 10-fold cross-validation with the best KNN model 
knn = knn_(n_neighbors=20)
print cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()

# 10-fold cross-validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean()

#------------------------------------------------------------------------------
# Cross-validation example: feature selection
# Goal:  Select whether the Newspaper feature should be included in the linear 
# regression model on the advertising dataset

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# read in the advertising data into a dataframe 
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

print data.head()
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales


# 10-fold cross-validation with all three features
lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
scores = -scores
print scores








    

    




