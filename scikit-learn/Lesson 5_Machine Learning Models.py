# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:06:33 2015

@author: jsullivan
"""

# Lesson 5 - Choosing a Machine Learning Model
#               How do I choose which model to use for my supervised learning?
#                   ()
#               How do I choose the best tuning parameters for that model?
#               How do I estimate the likely performance of my model on out-of-sample data?


# There are different Model Evaluation Procedures


# 1) - Train and test on the entire dataset 
#           1) - Train the model on the enitre dataset 
#           2) - Test the model on the same dataset, evaluate how well we did 
#                   by comparing the predicted response values with the true response values.


# LOAD THE DATA
# read in the iris dataset 
from sklearn.datasets import load_iris
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target




# LOGISTIC REGRESSION----------------------------------------------------------
# import
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with the data
logreg.fit(X, y)

# predict the response values for the observations in X
logreg.predict(X)

# store the predicted response values 
y_pred = logreg.predict(X)

# check how many predictions were generated
print len(y_pred)

# Need to determine how well the model performed
# compute classification accuracy for the logistic regression model
from sklearn import metrics
print metrics.accuracy_score(y, y_pred)
#------------------------------------------------------------------------------



#KNN (K=5)---------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print metrics.accuracy_score(y, y_pred)
#------------------------------------------------------------------------------



#KNN (K=1)---------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
y_pred = knn.predict(X)
print metrics.accuracy_score(y, y_pred)
#------------------------------------------------------------------------------
# These predictions are on the entire sample set, so the model may be overfit
# i.e., the model is learning the noise rather than generalizing to find the signal
# If a model is overfitting, then it is considered overly complex

# Evaluation procedure #2:  Train/test split
# 1) Split the dataset into a training set and a test set
# 2) Train the model on training set
# 3) Test the model on the testing set, and evaluate how well the model did
#    This is a better representation of how the model will perform on out-of-sample data

# print the shapes of X and y
print X.shape
print y.shape

# STEP 1:  split X and y into training and test sets 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=4) 
# The model is tested on 40% of the sample data
# Models are typically tested on 20% - 40% of the data
# Data is assigned using a random process
# random_state=4 provides a random state to use to test various models

# Shapes of the new objects
#print X_train.shape
#print X_test.shape
#print y_train.shape
#print y_test.shape

# STEP 2: train the model on the training set 
# Instantiate the model
logreg = LogisticRegression()
# Fit the model
logreg.fit(X_train, y_train)

# STEP 3: make predictions on the test set
y_pred = logreg.predict(X_test)

# compare the actual response values (y_test) with predicted response values (y_pred)
print metrics.accuracy_score(y_test, y_pred)

# Repeat for KNN = 5 
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred)

# Repeat for KNN = 1 
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred)

# Find a better K
score_list = []
for k in range (1,26):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    temp = metrics.accuracy_score(y_test, y_pred)
    score_list.append(temp)


import matplotlib.pyplot as plt

# plot the relationship between K and testing accuracy
k_range = range(1, 26)
plt.plot(k_range, score_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

# For KNN models, complexity is determined by the value of K (lower value = more complex)

# Now that we have identified the optimum value of K (11), train the entire 
# model on the sample set using K = 11 to make predictions on out-of-sample data
knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit (X, y)
knn.predict([3, 5, 4, 2])

# Downsides of train/test split?
# Provides a high-variance estimate of out-of-sample accuracy
# K-fold cross-validation overcomes this limitation
# But, train/test split is still useful because of its flexibility and speed







