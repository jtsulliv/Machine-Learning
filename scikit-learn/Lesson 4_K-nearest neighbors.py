# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:03:40 2015

@author: jsullivan
"""

# This is Lesson 4 - using the IRIS toy dataset 
# The observations are categorical, thus this is a CLASSIFICATION problem

# K-NEAREST NEIGHBORS classification algorithm

# These are the steps:
# 1)  Pick a value for K (i.e. 5)
# 2)  The model searches for the K observations in the training data that are 
#       'nearest' to the measurements of the unknown iris (numerical distance)
# 3)  Use the most popular response value from the K nearest neighbors as the 
#       predicted response value for the unknown iris.


#------------------------------------------------------------------------------
# Loading the data
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target

# print the shapes
print X.shape
print y.shape
#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
# scikit-learn 4-step ML modeling pattern

# 1) - Import the class you plan to use
from sklearn.neighbors import KNeighborsClassifier

# 2) - "Instantiate" the "estimator"
#           "estimator" is scikit-learn's term for model
#           "instantiate" means make an instance of 
knn = KNeighborsClassifier(n_neighbors=1)   # n_neighbors is the tuning parameter
print knn  # will show the other default parameters we did not adjust (see scikit-learn documentation)

# 3) - Fit the model with the data (aka "model training")
knn.fit(X, y)

# 4) - Predict the response for a new observation
#           New observations are called "out-of-sample" data
#           Uses the information it learned during the model training process
print knn.predict([3, 5, 4, 2])

# Can predict multiple responses at once
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print knn.predict(X_new)        
#------------------------------------------------------------------------------





#------------------------------------------------------------------------------
# Using a different value for K... this is known as "model tuning"

# Instantiate the model
knn = KNeighborsClassifier(n_neighbors=5)   # n_neighbors is the tuning parameter

# Fit the model
knn.fit(X, y)

# Make predictions
print knn.predict(X_new)        
#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
# Using a different classification model (logistic regression)

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model
logreg.fit(X, y)

# predict the response
print logreg.predict(X_new) 

 

 