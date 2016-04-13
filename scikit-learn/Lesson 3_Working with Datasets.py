# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:03:40 2015

@author: jsullivan
"""

#sklearn is the scikit-learn library
from sklearn.datasets import load_iris

iris = load_iris()

print iris.data

# print the names of the four features
print iris.feature_names

# print integers representing the species of each observation
print iris.target

# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print iris.target_names

# Each value we are predicting is known as the 'response'

# There are two types of supervised learning 
# 1)  Classification:  Supervised learning in which the response is categoriacal
# i.e. is an email "SPAM" or "HAM"
# 2)  Regression:  Supervised learning in which the response is ordered and continuous
# i.e. home prices, or heights of people

#There are four requirements for working with data in scikit-learn

# 1)  The FEATURES (independent variables) and RESPONSES (outputs) are separate objects
# 2)  FEATURES and RESPONSES should be numeric 
# 3)  FEATURES and RESPONSES should be NumPy arrays
# 4)  FEATURES and RESPONSES should have specific shapes (need to have same number of rows)

# 2)  Check the types of the FEATURES and RESPONSES
print type(iris.data)   #FEATURES
print type(iris.target) #RESPONSES   

# 4)  Check the shape of the FEATURES
#       first dimension = number of observations (rows) 
#       second dimension = number of features (columns)
print iris.data.shape

#     Check the shape of the RESPONSES
print iris.target.shape

# Store the FEATURE matrix in "X"
X = iris.data

# Store the response vector in "y"
y = iris.target


 