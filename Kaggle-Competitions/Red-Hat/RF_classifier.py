# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 12:15:00 2016

@author: EMASON
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

#Set seed so that can repeat experiments
np.random.seed(1)

#Load preprocessed Training Data
train_df=pd.read_csv('train_preproc.csv', index_col=0)

# Create Training Set without activity ID and outcome
X=train_df.drop(['activity_id','outcome'],axis=1)

# Create df of Outcome Only
target=train_df['outcome']

# Create Generator for 3 fold stratification
skf=StratifiedKFold(train_df['outcome'], n_folds=3,shuffle=True)

# Create Random Forest Classifier - all parameters are default
rfc=RandomForestClassifier()

# For each of the folds, train and test
for train_index, test_index in skf:
    
    X_train=X.iloc[train_index]
    X_test=X.iloc[test_index]
    y_train=target[train_index]
    y_test=target[test_index]
    
    #Create Classifer
    rfc.fit(X_train, y_train)
    y_score=rfc.predict(X_test)
    
    
    # Confusion Matrix
    print confusion_matrix(y_test, y_score, labels=[0,1])
    
    #Print ROC Curve
    fpr, tpr, _= roc_curve(y_test.values, y_score)
    roc_auc=auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()



