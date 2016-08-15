# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 17:57:11 2016

@author: EMASON
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load training and test set
train_df=pd.read_csv('train_preproc.csv', index_col=0)
test_df=pd.read_csv('test_preproc.csv', index_col=0)

# Create array of training targets
target=train_df['outcome']

#Create submission dataframe with activity ids
submission=pd.DataFrame()
submission['activity_id']=test_df['activity_id']


#drop extra features -activity id, outcome, and activity 1 features
train_df=train_df.drop(['activity_id','outcome','act_char_1','act_char_2','act_char_3','act_char_4', 'act_char_5','act_char_6','act_char_7','act_char_8','act_char_9'], axis=1)


test_df=test_df.drop(['activity_id','act_char_1','act_char_2','act_char_3','act_char_4', 'act_char_5','act_char_6','act_char_7','act_char_8','act_char_9'], axis=1)
#test_df=test_df.drop(['activity_id'], axis=1)


# Create Random Forest Classifier 
rfc=RandomForestClassifier(n_estimators=100)

# Fit the Random Forest Classifier
rfc.fit(train_df, target)

# Predict outcomes for the test data
submission['outcome']=rfc.predict(test_df)

# Write submission
submission.to_csv('holdthedoor_submission4.csv', index=False)


