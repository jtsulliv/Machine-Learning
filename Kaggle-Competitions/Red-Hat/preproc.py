# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 13:35:29 2016

@author: JSULLIVAN
"""

"""
Created on Thu Aug 04 11:20:20 2016
@author: JSULLIVAN
"""
# JS 8-Aug 2016
# Pre-processing the data 

import pandas as pd 
import numpy as np


act_train = 'act_train.csv'     # Activity test file
act_test = 'act_test.csv'
people = 'people.csv'           # People file


# Function to read in the files to pandas dataframe
def read_file(act_train):
    df = pd.read_csv(act_train)
    return df 
        
df_act_train = read_file(act_train)   # Activities training set dataframe
df_act_test = read_file(act_test)
df_p = read_file(people)        # People dataframe


df_p['date']=pd.to_datetime(df_p['date'])
baseline_date=df_p['date'].min()


# Function to preprocess the activity files
def preproc_act(df):
    
    #Changing the date object into a date data type 
    df['date']=pd.to_datetime(df['date'])
    
    #Adding feature for month of activity
    df['act_month']=df['date'].apply(lambda x:x.month)    
    
    #Changing date feature into a number of days from some baseline
    df['date']=df['date']-baseline_date
    df['date']=df['date'].apply(lambda x: x / np.timedelta64(1,'D'))
    
    #Might want to normalize the date days.     
    df=df.rename(columns = {'date':'act_date'})
    
    
    # Filling in the empty cells with dummy type 'type 0'    
    columns = list(df)    
    for col in columns[3:14]:
        df[col] = df[col].fillna('type 0') 
        
    # Changing categorical variables to integers 
    for col in columns[3:14]:
        df[col] = df[col].apply(lambda x: int((x.split(" ")[1])))
        
        
    # Renaming the columns to distinguish activity characteristics from people characteristics
    for col in columns[3:14]:
        df = df.rename(columns = {col:'act_'+col})
        
    # Changing people IDs to integers:
    df['people_id']=df['people_id'].apply(lambda x: int(float(x.split("_")[1])))
    
    return df 


# Function to preprocess the people data    
def people_preproc(df):
        
    #Adding feature for month of person
    df['people_month']=df['date'].apply(lambda x:x.month)
    #Change date into a number of days
    df['date']=df['date']-baseline_date
    df['date']=df['date'].apply(lambda x: x / np.timedelta64(1,'D'))

    #Might want to normalize the date days.     
    df=df.rename(columns = {'date':'people_date'})
    
    # Changing categorical variables to integers 
    columns = list(df)
    for col in (columns[1:4] + columns[5:12]):
        df[col] = df[col].apply(lambda x: int(x.split(" ")[1]))

    # Changing booleans to integers 
    for col in columns[12:40]:
        df[col] = df[col].astype(int)
    
    # Change char_30 to float
    df['char_38']=df['char_38'].astype(float)
    
    # Changing people IDs to integers:
    df['people_id']=df['people_id'].apply(lambda x:int(float(x.split("_")[1])))

    return df 


act_train_preproc = preproc_act(df_act_train)
act_test_preproc = preproc_act(df_act_test)
people_preproc = people_preproc(df_p)


# Merging the activity and people data 

def join_data(df1, df2):
    merged = df1.merge(df2, how = 'left', on = 'people_id')
    return merged 
    
train_merged = join_data(act_train_preproc, people_preproc)
test_merged = join_data(act_test_preproc, people_preproc)


# Writing the preprocessed data to csv files
train_merged.to_csv('train_preproc.csv')
test_merged.to_csv('test_preproc.csv')

