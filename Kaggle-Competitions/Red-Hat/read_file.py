# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 14:26:07 2016

@author: JSULLIVAN
"""

# Reading in the files and exporting a subset of the rows to take a quick look
# at the data


import pandas as pd
import numpy as np



act_train = 'act_train.csv'     # Activity test file
people = 'people.csv'           # People file


# Function to read in the files to pandas dataframe
def read_file(act_train):
    df = pd.read_csv(act_train)
    return df 
        
df_act = read_file(act_train)   # Activities training set dataframe
df_p = read_file(people)        # People dataframe




# Function to output a subset of the data to Excel for easier viewing 
def write_to_xlsx(df,file_name):
    df.to_excel(file_name)
    
    
write_to_xlsx(df_act[0:500],'act_train_subset.xlsx')
write_to_xlsx(df_p[0:500], 'people.xlsx')




# Quick look at 'act_test.csv' data  
print list(df_act)                              # Listing out the features
print df_act.describe()                         # Quick description of the data


# Quick look at 'people.csv' data
print list(df_p)                                # Listing out the features
print df_p.describe()                           # Quick description of the data