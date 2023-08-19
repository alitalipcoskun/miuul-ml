# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:38:11 2023

@author: ali_t
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns



def ColumnUniqueValues(dataframe, column):
    print(pd.Series({"Column Name" : column, "Unique Values" : df[column].nunique()}))





if __name__ == '__main__':
    df = pd.read_csv('C:\\Users\\ali_t\\.spyder-py3\\saves\\persona.csv')
    print(df.head())
    """
    #Find the number of unique values in "Source" column
    ColumnUniqueValues(df, "SOURCE")
    
    #Finding the number of unique values in "Price" column
    ColumnUniqueValues(df, 'PRICE')
    
    #Find the count_values of 'price' column
    print(df.groupby('PRICE').agg({'PRICE': ['count']}))
    
    #Find the sales for every country
    print(df.groupby('COUNTRY').agg({'COUNTRY': ['count']}))
    
    #Find the total profit from every country
    print(df.groupby('COUNTRY').agg({'PRICE': ['sum']}))    
    
    #Find the total sales for every source
    print(df.groupby('SOURCE').agg({'SOURCE': ['count']}))
    
    #Find the mean values of prices for every country
    print(df.groupby('COUNTRY').agg({'PRICE': ['mean']}))
    
    #Categorise with country and source to find the mean of price
    print(df.groupby(['COUNTRY', 'SOURCE']).agg({'PRICE': ['mean']}))
    """
    
    
    #print(df.groupby(['COUNTRY', 'SOURCE', 'SEX', 'AGE']).agg({'PRICE': ['mean']}))
    
    agg_df = df.sort_values(by = ['PRICE'], ascending = False)
    #print(agg_df)
    
    my_bins = [0, 18, 23, 30, 40, agg_df['AGE'].max()]
    
    
    labels = ['0_18', '19_23', '24_30', '31_40', '40_{}'.format(agg_df['AGE'].max())]
    
    agg_df.reset_index()
    agg_df['AGE_CAT'] = pd.cut(agg_df['AGE'], bins = my_bins, labels = labels)
    agg_df['CUSTOMER_BASED_LEVEL'] = ["_".join(i).upper() for i in agg_df.drop(['AGE', 'PRICE'], axis = 1).values]
    print(agg_df.head())
    agg_df.reset_index()
    agg_df['SEGMENT'] =  pd.qcut(x = agg_df['PRICE'], q=4, labels = ['D', 'C', 'B', 'A'])
    