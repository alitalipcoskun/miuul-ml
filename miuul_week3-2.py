# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:43:12 2023()

@author: ali_t
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno




def CaptureNaNValues(df, na_name = False):
    
    nancol=[column for column in df.columns if df[column].isnull().sum() > 0]
    n_miss = df[nancol].isnull().sum().sort_values(ascending = False)
    ratio = (df[nancol].isnull().sum() / df.shape[0] * 100).sort_values(ascending = False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ['n_miss', 'ratio'])
    
    print(missing_df)
    
    if na_name:
        return nancol



def missing_vs_target(df, target, na_cols):
    tempdf = df.copy()
    
    for column in na_cols:
        tempdf[column + '_NA_FLAG'] = np.where(df[column].isnull(), 1, 0)
        print(column, tempdf.groupby(column + '_NA_FLAG')[target].count())
    
    
    
    
    
    na_flags = tempdf.loc[:, tempdf.columns.str.contains('_NA_')].columns
    
    
    for column in na_flags:
        print(pd.DataFrame({"Target Mean": tempdf.groupby(column)[target].mean(),
                            "Count": tempdf.groupby(column)[target].count()}), end="\n###########################################################\n")
    





if __name__ == '__main__':
    
    df = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week3\\titanic.csv")
    
    
    #Finding that dataset has any null value or not
    """
    print(df.isnull().values.any())
    print(((df.isnull().sum() / df.shape[0]) * 100).sort_values(ascending = False))


    nancol = [column for column in df.columns if df[column].isnull().any()]
    print(nancol)
    
    msno.bar(df)
    plt.show()
    
    
    msno.matrix(df)
    plt.show()
    """
    
    na_cols = CaptureNaNValues(df, na_name = True)
    missing_vs_target(df, "Survived", na_cols)
    
    