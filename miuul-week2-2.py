# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:23:15 2023

@author: ali_t
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', False)


def category_summary(data_frame, column_name):
    result = pd.DataFrame({column_name: data_frame[column_name].value_counts(),
                           "Ratio": 100*data_frame[column_name].value_counts() / len(data_frame)
                           })
    print(result, end="\n\n")
    

def number_summary(dataframe, column_name):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print((dataframe[column_name].describe(quantiles)).T, end="\n\n")
    



def grabbing_column_names(dataframe, category_th = 10, cardinality_th = 20):
    
    #Finding actual categories
    categories = [column for column in dataframe.columns if str(dataframe[column].dtypes) in ['object', 'category', 'bool']]
    
    
    #Finding numbers with the help of category_th
    fake_numbers = [ column for column in dataframe.columns if (str(dataframe[column]) in ['int64', 'float64']) & (dataframe[column].nunique() < category_th)]
    
    #Finding garbage categories with cardinality_th
    fake_category = [column for column in dataframe.columns if (dataframe[column].dtypes in ['object', 'category']) & (dataframe[column].nunique() > cardinality_th)]
    print(fake_category)
    categories = categories + fake_numbers
    
    categories = [column for column in categories if column not in fake_category]
    
    nums = [column for column in dataframe.columns if column not in categories]
    
    return {'category': categories,
            'number': nums
        }


def identifier(df):
    #Finding actual categories
    category = [column for column in df.columns if str(df[column].dtypes) in ['object', 'category', 'bool']]

    #Finding numbers which is category actually, however its dtype is number
    fake_nums = [column for column in df.columns if ((df[column].dtypes in ['int64', 'float64']) & (df[column].nunique() < 8))]

    #Finding garbage categories
    fake_category = [column for column in df.columns if ((str(df[column].dtypes) in ['object', 'category']) & (df[column].nunique() > 20))]

    
    category = category + fake_nums
    category = [column for column in category if column not in fake_category]
    nums = [column for column in df.columns if column not in category]
    
    return {'category': category,
            'number': nums,
        }


def target_summary_with_cat(dataframe, target, categorical_column):
    print(pd.DataFrame({"Target Mean": dataframe.groupby(categorical_column)[target].mean()}))


if __name__ == '__main__':
    df = sns.load_dataset('titanic')
    df.info()
    
    column_info = grabbing_column_names(df)
    
    """
    for column in column_info['category']:
        category_summary(df, column)
    
    for column in column_info['number']:
        number_summary(df, column)
    """
    for column in column_info['category']:
        target_summary_with_cat(df, 'survived', column)
        
        
        
        
        
        
        
        
        




