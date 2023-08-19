# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 16:15:47 2023

@author: ali_t
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn as sns

from sklearn import preprocessing
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



def rare_analyser(df, target, cat_cols):
    for column in cat_cols:
        print(column, ':', len(df[column].value_counts()))
        print(pd.DataFrame({"COUNT": df[column].value_counts(),
                            "RATIO": df[column].value_counts()/len(df),
                            "TARGET_MEAN": df.groupby(column)[target].mean()}))


def get_columns(df, cat_th = 10, car_th = 20,get_columns = False):
    columns = df.columns
    
    categories = [column for column in columns if str(df[column].dtypes) in ['object', 'category', 'bool']]
    numbers = [column for column in columns if str(df[column].dtypes) in ['float64', 'int64']]    
    cat_but_car = [column for column in categories if df[column].nunique() > car_th]
    
    num_but_cat = [column for column in numbers if df[column].nunique() < cat_th]
    
    
    
    categories = categories + num_but_cat
    categories = [column for column in categories if column not in cat_but_car]
    
    numbers = [column for column in numbers if column not in num_but_cat]
    
    """
    print(len(df.columns))
    print(df.columns)
    print("Categories \n\n", len(categories), categories)
    print("Numbers \n\n", len(numbers), numbers)
    print("Cardinals \n\n", len(cat_but_car), cat_but_car)
    """
    
    
    if get_columns:
        return categories, numbers, cat_but_car

def label_encoder(dataframe, binary_col):
    le = preprocessing.LabelEncoder()
    for column in binary_col:
        dataframe[column] = le.fit_transform(dataframe[column])
    return dataframe



def rare_encoder(df, category, rare_percentage = 0.01):
    temp_df = df.copy()    
    rare_cols = [column for column in temp_df.columns if temp_df[column].dtypes == 'O' and (temp_df[column].value_counts() / len(temp_df) < rare_percentage).any(axis = None)]
    
    print(rare_cols)
    for var in rare_cols:
        temp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = temp[temp < rare_percentage].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
        
        
    return temp_df



"""
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis = None)]
    
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    
    return temp_df

"""


if __name__ == '__main__':
    df = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week3\\datasets\\titanic.csv")
    dfbig = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week3\\datasets\\application_train.csv")
    
    binary_col = [column for column in df.columns if (str(df[column].dtypes) not in ['float64', 'int64'] and df[column].nunique() == 2)]
    binary_col_big = [column for column in dfbig.columns if (str(dfbig[column].dtypes) not in ['float64', 'int64'] and dfbig[column].nunique() == 2)]
    
    category, number, cardinal = get_columns(dfbig, get_columns= True)
    temp_df = rare_encoder(dfbig, 0.1)
    #rare_analyser(temp_df, 'TARGET', category)
    print(temp_df.groupby('OCCUPATION_TYPE')['TARGET'].count())
    
    
    """
    print(binary_col)
    print(dfbig[binary_col_big].head())
    dff = df.copy()
    dff_big = dfbig.copy()
    dff = label_encoder(dff, binary_col)
    dff_big = label_encoder(dff_big, binary_col_big)
    print(dff_big[binary_col_big].head())
    """