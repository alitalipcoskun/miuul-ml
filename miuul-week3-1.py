# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:29:53 2023

@author: ali_t
"""

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def data_categorization(df, cat_th = 10, car_th = 20):
    
    #Finding categoric values
    category = [column for column in df.columns if df[column].dtypes in ['object', 'category', 'bool']]
    
    #Finding numbers
    number = [column for column in df.columns if df[column].dtypes in ['float64', 'int64']]
    
    #Finding numbers that behaves like category
    num_but_cat = [column for column in number if df[column].nunique() < cat_th]
    
    #Finding categories that behaves like cardinal
    cat_but_car = [column for column in category if df[column].nunique() > car_th]

    #Removing categories from numbers
    number = [column for column in number if (column not in num_but_cat)]
    
    #Removing cardinals from category
    category = [column for column in category if column not in cat_but_car]
    
    
    #removing numbers which behaveslike category
    number = [column for column in number if column not in num_but_cat]
    
    #appending categories which behaves like numbers
    category = category + num_but_cat
    
    
    
    
    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'Category Columns: {len(category)}')
    print(f"Number Columns: {len(number)}")
    print(f"Categoric but Cardinal: {len(cat_but_car)}")
    print(f"Number but Categoric: {len(num_but_cat)}")
    print("############################################")
    print("Result:")
    print(f"{len(category)} category, {len(number)} number, {len(cat_but_car)} cardinal variables.")
    print("############################################")
    
    return category, number, cat_but_car


def load():
    data = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week3\\titanic.csv")
    return data


def load_big_data():
    data = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week3\\application_train.csv")
    return data


def Outliers(dataframe, column):
    low, up = OutlierTresholds(dataframe, column)
    
    if dataframe[(dataframe[column] < low) | (dataframe[column] > up)].any(axis = None):
        return True
    else:
        return False


def OutlierTresholds(dataframe, column, q1 = 0.25, q3 = 0.75):
    quartile1 = dataframe[column].quantile(q1)
    quartile3 = dataframe[column].quantile(q3)
    
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    
    return low_limit, up_limit


def GrabOutliers(df, column, index = False):
    low, up = OutlierTresholds(df, column)

    
    if df[(df[column] < low) | (df[column] > up)].shape[0] > 10:
        print(df[(df[column] < low) | (df[column] > up)].head())
    else:
        print(df[(df[column] < low) | (df[column] > up)])
    
    if index == True:
        outlier_index = df[(df[column] < low) | (df[column] > up)].index
        return outlier_index


def CheckOutliers(df, column):
    low, up = OutlierTresholds(df, column)
    if df[(df[column] < low) | (df[column] > up)].any(axis = None) == False:
        return False
    else:
        return True


def ReplaceWithTresholds(df, column, q1 = 0.25, q3 = 0.75):
    low, up = OutlierTresholds(df, column, q1 = q1, q3 = q3)
    
    df.loc[(df[column] < low), column] = low
    df.loc[(df[column] > up), column] = up










if __name__ == '__main__':
    """
    dfsmall = load()
    dfbig = load_big_data()
    #print(dfsmall.head())
    #print(dfbig.head())
    #print(dfsmall.columns)
    #Outliers
    category, number, cat_but_car = data_categorization(dfsmall)
    
    
    for column in number:
        ReplaceWithTresholds(dfsmall, column)

    for column in number:
        print(column, CheckOutliers(dfsmall, column))
    """
    
    df = sns.load_dataset('diamonds')
    df = df.select_dtypes(include = ['int64', 'float64'])
    df = df.dropna()
    
    for column in df.columns:
        print(column, CheckOutliers(df, column))
        
    clf = LocalOutlierFactor(n_neighbors= 20)
    clf.fit_predict(df)
    df_scores = clf.negative_outlier_factor_
    print(df_scores)
    sortedVal = np.sort(df_scores)
    
    
    scores = pd.DataFrame(sortedVal)
    scores.plot(stacked = True, xlim = [0, 20], style = '.-')
    plt.show()
    
    #grafikten treshold hakkında yorum yaptık
    th = sortedVal[3]
    
    print(df[df_scores < th])
    
    
    