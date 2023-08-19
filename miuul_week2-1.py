# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 18:57:57 2023

@author: ali_t
"""

import numpy as np
import pandas as pd
import seaborn as sns



if __name__ == '__main__':
   
    #Importing dataframe called "Titanic" with the help of seaborn
    df = sns.load_dataset("titanic")
    #print(df.info())
    #print(df.head())
    
    
    #Finding sex of people
    sex = df.groupby(["alive", "sex"]).agg({ "alive": ["count"]})
    print(sex)
    
    #Finding numbers of unique values for every column
    values = {column: df[column].nunique() for column in df.columns}
    print(values)
    
    
    #Finding the number of unique values for "pclass" column
    pclassVal = df.groupby("pclass").agg({"pclass": ["count"]})
    print("Pclass unique values are {}".format(pclassVal))
    totalVal = df.groupby('pclass').agg({'pclass': ["count"]})
    print(totalVal)
    
    #Finding the unique values of 'pclass' and 'parch' columns
    totalValue = pd.concat([df['pclass'], df['parch']]).unique()
    print(totalValue)
    
    #Checking the Dtype of 'embark_town' column and changed it as 'category' type
    print(df['embarked'].dtype)
    df = df.astype({'embarked': 'category'})
    print(df['embarked'].dtype)
    
    
    #Selecting rows which has embarked has 'C'
    embarkedFilter = df['embarked'] == 'C'
    print(df[embarkedFilter])
    
    
    #Selecting rows which has embarked has not 'S'
    embarkedFilter = df['embarked'] != 'S'
    print(df[embarkedFilter])
    
    
    #Selecting rows which has age less than 30 and woman as gender
    filtered = (df['sex'] == 'female') & (df['age'] < 30)
    print(df[filtered])
    
    
    #Selecting rows that includes age is bigger than 70, or fare is bigger than 500
    filtered = (df['age'] > 70) | (df['fare'] > 500)
    print(df[filtered])
    
    #Finding NaN values for every single column
    print(df.isnull().sum())
    
    
    #Removing who column from dataframe
    newdf = df.drop(['who'], axis = 1)
    print('who' in newdf.columns)
    
    #Filling 'NaN' values of 'deck' column
    deckMode = df.mode()['deck'][0]
    temp = df
    temp['deck']= df['deck'].fillna(deckMode)
    print(temp.tail())
    
    
    #Filling 'age' column with mean value of age column
    print(df.tail())
    ageMean = df['age'].mean()
    temp = df
    temp['age'] = temp['age'].fillna(ageMean)
    print(temp['age'])
    
    
    #Aggregating columns pclass and sex and seeing survived column count
    result = df.groupby(['sex', 'pclass']).agg({'survived': ['sum', 'count', 'mean']})
    print(result)
    
    
    #Creating new column called 'age_flag', if age is less than 30 flag will be 1, otherwise 0
    df['age_flag'] = df.loc[:, 'age'].apply(lambda x: 1 if x < 30 else 0)
    print(df.tail())
    
    
    #Importing dataset called 'tips'
    df = sns.load_dataset('tips')
    print(df.info())
    print(df.head())
    
    
    
    #Getting time column with time column
    times = df.groupby(['time']).agg({'total_bill': ['min', 'max', 'mean']})
    print(times)
    
    
    #Getting female's total_bill and tip for day
    filtered = df['sex'] == 'Female'
    output = df[filtered].groupby('day').agg({'total_bill': ['min', 'max', 'mean'], 'tip' : ['min', 'max', 'mean']})
    print(output)

    
    #Finding the mean of total bills are bigger than 10 and size is less than 3
    output = df.loc[(df['total_bill'] > 10) & (df['size'] < 3), ['total_bill']].agg({'total_bill': ['mean']})
    print(output)
    
    
    #Creating a column called 'total_bill_tip_sum' as total_bill + tip
    df['total_bill_tip_sum'] = df['total_bill'] + df['tip']
    series = df.loc[0:30,'total_bill_tip_sum']
    series = series.sort_values(ascending = False)
    print(series)
    
    
    
    