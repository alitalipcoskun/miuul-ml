# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:27:37 2023

@author: ali_t
"""

import numpy as np
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

if __name__ == '__main__':
    
    df = sns.load_dataset("titanic")
    #print(df.head())
    mariage = df['alone'].value_counts()
    #print(df.info())
    #print(df.isnull().values.any())
    ageFilter = (df["age"] > 50) & (df["sex"] == 'male')
    
    #print(df.loc[ageFilter, 
     #            ['age','class', 'sex']])
    
    
    print(df.dtypes)
    functions = ["min", "max", "sum", "mean"]
    df.drop(["pclass"], axis = 1, inplace = True)
    aggDict = {column: functions for column in df.columns if (df[column].dtype == np.float64) or (df[column].dtype == np.int64)}
    #{for column in df.columns}
    aggDict["sex"] = "count"
    #print(df.groupby(["sex", "embark_town", "class"]).agg({
    #    "age": "mean",
    #    "survived": "mean",
    #   "sex": "count"}))
    
    df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
    table = df.pivot_table("survived", "sex", ["class", "new_age"])
    print(table)