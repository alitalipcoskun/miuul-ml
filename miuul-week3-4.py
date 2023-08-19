# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 00:58:26 2023

@author: ali_t
"""


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score




def missing_values_table(df, na_name = False):
    
    na_cols = [column for column in df.columns if df[column].isnull().sum() > 0]
    n_miss = df[na_cols].isnull().sum().sort_values(ascending = False)
    freq = (df[na_cols].isnull().sum()/df.shape[0]* 100).sort_values(ascending = False)
    
    missing_df = pd.concat([n_miss, np.round(freq, 2)], axis = 1, keys = ['n_miss', 'ratio'])
    print(missing_df)
    

    if na_name:
        return na_cols
    
    




def check_outlier(df, cols):        
    for col in cols:
        print(col, df[col].isnull().sum())
    


def replace_with_thresholds(df, column, q1 = 0.25, q3 = 0.75):
    q1 = df[column].quantile(q1)
    q3 = df[column].quantile(q3)
    
    iqr = q3- q1
    up = q3 + 1.5 * iqr
    down = q1 - 1.5*iqr
    
    
    df.loc[df[df[column] < down].index, column] = down
    df.loc[df[df[column] > up].index, column] = up


def label_encoding(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df


    

def get_columns(df, cat_th = 10, car_th = 20):
    
    categories = [column for column in df.columns if str(df[column].dtypes) in ['object', 'category', 'bool']]
    numbers = [column for column in df.columns if str(df[column].dtypes) in ['float64', 'int64']]
    
    num_but_cat = [column for column in numbers if df[column].nunique() < cat_th]
    cat_but_car = [column for column in categories if df[column].nunique() > car_th]
    
    categories = categories + num_but_cat
    numbers = [column for column in numbers if column not in num_but_cat]
    categories = [column for column in categories if column not in cat_but_car]
    
    return categories, numbers, cat_but_car
        


def rare_analyser(df, target, cat_cols):
    
    for column in cat_cols:
        print(column, ":", len(df[column].value_counts()))
        print(pd.DataFrame({"COUNT": df[column].value_counts(),
                            "RATIO": df[column].value_counts()/ len(df),
                            "TARGET MEAN": df.groupby(column)[target].mean()}))




def rare_encoder(df, cat_cols, rare_th = 0.01):
    
    df_copy = df.copy()
    
    rare_cols = [column for column in df.columns if df[column].dtypes == 'O' and (df[column].value_counts() / len(df) < rare_th).any(axis = None)]


    for var in rare_cols:
        temp = df_copy[var].value_counts() / len(df_copy)
        rare_labels = temp[temp < rare_th].index
        df_copy[var] = np.where(df_copy[var].isin(rare_labels), 'Rare', df_copy[var])


    return df_copy


def one_hot_encoder(df, cat_cols, drop_first = True):
    dataframe = pd.get_dummies(df, columns = cat_cols, drop_first= drop_first)
    return dataframe




if __name__ == '__main__':
    df = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week3\\datasets\\titanic.csv")
    #print(df.head())
    
    df.columns = [column.upper() for column in df.columns]
    
    #NEW COLUMNS
    df['NEW_CABIN_BOOL'] = df['CABIN'].notnull().astype('int64')
    
    df['NEW_NAME_COUNT'] = df['NAME'].str.len()
    
    df['NEW_NAME_WORD_COUNT'] = df['NAME'].apply(lambda x: len(str(x).split(" ")))
    
    df['NEW_NAME_DR'] = df['NAME'].apply(lambda x: len([x for x in x.split() if x.startswith('Dr')]))
    
    df['NEW_TITLE'] = df['NAME'].str.extract(' ([A-Za-z]+)\.', expand = False)
    
    df['NEW_FAMILY_SIZE'] = df['PARCH'] + df['SIBSP'] + 1
    
    df['NEW_AGE_PCLASS'] = df['AGE'] * df['PCLASS']
    
    df.loc[((df['PARCH'] + df['SIBSP']) > 0), "NEW_IS_ALONE"] = "NO"
    
    df.loc[((df['PARCH'] + df['SIBSP']) ==  0), "NEW_IS_ALONE"] = "YES"
    
    df.loc[(df['AGE'] < 18), "NEW_AGE_CAT"] = "young"
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 50), "NEW_AGE_CAT"] = "mature"
    df.loc[(df['AGE'] >= 50), "NEW_AGE_CAT"] ="senior"
    
    df.loc[((df['AGE'] < 18) & (df['SEX'] == 'male')), "NEW_SEX_CAT"] = "youngmale"
    df.loc[((df['AGE'] >= 18) & (df['AGE'] < 50) & (df['SEX'] == "male")), "NEW_SEX_CAT"] = "maturemale"
    df.loc[((df['AGE'] >= 50) & (df['SEX'] == 'male')), "NEW_SEX_CAT"] = "seniormale"
    
    df.loc[((df['AGE'] < 18) & (df['SEX'] == 'female')), "NEW_SEX_CAT"] = "youngfemale"
    df.loc[((df['AGE'] >= 18) & (df['AGE'] < 50) & (df['SEX'] == "female")), "NEW_SEX_CAT"] = "maturefemale"
    df.loc[((df['AGE'] >= 50) & (df['SEX'] == 'female')), "NEW_SEX_CAT"] = "seniorfemale"
    
    category, number, cardinal = get_columns(df)
    
    #REMOVING UNNECESARRY COLUMNS
    number = [column for column in number if column != 'PASSENGERID']
    remove_cols = ["TICKET", "NAME", "CABIN"]
    df.drop(remove_cols, inplace = True, axis = 1)
    for column in number:
        replace_with_thresholds(df, column)
    check_outlier(df, number)
    missing_values_table(df)
    
    df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
    missing_values_table(df)
    
    df['NEW_AGE_PCLASS'] = df['AGE'] * df['PCLASS']
    df.loc[(df['AGE'] < 18), "NEW_AGE_CAT"] = "young"
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 50), "NEW_AGE_CAT"] = "mature"
    df.loc[(df['AGE'] >= 50), "NEW_AGE_CAT"] ="senior"
    
    df.loc[((df['AGE'] < 18) & (df['SEX'] == 'male')), "NEW_SEX_CAT"] = "youngmale"
    df.loc[((df['AGE'] >= 18) & (df['AGE'] < 50) & (df['SEX'] == "male")), "NEW_SEX_CAT"] = "maturemale"
    df.loc[((df['AGE'] >= 50) & (df['SEX'] == 'male')), "NEW_SEX_CAT"] ="seniormale"
    
    df.loc[((df['AGE'] < 18) & (df['SEX'] == 'female')), "NEW_SEX_CAT"] = "youngfemale"
    df.loc[(df['AGE'] > 18) & (df['AGE'] < 50) & (df['SEX'] == "female"), "NEW_SEX_CAT"] = "maturefemale"
    df.loc[(df['AGE'] >= 50) & (df['SEX'] == 'female'), "NEW_SEX_CAT"] ="seniorfemale"
    
    
    
    df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'O' and len(x.unique()) <= 10) else x)
    #print(df["NEW_AGE_CAT"].isnull().any())
    
    #print(df.loc[df[df["NEW_AGE_CAT"].isnull() == True].index])
    missing_values_table(df)
    df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'O' and len(x.unique()) <= 10) else x)
    missing_values_table(df)
    
    binary_cols = [col for col in df.columns if str(df[col].dtypes) not in ['int64', 'float64']
                   and df[col].nunique() == 2]
    #print(binary_cols)
    
    for col in binary_cols:
        df = label_encoding(df, col)
    #print("PASSENGERID" in category)
    rare_analyser(df, "SURVIVED", category)
    df = rare_encoder(df, category, 0.01)
    rare_analyser(df, "SURVIVED", category)
    
    
    ohe_cols = [column for column in df.columns if 10 >= df[column].nunique() > 2]
    #print(ohe_cols)
    dff = one_hot_encoder(df, ohe_cols)
    #print(dff.shape)
    category, number, cardinal = get_columns(dff)
    number = [column for column in number if column != 'PASSENGERID']
    
    rare_analyser(dff, 'SURVIVED', category)
    
    
    useless_cols = [column for column in dff.columns if dff[column].nunique() == 2 and (dff[column].value_counts() / len(dff) < 0.01).any(axis = None)]
    #print(useless_cols)
    
    
    scaler = StandardScaler()
    dff[number] = scaler.fit_transform(dff[number])
    dff.drop(['PASSENGERID'], axis = 1, inplace = True)
    #print(dff.head())
    #print(dff.shape)
    
    y = dff["SURVIVED"]
    x = dff.drop(["SURVIVED"], axis = 1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 17) 
    
    rf_model = RandomForestClassifier(random_state= 46).fit(x_train, y_train)
    y_pred = rf_model.predict(x_test)
    
    print(accuracy_score(y_pred, y_test))