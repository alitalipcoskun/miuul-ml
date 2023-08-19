# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 12:47:06 2023

@author: ali_t
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
pd.set_option('display.float_format', lambda x: '%.2f' % x)
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot = True, fmt = ".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title("Accuracy score is {0}".format(acc), size = 10)
    plt.show()



def get_columns(df, cat_th = 20, car_th = 20):
    
    #Seperating categoric columns and numeric columns
    categories = [column for column in df.columns if str(df[column].dtypes) in ['object', 'category', 'bool']]
    numbers = [column for column in df.columns if (df[column].dtypes) in ['int64', 'float64']]
    
    
    #Finding the outlier columns for number and category
    num_but_cats = [column for column in numbers if df[column].nunique() < cat_th]
    cat_but_car = [column for column in categories if df[column].nunique() > car_th]
    
    #concatelating the all categories with outlier numbers
    categories = categories + num_but_cats
    
    #Getting rid of cardinals in categories
    categories = [column for column in categories if column not in cat_but_car]
    
    #Getting rid of categories in numbers
    numbers = [column for column in numbers if column not in num_but_cats]
    
    #PRINTING THE INFORMATION ABOUT DATAFRAME
    print(f"Dataframe has {len(df.columns)} columns.")
    print("#############################################################")
    print(f"{len(numbers)} of them are numberic columns.")
    print("#############################################################")
    print(f"{len(categories)} of them are categoric columns")
    print("#############################################################")
    print(f"{len(cat_but_car)} of them are cardinal columns")
    print("#############################################################")
    
    
    return categories, numbers, cat_but_car
    


def outlier(df, column, q1 = 0.25, q3 = 0.75, graph = False):
    quantile3 = df[column].quantile(q3)
    quantile1 = df[column].quantile(q1)

    iqr = quantile3 - quantile1
    up = quantile3 + 1.5*iqr
    down = quantile1 - 1.5 * iqr
    
    return up, down
    if graph:
        plt.set_title(column)
        plt.boxplot(df[column])
        plt.show()        


def find_outlier(df, column, graph = False):
    top, bottom = outlier(df, column, graph= graph)
    
    if(df[(df[column] > top) | (df[column] < bottom)]).any(axis = None):
       return True
    else:
        return False


def check_nan_values(df):
    nulls = [column for column in df.columns if df[column].isnull().sum() > 0]
    
    na_vals = []
    for column in nulls:
        nans = df.loc[df[column].isnull(), column].index
        na_vals.append(nans)
    return nulls, na_vals


def change_outliers(df, column):
    top, bottom = outlier(df, column, q1= 0.05, q3 = 0.95)
    
    df.loc[df[column] > top, column] = top
    df.loc[df[column] < bottom, column] = bottom


def target_summary_with_num(df, target, column):
    print(df.groupby(target).agg({column: ['mean']}))
    
    
    

def target_summary(df, target, numbers):
    for column in numbers:
        target_summary_with_num(df, target, column)


def fill_zeros_with_null(df, column):
    df.loc[df[column] == 0, column] = np.nan


def one_hot_encoder(df, cat_cols):
    dataframe = pd.get_dummies(df[cat_cols], drop_first= True)
    return dataframe


def rare_analyser(df, target, cat_cols):
    
    for column in cat_cols:
        print(column, ':', len(df[column].value_counts()))
        print(pd.DataFrame({'COUNT': df[column].value_counts(),
                            'FREQ': df[column].value_counts() / len(df),
                            'TARGET_MEAN': df.groupby(column)[target].mean()}))


def rare_encoder(df, categories, rare_th = 0.01):
    dataframe = df.copy()
    
    rare_cols = [column for column in dataframe.columns if dataframe[column].dtypes == 'O' and (dataframe[column].value_counts() / len(dataframe) < rare_th).any(axis = None)]

    for column in rare_cols:
        temp = dataframe[column].value_counts() / len(dataframe)
        rare_labels = temp[temp < rare_th].index
        dataframe[column] = np.where(dataframe[column].isin(rare_labels), 'Rare', dataframe[column])
    
    return dataframe

def number_to_object(df, cat_cols):
    cats = {column: 'O' for column in cat_cols}
    dff = df.astype(cats)
    return dff



def one_hot_encoder(df, cat_cols, drop_first = True):
    dataframe = pd.get_dummies(df, columns = cat_cols, drop_first= drop_first)
    return dataframe


if __name__ == '__main__':
    imputer = KNNImputer(n_neighbors= 5, weights='distance')
    scaler = RobustScaler()
    
    df = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week4\\diabetes.csv")
    print(df.info())
    print(df.head())
    df.columns = [column.upper() for column in df.columns]
    categories, numbers, cardinals = get_columns(df)
    
    for column in numbers:
        print(column, find_outlier(df, column))
    check_nan_values(df)
    print(df["OUTCOME"].value_counts())
    print(df.describe().T)
    cols = [column for column in numbers if column not in ['PREGNANCIES']]
    
    """
    for column in cols:
        fill_zeros_with_null(df, column)
    """
    
    target_summary(df, 'OUTCOME', numbers)
    df.loc[df['AGE'] < 35, 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 35) & (df['AGE'] < 45), 'NEW_AGE_CAT'] = 'mature'
    df.loc[df['AGE'] >= 45, 'NEW_AGE_CAT'] = 'risk'
    df.loc[df['BMI'] < 18.5, 'NEW_WEIGHT_CAT'] = 'underweight'
    df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 25), 'NEW_WEIGHT_CAT'] = 'healthyweight'
    df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30), 'NEW_WEIGHT_CAT'] = 'overweight'
    df.loc[(df['BMI'] >= 30) & (df['BMI'] < 35), 'NEW_WEIGHT_CAT'] = 'c1_obesity'
    df.loc[(df['BMI'] >= 35) & (df['BMI'] < 40), 'NEW_WEIGHT_CAT'] = 'c2_obesity'
    df.loc[(df['BMI'] >= 40), 'NEW_WEIGHT_CAT'] = 'c3_obesity'
    df.loc[(df['PREGNANCIES'] >= 7), 'PREGNANCIES'] = '7+'

    categories, numbers, cardinals = get_columns(df)
    categories = [column for column in categories if column != 'OUTCOME']

    rare_analyser(df, 'OUTCOME', categories)

    #df = number_to_object(df, categories)
    print(df.head())
    categories, numbers, cardinals = get_columns(df)
    dff = rare_encoder(df, categories, rare_th = 0.02)
    dff = one_hot_encoder(dff, categories)

    rare_analyser(df, 'OUTCOME', categories)
    dfff = dff.copy()

    dfff[dff.columns] = scaler.fit_transform(dff)

    

    
    """
    nulls, indexes = check_nan_values(dfff)
    
    print(nulls)
    print(indexes)
    
    dff[nulls] = imputer.fit_transform(dff[nulls])
    print(dff.info())
    """
    x = dfff.drop(['OUTCOME_1'], axis = 1)
    y = dfff['OUTCOME_1']
    #x_train, x_test, y_train, y_test = train_test_split(x,  y, test_size= 0.2, random_state= 17)
    
    
    knn = KNeighborsClassifier()
    knn.fit(x, y)
    random_user = x.sample(1, random_state = 45)
    print(random_user)
    print(knn.predict(random_user))
    y_pred = knn.predict(x)
    plot_confusion_matrix(y, y_pred)
    cross_val_result = cross_validate(knn, x, y, cv = 5, scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
    result_df = pd.DataFrame(cross_val_result)
    print(result_df)
    for column in result_df.columns:
        print(column, result_df[column].mean())
    y_prob = knn.predict_proba(x)[:, 1]
    """
    print(classification_report(y, y_pred))
    print(roc_auc_score(y, y_prob))
    """
    
    """
    log = LogisticRegression()
    log.fit(x, y)
    cross_val_result = cross_validate(log, x, y, cv = 5, scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

    print(cross_val_result['test_f1'].mean())
    """
    print(knn.get_params())
    print(range(2, 50))
    knn_params = {"n_neighbors": range(2, 50), 
                  "weights": ['uniform', 'distance'],
                  "algorithm": ['ball_tree', 'kd_tree', 'brute'],
                  "leaf_size": range(1, 5),
                  "p": range(1, 3),
                  "metric": ['cosine', 'cityblock', 'euclidean', 'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean']}
    print(knn_params)
    knn_temp_params = {"n_neighbors": range(2, 50),}
    """
    knn_gridS_best = GridSearchCV(knn, 
                                  knn_params, 
                                  cv = 5, 
                                  n_jobs = -1, 
                                  verbose = 1).fit(x, y)
    print(knn_gridS_best.best_params_)
    """
    best_paramss = {'algorithm': 'brute', 'leaf_size': 1, 'metric': 'cosine', 'n_neighbors': 42, 'p': 1, 'weights': 'distance'}
    
    knn_final = knn.set_params(**best_paramss)
    cv_results = cross_validate(knn_final, x, y,
                                cv = 5,
                                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
    res_df = pd.DataFrame(cv_results)
    for column in res_df.columns:
        print(column, res_df[column].mean())
    