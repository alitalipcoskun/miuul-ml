# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 20:02:28 2023

@author: ali_t
"""

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)
warnings.simplefilter(action = 'ignore', category= Warning)



if __name__ == '__main__':
    df = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week4\\diabetes.csv")
    y = df['Outcome']
    X = df.drop(['Outcome'], axis = 1)
    
    
    rf = RandomForestClassifier(random_state= 17)
    
    rf_params = {'max_depth': [5, 8, None],
                 'max_features': [3, 5, 7, "auto"],
                 'min_samples_split': [2, 5, 8, 15, 20],
                 'n_estimators': [100, 200, 500]}
    
    cv_results = cross_validate(rf, X, y,
                                cv = 10,
                                scoring = ['accuracy', 'f1', 'roc_auc'])
    
    
    result_df = pd.DataFrame(cv_results)
    
    for column in result_df.columns:
        print(column, result_df[column].mean())
    
    """
    rf_best_grid = GridSearchCV(rf, rf_params, 
                                cv = 5, n_jobs = -1, 
                                verbose = True).fit(X, y)
    
    print(rf_best_grid.best_params_)
    """

    besties =  {'max_depth': None,
                'max_features': 5,
                'min_samples_split': 8,
                'n_estimators': 500}  
    print("Second!")
    rf = rf.set_params(**besties, random_state = 17).fit(X, y)
    
    cv_results = cross_validate(rf, X, y,
                                cv = 10,
                                scoring = ['accuracy', 'f1', 'recall', 'roc_auc'])
    
    result_df = pd.DataFrame(cv_results)
    
    for column in result_df.columns:
        print(column, result_df[column].mean())