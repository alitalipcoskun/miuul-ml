# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:04:45 2023

@author: ali_t
"""

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

pd.set_option('display.max_columns', None)
warnings.simplefilter(action = 'ignore', category =Warning)


def get_columns(dataframe, cat_th = 20, car_th = 20):
    
    categories = [column for column in dataframe.columns if str(dataframe[column].dtypes) in ['category', 'object', 'bool']]
    numbers = [column for column in dataframe.columns if str(dataframe[column].dtypes) in ['int64', 'float64']]
    
    cat_but_car = [column for column in categories if dataframe[column].nunique() > car_th]
    num_but_cat = [column for column in numbers if dataframe[column].nunique() < cat_th]
    
    
    categories = categories + num_but_cat
    
    categories = [column for column in categories if column not in cat_but_car]
    numbers = [column for column in numbers if column not in num_but_cat]
    
    
    print("DATAFRAME HAS {0} columns".format(len(dataframe.columns)))
    print("###########################################################")
    print("DATAFRAME HAS {} NUMBERIC columns".format(len(numbers)))
    print("###########################################################")
    print("DATAFRAME HAS {} CATEGORIC columns".format(len(categories)))
    print("###########################################################")
    print("DATAFRAME HAS {} CARDINAL columns".format(len(cat_but_car)))
    print("###########################################################")            


    return categories, numbers, cat_but_car


def one_hot_encoder(df, cat_cols):
    dataframe = pd.get_dummies(df[cat_cols], drop_first= True)
    return dataframe
    





def print_NaN_values(dataframe):
    for column in dataframe.columns:
        print(column, dataframe[column].isnull().sum())


def change_zero_to_nan(dataframe, num_cols):
    
    for column in num_cols:
        df.loc[df[column] == 0, column] = np.nan


def rare_analyzer(df, target, column):
    
    print(column, ':', len(df[column].value_counts()))
    print(pd.DataFrame({"COUNT": df[column].value_counts(),
                        "FREQ": df[column].value_counts()/len(df),
                        "TARGET_MEAN": df.groupby(column)[target].mean()}))

    
def give_NaN_indexes(dataframe):
    
    columnIndexDict = {}
    for column in dataframe.columns:
        if dataframe[column].isnull().values.any():
            columnIndexDict[column] = dataframe[dataframe[column].isnull()].index
        
    return columnIndexDict


def outlier_tresholds(df, column, q1 = 0.25, q3 = 0.75):
    quantile1 = df[column].quantile(q1)
    quantile3 = df[column].quantile(q3)
    
    iqr = quantile3 - quantile1
    
    upper = quantile3 + 1.5 *iqr
    lower = quantile1 - 1.5*iqr
    
    return upper, lower

def change_outlier(df, column, q1 = 0.25, q3 = 0.75):
    up, low = outlier_tresholds(df, column, q1 = q1, q3 = q3)
    
    df.loc[df[column] > up, column] = up
    df.loc[df[column] < low, column] = low



def plot_importance(model, features):
    num = len(features)
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    
    plt.figure(figsize=(10 ,10))
    sns.set(font_scale= 1)
    sns.barplot(x = "Value", y= "Feature", data = feature_imp.sort_values(by ="Value", ascending = False)[0:num])
    
    plt.title('Features')
    plt.tight_layout()

    plt.show()
if __name__ == '__main__':
    # DATA ANALYSIS
    scaler = RobustScaler()
    #minmax = Mi
    clf = LocalOutlierFactor(n_neighbors= 20)
    df = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week4\\diabetes.csv")
    #print(df.head())
    df.columns = [column.upper() for column in df.columns]
    categories, numbers, cardinals = get_columns(df)
    #print(numbers)
    change_zero_to_nan(df, numbers)
    print_NaN_values(df)
    imputer = KNNImputer(n_neighbors=5)
    data = imputer.fit_transform(df)
    df_new = pd.DataFrame(data, columns = df.columns)
    #print_NaN_values(df_new)
    for column in df_new.columns:
      change_outlier(df_new, column, q1 = 0.05, q3 = 0.95)
    
    """
    for column in df.columns:
        plt.boxplot(df[column])
        plt.title(column)
        plt.show()
    """
    for column in categories:
        rare_analyzer(df, 'OUTCOME', column)
    
    clf.fit_predict(df_new)
    df_scores = clf.negative_outlier_factor_
    
    #print(np.sort(df_scores)[0:5])
    scores = pd.DataFrame(np.sort(df_scores))
    scores.plot(stacked = True, xlim = [0, 20], style = '.-')
    plt.show()
    th = np.sort(df_scores)[1]
    #print(th)
    outliers = df_new[df_scores <= th]
    #print(outliers)
    #print(df_new.describe([0.01, 0.05, 0.75, 0.90, 0.99]))
    df_new = df_new.drop(axis = 0, labels = df_new[df_scores <= th].index)
    #print(df_new.head())
    
    
    #FEATURE ENGINEERING
    df_new.loc[df_new['PREGNANCIES'] > 0, 'NEW_PREGNANCY_CAT'] = 1
    df_new.loc[df_new['PREGNANCIES'] == 0, 'NEW_PREGNANCY_CAT'] = 0
    
    df_new.loc[df_new['AGE'] < 45, 'NEW_AGE_CAT'] = 'young'
    df_new.loc[(df_new['AGE'] >= 45) & (df_new['AGE'] <= 64), 'NEW_AGE_CAT'] = 'risk'
    df_new.loc[df_new['AGE'] > 64, 'NEW_AGE_CAT'] = 'elder'
    
    df_new.loc[df_new['PREGNANCIES'] >= 7, 'PREGNANCIES'] = '7+'
    
    df_new.loc[df_new['BMI'] <= 18.5, 'NEW_BMI_CAT'] = 'underweight'
    df_new.loc[(df_new['BMI'] > 18.5) & (df_new['BMI'] < 25), 'NEW_BMI_CAT'] = 'normalweight'
    df_new.loc[(df_new['BMI'] >= 25) & (df_new['BMI'] < 30), 'NEW_BMI_CAT'] = 'overweight'
    df_new.loc[(df_new['BMI'] >= 30) & (df_new['BMI'] < 35), 'NEW_BMI_CAT'] = 'c1_obesity'
    df_new.loc[(df_new['BMI'] >= 35) & (df_new['BMI'] < 40), 'NEW_BMI_CAT'] = 'c2_obesity'
    df_new.loc[(df_new['BMI'] >= 40), 'NEW_BMI_CAT'] = 'c3_obesity'
    print(df_new.head())
        
    
    print_NaN_values(df_new)
    categories, numbers, cardinals = get_columns(df_new)
    
    for column in categories:
        rare_analyzer(df_new, 'OUTCOME', column)
    
    df_new = one_hot_encoder(df_new, categories)
    transformed = scaler.fit_transform(df_new)
    
    df_new = pd.DataFrame(transformed, columns = df_new.columns)
    
    
    y = df_new['OUTCOME']
    X = df_new.drop(['OUTCOME'], axis = 1)
    #print(df_new.head())
    
    """
    cart_model = DecisionTreeClassifier(random_state= 1).fit(X, y)
    
    y_pred = cart_model.predict(X)
    y_prob = cart_model.predict_proba(X)[:, 1]
    
    print(classification_report(y, y_pred))
    print(roc_auc_score(y, y_pred))
    """
    
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 17)
    cart_model = DecisionTreeClassifier(random_state= 1)
    cart_model.fit(X_train, y_train)
    y_pred = cart_model.predict(X_test)
    y_prob = cart_model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print(roc_auc_score(y_test, y_prob))
    """
    
    cart_model = DecisionTreeClassifier(random_state= 17)
    """
    cv_results = cross_validate(cart_model,
                                X, y,
                                cv = 10,
                                scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc'])
    
    
    result_df = pd.DataFrame(cv_results)
    for column in result_df.columns:
        print(column, result_df[column].mean())
        
    print(cart_model.get_params())
    """
    
    cart_params = {'max_depth': range(1, 20),
                   'min_samples_split': range(2, 20)}
    
    cart_best_grid = GridSearchCV(cart_model,
                                  cart_params,
                                  scoring = 'f1',
                                  cv = 5,
                                  n_jobs = -1,
                                  verbose = False).fit(X, y)
    
    
    print(cart_best_grid.best_params_)
    print(cart_best_grid.best_score_)
    #cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state= 17).fit(X, y)
    
    cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)
    cv_results = cross_validate(cart_final,
                                X, y,
                                cv = 5,
                                scoring = ['accuracy', 'f1', 'roc_auc', 'recall', "precision"])
    
    
    cv_results = pd.DataFrame(cv_results)
    
    for column in cv_results.columns:
        print(column, cv_results[column].mean())
    
    plot_importance(cart_final, X)