# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:15:45 2023

@author: ali_t
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0
    
    for i in range(0, m):
        y_hat = b + w*X[i]
        y = Y[i]
        
        sse += (y_hat - y) ** 2
        
    
    mse = sse / m
    return mse



def update_weights(Y, b, w, X, lr = 0.1):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    
    for i in range(0, m):
        y_hat = b + w*X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    
    
    new_b = b - (lr * 1 / m * b_deriv_sum)
    new_w = w - (lr * 1 / m * w_deriv_sum)
    
    return new_b, new_w
    



def train(Y, initial_b, initial_w, X, num_iters, lr = 0.1):
    print("STARTING GRADIENT DESCENT AT B = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                            cost_function(Y, initial_b, initial_w, X)))
    
    b = initial_b
    w = initial_w
    
    cost_history = []
    
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, lr)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)
        
        if i % 100 == 0:
            print("iter = {:d} b = {:.2f} w = {:.4f} mse = {:.4}".format(i, b, w, mse))
    
    print("AFTER {0} ITERATIONS b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, X))
    
    return cost_history, b, w


if __name__ == '__main__':
    df = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week4\\advertising.csv")
    print(df.head())
    
    """
    x = df[['TV']]
    y = df[['sales']]
    
    
    #MODEL
    
    
    #y_hat = b + w*x
    #b (bias)
    b = reg_model.intercept_[0]
    
    #w
    w1 = reg_model.coef_[0][0]
    
    
    #150 birimlik TV harcamasında satış beklentimiz...
    ans = b + w1 * 150
    print(ans)
    
    
    g = sns.regplot(x = x, y = y, scatter_kws= {'color': 'b', 's': 9},
                    ci = False, color = 'r')
    g.set_title(f"Model Denklemi: Sales = {np.round(b, 2)} + tv*{np.round(w1, 2)}")
    g.set_ylabel("Satış sayısı")
    g.set_xlabel("TV Harcamaları")
    plt.xlim(-10, 310)
    plt.ylim(bottom = 0)
    plt.show()
    
    y_pred = reg_model.predict(x)
    print(mean_squared_error(y, y_pred))
    """
    x = df.drop('sales', axis = 1)
    y = df[['sales']]
    x = df [['radio']]
    y = df[['sales']]
    X = df['radio']
    Y = df['sales']
    reg_model = LinearRegression().fit(x, y)
    
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 1)
    y_pred = reg_model.predict(x_train)
    print(np.sqrt(mean_squared_error(y_train, y_pred)))
    print(reg_model.score(x_train, y_train))
    print(np.mean(np.sqrt(-cross_val_score(reg_model, 
                                           X = x,
                                           y = y,
                                           cv = 10,
                                           scoring = "neg_mean_squared_error"))))
    
    
    initial_b = 0.001
    initial_w = 0.001
    num_iters = 10000
    
    
    b, W = train(Y, initial_b, initial_w, X, num_iters = 1000)