# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:32:46 2023

@author: ali_t
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from sklearn.cluster import AgglomerativeClustering



if __name__ == '__main__':
    
    df = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week4\\hitters.csv")
    cols = [column for column in df.columns if df[column].dtypes != 'O']
    df = df[cols]
    df.dropna(inplace = True)
    df = df.drop(['Salary'], axis = 1)
    print(df.shape)
    print(df.head())
    
    df = StandardScaler().fit_transform(df)
    
    pca = PCA()
    
    pca_fit = pca.fit_transform(df)
    
    cumilative_sum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(cumilative_sum)
    plt.xlabel("Number of component")
    plt.ylabel("Variance ratio")
    plt.show()
    