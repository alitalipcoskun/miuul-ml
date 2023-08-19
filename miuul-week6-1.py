# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:44:50 2023

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

pd.set_option('display.max_columns', None)
warnings.simplefilter(action = 'ignore', category =Warning)

if __name__ == '__main__':
    df = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week4\\USArrests.csv", index_col= 0)
    #print(df.head())
    #print(df.describe().T)
    
    scaler = MinMaxScaler((0, 1))
    df = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters= 4).fit(df)
    
    """
    kmeans = KMeans(n_clusters= 4, random_state= 17).fit(df)
    print(kmeans.get_params())
    pca = PCA(n_components= 2)
    reducted_data = pca.fit_transform(df)
    #print(reducted_data)
    kmeans= KMeans(n_clusters = 4, random_state = 17).fit(reducted_data)
    print(kmeans.labels_)
    
    for i in range(len(reducted_data)):
        plt.scatter(x = reducted_data[i][0], y  = reducted_data[i][1], cmap = kmeans.labels_[i])
    
    plt.legend([0, 1, 2, 3], ['class 0', 'class 1', 'class 2', 'class 3'])
    
    
    ssd = []
    K = range(1, 30)
    
    for k in K:
        kmeans = KMeans(n_clusters= k).fit(df)
        ssd.append(kmeans.inertia_)
    
    print(ssd)
    plt.plot(K, ssd, "bx-")
    plt.xlabel("K values")
    plt.ylabel("SSD/SSR/SSE")
    plt.title("Elbow Method")
    plt.show()
    """
    
    
    #KMEANSCLUSTERING
    ###########################
    #NOT: UZAKLIK TEMELI OLAN YÖNTEMLERDE STANDARTIZASYON YAPMAK PERFORMANSI ARTTIRIR. Tüm yorum satırı içeren
    #yerlerde esasında standartizasyon işleminden geçmiş olan verileri kullanıyoruz.
    ###########################
    
    kmeans = KMeans()
    
    elbow = KElbowVisualizer(kmeans, k = (2, 20))
    elbow.fit(df)
    elbow.show()
    
    kmeans = KMeans(n_clusters= elbow.elbow_value_).fit(df)
    kmeans_clusters = kmeans.labels_
    
    df = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week4\\USArrests.csv", index_col= 0)
    df['cluster'] = kmeans_clusters + 1
    #print(df.head())
    
    
    hc_average = linkage(df, 'average')
    
    plt.title('Dendogram')
    plt.xlabel('Observations')
    plt.ylabel('Distances')
    
    dendrogram(hc_average,)
               #truncate_mode= "lastp",
               #p = 10,
               #show_contracted = True,
               #leaf_font_size= 10)
    plt.axhline(y = 0.6, color = 'r', linestyle = '--')
    plt.axhline(y = 0.5, color = 'b', linestyle = '--')
    plt.show()
    
    hi_cluster = AgglomerativeClustering(n_clusters = 5, linkage = 'average')
    clusters = hi_cluster.fit_predict(df)
    
    df = pd.read_csv("C:\\Users\\ali_t\\.spyder-py3\\saves\\datasets-week4\\USArrests.csv", index_col= 0)
    df['hi_clusters_no'] =clusters + 1
    df['clusters'] = kmeans_clusters + 1
    print(df.head())