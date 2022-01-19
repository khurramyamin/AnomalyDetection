# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 23:56:09 2021

@author: Khurram Yamin
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.cluster import KMeans
from numpy import linalg as LA
from matplotlib import pyplot as plt

import scipy.stats as st
from scipy.stats import chi2
import statsmodels.formula.api as sm
import itertools

'''
df = pd.read_csv('clean_argentina.csv', engine='python')
print(df.columns)
new = df[df["CODIGO.PROVINCIA"]==23]

scaler = MinMaxScaler() 
scaled_values = StandardScaler().fit_transform(new) 
new.loc[:,:] = scaled_values

filt = new[["nbi", 
            "masculinidad", "extranjeros", "analfabetismo", "no_usa_pc", 
            "menor_15", "mayor_65", "desocupados", 
            "universitarios", "per_propietario", "per_urban","p.votes.FPV", "p.votes.Cam", 
             "p.votes.UNA", "p.votes.other"]]


indp = new[["p.votes.FPV", "p.votes.Cam", 
             "p.votes.UNA", "p.votes.other"]]
dep = new[["nbi", 
            "masculinidad", "extranjeros", "analfabetismo", "no_usa_pc", 
            "menor_15", "mayor_65", "desocupados", 
            "universitarios", "per_propietario", "per_urban"]]

'''

df = pd.read_csv('clean_argentina.csv', engine='python')

print(df.columns)
scaler = MinMaxScaler() 
scaled_values = StandardScaler().fit_transform(df) 
df.loc[:,:] = scaled_values


Y = df[['p_votes_fpv', 'p_votes_cam','p_votes_una']]


new_total = df[["nbi",  "analfabetismo", "no_usa_pc",  "desocupados", 
            "universitarios", "per_propietario", "per_urban"]]
X = new_total.to_numpy()
total=df




sumVarianceList = []
cov = np.cov(Y)
det = LA.det(cov)
norm = LA.norm(Y,2)
y_std = ((.5**2)* Y.shape[1])**.5



for i in range(15,16):
    breaker = False
    db = KMeans(n_clusters=i, random_state=0).fit(X)
    db_pred = KMeans(n_clusters=i, random_state=0).fit_predict(X)
    print( db.labels_)
    
    num_clust = i
    
    fig, ax = plt.subplots()

    
    plt.title('Estimated number of clusters: %d' % i)
    

    scatter = ax.scatter(X[:, 4], X[:, 5], c=db_pred)
    
   
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Classes")
 
    
    
    plt.xlabel("lattitude")
    plt.ylabel("longitude")
    plt.show()
    
    
    
    
    
    label = db.labels_
    total["labels"] = db.labels_
    new_total["labels"] = db.labels_
    
    
    varianceList = []
    scaledVarianceList = []

    for j in range(num_clust):
        A = total[total["labels"]==j]
        B = A[['p_votes_fpv', 'p_votes_cam','p_votes_una']]
        C = B.to_numpy()
        
        if (C.shape[0] < 3):
            breaker = True
            break
        D = A.to_numpy()
        
        db = DBSCAN(eps=y_std, min_samples=34,p=2).fit(C)
        print( db.labels_)
        print(db.get_params(deep=True))
        
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        if j==0:
            B["innerLabels"] = labels
            new_var = B[B["innerLabels"]==-1]
        else:
            B["innerLabels"] = labels
            newer_var = B[B["innerLabels"]==-1]
            new_var = pd.concat([new_var,newer_var])
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
        
            class_member_mask = (labels == k)
        
            xy = D[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 9], xy[:, 12], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
        
            xy = D[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 9], xy[:, 12], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
        
        mad = np.mean(np.absolute(C - np.mean(C, None)), None)
        varianceList.append(mad)
        scaledVarianceList.append(mad*C.shape[0])
        
        
        plt.title('Cluster Number: %d' % i)
        
        #plt.xlabel("percent Trump Absentee")
        #plt.ylabel("percent Trump Election Day")
        
        
        plt.xlim(-2.5, 2.5)
        plt.ylim(-3,3)
        plt.show()
        plt.close()    
        
    if (breaker == False):
        sumVariance = sum(scaledVarianceList)
        length  = len(total[total["labels"]==-1])
        sumVarianceList.append([num_clust,sumVariance])


num_clust= [x[0] for x in sumVarianceList ]
total_variance = [x[1] for x in sumVarianceList ]



#combined = list(zip(variances,outliers))
#combined.sort(key=lambda tup: tup[0])

plt.title("Minimization of total absolute deviation")
plt.xlabel("k-value")
plt.ylabel("total absolute deviation")
        
plt.plot(num_clust,total_variance)











