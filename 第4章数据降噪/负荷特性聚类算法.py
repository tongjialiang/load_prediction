#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import math
from sklearn.cluster import KMeans
import os
import shutil
import datetime
import time
import numpy as np

import pandas as pd
import scipy.stats
import pickle
import pprint
import calendar
import GetClusteringXandBusname_long
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn import mixture
from scipy.cluster.hierarchy import linkage
from sklearn import preprocessing
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import Build_sample_for_kmeans_classification as bu


start =time.clock()
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
#画点
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
#画聚类中心
def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=50, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=5, linewidths=10,
                color=cross_color, zorder=11, alpha=1)
#画决策边界
def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
#np.r_：是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
#np.c_：是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")#画等高线，填充
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')#画等高线，边界
#通过extent参数设置图形的坐标范围[xmin, xmax, ymin, ymax]
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')
#读取公司名称：均值，方差的字典，已去除异常公司
x,busname=bu.do_GetClusteringXandBusname_long()
x=preprocessing.scale(x)#归一化

# print(x.shape)
# print(len(x))
# print(x[:10])
# print(busname.shape)
# print(len(busname))
# print(busname[:10])
#存放调参结果
df_res = pd.DataFrame(columns = ['聚类个数','轮廓系数','每一类的数据个数','初始化个数','迭代次数','分类无效的公司'])

# n_clusters=5
# n_init = 50
# max_iter=50
def do_kmeans(n_clusters,n_init,max_iter):
    global df_res
    kmeans_iter1 = KMeans(n_clusters,'k-means++',n_init,max_iter)
    #init = 'random'初始化中心点的方法，n_init做几次初始化，取最优的模型 max_iter迭代次数
    kmeans_iter1.fit(x)
    #打印中心点
    print("打印中心点")
    print(kmeans_iter1.cluster_centers_)
    print("打印标签")
    #打印标签
    print(kmeans_iter1.labels_)
    #聚类类型，每一类有多少数据
    kmeans_count=np.unique(kmeans_iter1.labels_,return_counts=True)
    #计算无效公司数
    invalid_data=sum(kmeans_count[1][kmeans_count[1]<=10])


    score=kmeans_iter1.score(x)
    print("打印轮廓系数")
    #打印轮廓系数
    score2=silhouette_score(x, kmeans_iter1.labels_)
    print(score2)
    #无效分类的公司(该公司所属类别中所含的公司数量小于10)
    df_res=df_res.append([{'聚类个数':n_clusters,'轮廓系数':score2,'每一类的数据个数':kmeans_count[1],
                    '初始化个数':n_init,'迭代次数':max_iter,'分类无效的公司':invalid_data}])
#


#调参n_init
# for i in range(1,1000,10):
#     do_kmeans(n_clusters=3,n_init=i,max_iter=500)
# df_res.to_csv(path_or_buf='D:\\聚类kmeans调参n_init结果.csv', encoding="utf_8_sig",index=False)

#调参max_iter
# for i in range(1,1000,10):
#     do_kmeans(n_clusters=3,n_init=100,max_iter=i)
# df_res.to_csv(path_or_buf='D:\\聚类kmeans调参max_iter结果.csv', encoding="utf_8_sig",index=False)

#调参n_clusters(看统计表)
# for i in range(2,300,1):
#     do_kmeans(n_clusters=i,n_init=100,max_iter=100)
# df_res.to_csv(path_or_buf='D:\\聚类kmeans调参n_clusters结果(统计表).csv', encoding="utf_8_sig",index=False)

#调参n_clusters(画决策边界)
# for i in range(2,101,1):
#     plt.figure(figsize=(12, 8))
#     KMeans_temp=KMeans(n_clusters=i,n_init=100,max_iter=100)
#     KMeans_temp.fit(x)
#     plot_decision_boundaries(KMeans_temp, x, show_xlabels=False, show_ylabels=False)
#     plt.title('n_clusters='+str(i))
#     plt.xlabel("$mean$", fontsize=14)#x坐标显示的内容
#     plt.ylabel("$std$", fontsize=14, rotation=0)
#     plt.savefig("d:\\kmeans决策边界\\kmeans_classification_n_clusters="+str(i))
# df_res.to_csv(path_or_buf='D:\\聚类kmeans调参n_clusters结果.csv', encoding="utf_8_sig",index=False)


# 最优参数为：n_clusters=27,n_init=100,max_iter=100
# 以下，处理最优模型，以字典的形式保存，键为类别名称，值为该类别中的公司名称
KMeans_best=KMeans(n_clusters=25,init='k-means++',n_init=100,max_iter=100)
KMeans_best.fit(x)
print(KMeans_best.cluster_centers_)
plot_decision_boundaries(KMeans_best, x, show_xlabels=False, show_ylabels=False)

print(KMeans_best.labels_)
kmeans_count=np.unique(KMeans_best.labels_,return_counts=True)
print(kmeans_count)
score=KMeans_best.score(x)
score2=silhouette_score(x, KMeans_best.labels_)
print(score2)
#
res_kmeans={"class1":busname[KMeans_best.labels_==0],
           "class2":busname[KMeans_best.labels_==1],
           "class3":busname[KMeans_best.labels_==2],
           "class4":busname[KMeans_best.labels_==3],
           "class5":busname[KMeans_best.labels_==4],
           "class6":busname[KMeans_best.labels_==5],
           "class7":busname[KMeans_best.labels_==6],
           "class8":busname[KMeans_best.labels_==7],
            "class9":busname[KMeans_best.labels_==8],
            "class10":busname[KMeans_best.labels_==9],
            "class11":busname[KMeans_best.labels_==10],
            "class12":busname[KMeans_best.labels_==11],
            "class13":busname[KMeans_best.labels_==12],
            "class14":busname[KMeans_best.labels_==13],
            "class15":busname[KMeans_best.labels_==14],
            "class16":busname[KMeans_best.labels_==15],
            "class17":busname[KMeans_best.labels_==16],
            "class18":busname[KMeans_best.labels_==17],
            "class19":busname[KMeans_best.labels_==18],
            "class20":busname[KMeans_best.labels_==19],
            "class21":busname[KMeans_best.labels_==20],
            "class22":busname[KMeans_best.labels_==21],
            "class23":busname[KMeans_best.labels_==22],
            "class24": busname[KMeans_best.labels_ == 23],
            "class25": busname[KMeans_best.labels_ == 24]
 }
plt.savefig("C:\\实验记录\\聚类kmeans对数据集做分类-标准化\\kmeans决策边界-微调\\kmeans_classification_n_clusters="+str(13))
plt.show()
1/0
with open('d:/kmeans_for_classification_c3.pk', 'wb+') as f:
    pickle.dump(res_kmeans, f)
with open('d:/KMeans_best_c3.pk', 'wb+') as f:
    pickle.dump(KMeans_best, f)

f.close()














