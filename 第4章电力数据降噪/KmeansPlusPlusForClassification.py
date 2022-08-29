import math
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
import torch
import gc
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.utils import to_time_series_dataset
try:
    start =time.clock()
except:
    start =time.perf_counter()

with open('D:\\实验记录\\pk\\busid_RTTS_and_busid_STTS_Norm_所有长度序列_Fourier.pk', 'rb') as f:
    data = pickle.load(f)
f.close()
################################################处理RT##############################################################
#拼接rt大列表
busID_list=[]  #存储企业名称
rt_large_list=[]#存储拼接后的序列
dict_rt_name_class={}
for i in data[0]:
    busID_list.append(i)
    rt_large_list.append(data[0][i][:96*10].tolist())
#打印busID_list和rt_large_list的维度
print(len(busID_list))  #9581
print(len(rt_large_list))#9581

rt_bis = to_time_series_dataset(rt_large_list)
del rt_large_list
gc.collect()
km_dba = TimeSeriesKMeans(n_clusters=40, metric="dtw",n_jobs=-1).fit(rt_bis)#,max_iter_barycenter=5
centers=km_dba.cluster_centers_  #聚类中心

print(km_dba.labels_.shape)#(3, 9, 1)
print(km_dba.labels_)#[0 0 2 1 0 2]
#聚类类型，每一类有多少数据
kmeans_count=np.unique(km_dba.labels_,return_counts=True)
print(kmeans_count)
#计算无效公司数
invalid_data=sum(kmeans_count[1][kmeans_count[1]<=10])
print(invalid_data)

#构建存放聚类结果的字典，键是公司名，值是类别
for index,i in enumerate(km_dba.labels_):
    name=busID_list[index]#公司名
    val=i+1#属于第几类
    dict_rt_name_class[name]="rtclass_"+str(val)
print(dict_rt_name_class)
#构建rt结果集
res_rt=[dict_rt_name_class,km_dba.labels_,kmeans_count,invalid_data,centers]

with open('D:\\实验记录\\pk\\dtw聚类结果_rt_频域分解后分40类.pk', 'wb+') as f:
    pickle.dump(res_rt, f)
f.close()
print("RT已经处理完毕")
################################################处理ST##############################################################
#拼接st大列表
busID_list2=[]  #存储企业名称
st_large_list=[]#存储拼接后的序列
dict_st_name_class={}
for i in data[1]:
    busID_list2.append(i)
    st_large_list.append(data[1][i][:].tolist())
#打印busID_list和rt_large_list的维度
print(len(busID_list2))  #9258
print(len(st_large_list))#9258

st_bis = to_time_series_dataset(st_large_list)
del data
del st_large_list
gc.collect()
km_dba2 = TimeSeriesKMeans(n_clusters=40, metric="dtw",n_jobs=-1).fit(st_bis)#,max_iter_barycenter=5
centers=km_dba2.cluster_centers_  #聚类中心

print(km_dba2.labels_.shape)#(3, 9, 1)
print(km_dba2.labels_)#[0 0 2 1 0 2]
#聚类类型，每一类有多少数据
kmeans_count2=np.unique(km_dba2.labels_,return_counts=True)
print(kmeans_count2)
#计算无效公司数
invalid_data=sum(kmeans_count2[1][kmeans_count2[1]<=10])
print(invalid_data)

# score2=silhouette_score(st_bis, km_dba2.labels_)
# print(score2)#轮廓系数
#构建存放聚类结果的字典，键是公司名，值是类别
for index,i in enumerate(km_dba2.labels_):
    name=busID_list2[index]#公司名
    val=i+1#属于第几类
    dict_st_name_class[name]="stclass_"+str(val)
print(dict_st_name_class)
#构建st结果集
res_st=[dict_st_name_class,km_dba2.labels_,kmeans_count2,invalid_data,centers]

with open('D:\\实验记录\\pk\\dtw聚类结果_st_频域分解后分40类.pk', 'wb+') as f:
    pickle.dump(res_st, f)
f.close()

try:
    end =time.clock()
except:
    end =time.perf_counter()
print('Running time: %s Minutes'%((end-start)/60))