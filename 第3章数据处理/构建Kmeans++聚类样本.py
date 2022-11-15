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
import tslearn
from tslearn.clustering import TimeSeriesKMeans
from tslearn.generators import random_walks
from tslearn.utils import to_time_series_dataset
count_rt=0#有多少个文件
count_st=0
dict_rt={}
dict_mt_st={}
min_=999999
for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按地域划分数据集_去异常"):
        for i in filelist:
            #filename=root+"\\"+i
            #print(filename)
            if i == 'RT_data.csv':
                count_rt+=1
                #print(i)#RT_data.csv
                business_id=root.split("\\")[-2] + "_" + root.split("\\")[-1]  #萧山区供电分公司5_9
                #print(business_id)
                # print(i)#RT_data.csv
                # print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                # print(business_id)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                # if len(RT_data["瞬时有功(kW)"])>=100:
                #dict_rt[business_id]=RT_data["瞬时有功(kW)"][:1000].to_numpy()
                dict_rt[business_id] = RT_data["瞬时有功(kW)"][:96*21].to_numpy()
                # else:
                #     print("small_rt="+str(len(RT_data["瞬时有功(kW)"])))

            if i == 'ST_data.csv':
                count_st+=1
                #print(i)#RT_data.csv
                business_id=root.split("\\")[-2] + "_" + root.split("\\")[-1]  #萧山区供电分公司5_9
                # print(i)#RT_data.csv
                # print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                #print(business_id)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                #print(business_id)
                #print(ST_data["平均负荷(kW)"][::-1])
                #print(ST_data["平均负荷(kW)"][::-1].to_numpy())
                #stlen= len(ST_data["平均负荷(kW)"][::-1].to_numpy())
                # if stlen<60:
                #     print("small_st="+str(stlen))
                # if len(ST_data["平均负荷(kW)"][::-1])>=60:
                #dict_mt_st[business_id]=ST_data["平均负荷(kW)"][::-1][:400].to_numpy()
                dict_mt_st[business_id]=ST_data["最大负荷(kW)"][::-1][:7*24].to_numpy()

print("count_rt="+str(count_rt))
print("count_st="+str(count_st))

res=[dict_rt,dict_mt_st]

with open('D:\\实验记录\\pk\\busid_RTTS_and_busid_STTS_所有长度序列.pk', 'wb+') as f:
    pickle.dump(res, f)
f.close()
























# X = random_walks(n_ts=50, sz=32, d=1)
# print(X.shape) #(50, 32, 1)
#
# km_dba = TimeSeriesKMeans(n_clusters=3, metric="dtw",random_state=0,n_jobs=-1).fit(X)#,max_iter_barycenter=5
# centers=km_dba.cluster_centers_
# print(centers.shape) #(3, 32, 1)
# print(km_dba.labels_.shape)#(50,)
#
# X_bis = to_time_series_dataset([[1, 2, 3, 4],[1, 2, 3],[2, 5, 6, 7, 8, 9],[4,44,4],[1,2,3],[1,2,3,4,5,6,7,8,9]])
# km_dba2 = TimeSeriesKMeans(n_clusters=3, metric="dtw",random_state=0,n_jobs=-1).fit(X_bis)#,max_iter_barycenter=5
# centers2=km_dba2.cluster_centers_
# print(centers2.shape) #(3, 32, 1)
# print(centers2)
# print(km_dba2.labels_.shape)#(3, 9, 1)
# print(km_dba2.labels_)#[0 0 2 1 0 2]


