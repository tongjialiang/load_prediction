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
#from ShowapiRequest import ShowapiRequest
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
res_rt={}#存放rt数据的字典
res_st={}#存放st数据的字典

with open('D:\\实验记录\\重要结果文件\\pk\\busid_RTTS_and_busid_STTS_Norm_所有长度序列_Fourier.pk', 'rb') as f:
    data = pickle.load(f)
f.close()
type_new=["采矿业","电力热力燃气及水生产和供应业","金融业与房地产业与租赁和商务服务业","建筑业","交通运输仓储和邮政业",
"教育","居民服务修理和其他服务业","科学研究和技术服务业与信息传输软件与信息技术服务业","农林牧渔业","批发和零售业"
    ,"水利环境和公共设施管理业","卫生和社会工作与公共管理社会保障和社会组织","文化体育和娱乐业"
    ,"制造业_机械电子制造业","制造业_轻纺工业","制造业_资源加工工业","住宿和餐饮业"]
#读企业信息查询表
qyxx_root="D:\\实验记录\\重要结果文件\\企业信息查询表.csv"
try:
    qyxx = pd.read_csv(qyxx_root, encoding='utf-8', sep=',')
except:
    qyxx = pd.read_csv(qyxx_root, encoding='gbk', sep=',')
#print(qyxx)
for bustype_name in type_new:
    all_id=qyxx[qyxx["所属行业"]==bustype_name]["id"].values#遍历行业，取得属于这个行业的所有id
    # print(all_id)
    # 1/0
    ################################################处理RT##############################################################
    #拼接rt大列表
    busID_list=[]  #存储企业名称
    rt_large_list=[]#存储拼接后的序列
    dict_rt_name_class={}
    gc.collect()
    for i in data[0]:
        if i in all_id:
            busID_list.append(i)
            rt_large_list.append(data[0][i][:96*10].tolist())
    #打印busID_list和rt_large_list的维度
    print(len(busID_list))  #56
    print(len(rt_large_list))#56
    #print(busID_list)  #9581
    #print(rt_large_list)#9581

    rt_bis = to_time_series_dataset(rt_large_list)
    del rt_large_list
    gc.collect()
    km_dba = TimeSeriesKMeans(n_clusters=2, metric="dtw",n_jobs=-1).fit(rt_bis)#,max_iter_barycenter=5
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
        dict_rt_name_class[name]="rtclass"+bustype_name+str(val)
    #print(dict_rt_name_class)
    res_rt.update(dict_rt_name_class)
    #res_rt.update({"aaa":222})
    # print(res_rt)
    # 1/0
    # #构建rt结果集
    # res_rt=[dict_rt_name_class,km_dba.labels_,kmeans_count,invalid_data,centers]

    print("RT已经处理完毕")
    ################################################处理ST##############################################################
    #拼接st大列表
    busID_list2=[]  #存储企业名称
    st_large_list=[]#存储拼接后的序列
    dict_st_name_class={}#存储聚类结果
    gc.collect()
    for i in data[1]:
        if i in all_id:
            busID_list2.append(i)
            st_large_list.append(data[1][i][:].tolist())
    #打印busID_list和rt_large_list的维度
    print(len(busID_list2))  #9258
    print(len(st_large_list))#9258
    st_bis = to_time_series_dataset(st_large_list)
    # del data
    del st_large_list
    gc.collect()
    km_dba2 = TimeSeriesKMeans(n_clusters=2, metric="dtw",n_jobs=-1).fit(st_bis)#,max_iter_barycenter=5
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
        dict_st_name_class[name]="stclass"+bustype_name+str(val)
    #print(dict_st_name_class)
    res_st.update(dict_st_name_class)
    # print(res_st)
    # 1/0
    #构建st结果集
    #res_st=[dict_st_name_class,km_dba2.labels_,kmeans_count2,invalid_data,centers]
with open('D:\\实验记录\\重要结果文件\\pk\\dtw聚类结果_rt_在行业划分的基础上_频域分解后分2类.pk', 'wb+') as f:
    pickle.dump(res_rt, f)
f.close()

with open('D:\\实验记录\\重要结果文件\\pk\\dtw聚类结果_st_在行业划分的基础上_频域分解后分2类.pk', 'wb+') as f:
    pickle.dump(res_st, f)
f.close()

try:
    end =time.clock()
except:
    end =time.perf_counter()
print('Running time: %s Minutes'%((end-start)/60))