#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
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


start =time.clock()
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
with open('d:\\dict_BusTypeOrArea_MeanAndStd_method2.pk', 'rb') as f:
    dict_bus = pickle.load(f)
    #print(dict_bus)
#评估器：评估手工注入的异常数据能否都被捕获到？

def evaluate_get_myexceptdata(sample_exception,lable):
    count=0
    exceptedatalist=sample_exception[np.where(lable==-1)]
    for i in exceptedatalist:
        if str(i).split("_")[-2]=='rt异常数据' or str(i).split("_")[-2]=='long异常数据':
            count+=1
    return count==10
#打印发现的异常数据

def print_exceptdata(sample_exception,lable):
    exceptedatalist = sample_exception[np.where(lable == -1)]
    return exceptedatalist
#评估器：模型发现异常数据的个数

def evaluate_num_exceptdata(labels_):
    res=np.unique(labels_, return_counts=True)
    index=np.where(res[0]==-1)
    if len(index[0])==0:
        return 0
    res1=res[1][index][0]
    #print(res1)
    return res1
#评估器：聚类任务的轮廓系数是否大于0.96

def evaluat_score(sample_exception,labels_):
    score=silhouette_score(sample_exception, labels_)
    print(score)
    return score>0.94
#评估器：评估若使用该模型重新划分数据集并归一化后，能改善样本不平衡性的程度。

def evaluate_mean_exception(dict_bus,exdata):
    count_rt = 0
    count_st = 0
    rt=0 #实时均值
    st=0 #长期均值
    if len(exdata)==0:
        return [0,0]
    for i in exdata:
        if i in dict_bus:
            if 'rt_mean' in dict_bus[i]:
                rt+=dict_bus[i]['rt_mean']
                count_rt+=1
            if 'st_mean' in dict_bus[i]:
                st+=dict_bus[i]['st_mean']
                count_st+=1
    # print(rt/count_rt)
    # print(st/count_st)
    if count_rt==0 or count_st==0:
        return [0, 0]
    return [rt/count_rt,st/count_st]
#读取聚类样本文件
with open('d:/exception_detection_samples.pk', 'rb') as f:
    data = pickle.load(f)

rt=data['rt'][0]#(9661, 2) 实时用电数据
rt_business=data['rt'][1]#(9661,) 实时用电数据公司名称
rt = preprocessing.scale(rt)#归一化

long=data['long'][0]#(9661, 2) 长期用电数据
long_business=data['long'][1]#(9661,) 长期用电数据公司名称
long = preprocessing.scale(long)#归一化

rt_exception=data['rt_exception'][0]#(9671, 2) 实时用电数据（最后10条为异常数据）
rt_business_exception=data['rt_exception'][1]#(9671,)实时用电数据公司名称（最后10条为异常数据）
rt_exception = preprocessing.scale(rt_exception)

long_exception=data['long_exception'][0]#(9671, 2)长期用电数据（最后10条为异常数据）
long_business_exception=data['long_exception'][1]#(9671,)长期用电数据公司名称（最后10条为异常数据）
long_exception = preprocessing.scale(long_exception)#归一化

#存放调参结果
df_res = pd.DataFrame(columns = ['eps','min_samples'
    ,'发现所有异常数据','高轮廓系数','改善不平衡性(实时)','改善不平衡性(长期)','各类别数据量'
    ,'离群点个数','异常数据','非核心点个数'])
# print(long_business_exception[-10:])
# print(long_business_exception.shape)
print("打印实时数据的最小值和最大值，以确定eps范围")
print(np.min(rt_exception))#-0.03444010692261343
print(np.max(rt_exception))#77.80098022909692
###########
#eps 0.01-
#########
#np.linspace(0.01, 2.30, 100),np.linspace(5, 10, 6)

#寻找合适的参数
def do_dbscan(eps_ms_list,which_exception,which_business_exception):#long_business_exception,rt_exception
    global df_res
    for i in eps_ms_list:
        eps=i[0]
        min_samples=i[1]
        dbscan = DBSCAN(eps ,min_samples)#半径，最小样本数
        dbscan.fit(which_exception)#long_business_exception
        #print(dbscan.labels_)
        #print(np.unique(dbscan.labels_,return_counts=True))
        test1=evaluate_get_myexceptdata(which_business_exception,dbscan.labels_)
        #print(test1)
        test2=evaluate_num_exceptdata(dbscan.labels_)
        #print(num)
        test3=evaluat_score(which_exception,dbscan.labels_)
        exdata=print_exceptdata(which_business_exception,dbscan.labels_)
        #print(exdata)
        test4=evaluate_mean_exception(dict_bus,exdata)
        #print("核心样本数")
        #print(len(dbscan.core_sample_indices_))
        core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_mask[dbscan.core_sample_indices_] = True  #core_sample_indices_核心样本index
        anomalies_mask = dbscan.labels_ == -1#离群点噪声index
        non_core_mask = ~(core_mask)#边界点与离群点
        boundary_mask = ~(core_mask | anomalies_mask)#边界index
        #存放调参结果
        #print("333333333333333333333")
        print(test4)
        df_res=df_res.append([{'eps':eps,'min_samples':min_samples,'发现所有异常数据':test1,
                        '高轮廓系数':test3,'改善不平衡性(实时)':str(test4[0]),'改善不平衡性(长期)':str(test4[1]),
                        '各类别数据量':str(np.unique(dbscan.labels_, return_counts=True)[1]),
                        '离群点个数': test2,'非核心点个数':sum(non_core_mask),'异常数据':exdata}])

    df_res.to_csv(path_or_buf='D:\\聚类DBscan调参结果_正确参数2.csv', encoding="utf_8_sig",index=False)
    return exdata#异常用电公司




#以下，寻找合适的参数
#eps_list, min_samples_list = np.meshgrid(np.linspace(0.5, 3.16, 100),np.linspace(3, 5, 3))
#eps_list, min_samples_list = np.meshgrid(np.linspace(0.001, 0.23, 200),np.linspace(5, 10, 6))
#eps_ms_list=np.c_[eps_list.ravel(), min_samples_list.ravel()]
#print(eps_ms_list)
#do_dbscan(eps_ms_list)#循环调用dbscan,寻找最好的参数
#do_dbscan(eps_ms_list,long_exception,long_business_exception)
end =time.clock()
print('Running time: %s Minutes'%((end-start)/60))


#打印异常数据
res1=[]
res2=[]
res1=do_dbscan([[1.386,4]],long_exception,long_business_exception)
res2=do_dbscan([[0.06659,9]],rt_exception,rt_business_exception)
res_all=[res2,res1]
with open('d:/exception_company.pk', 'wb+') as f:
    pickle.dump(res_all, f)
f.close()
# Running time: 15.421217316666667 Minutes
#性能优化方案：
#使用多线程
#Running time: 19.149653968333332 Minutes














