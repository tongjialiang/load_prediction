#!/usr/bin/python
# -*- coding: utf-8 -*-
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
dir1=["按地域划分数据集(方案1-汇总标准化)","按地域划分数据集(方案1-汇总标准化)_去异常",
       "按地域划分数据集(方案2-对每个企业标准化)","按地域划分数据集(方案2-对每个企业标准化)_去异常"]
h1=["杭州市区","淳安县","富阳区","海宁市","海盐县","嘉善县",
      "嘉兴市桐乡市","建德市","临安区","平湖市","萧山区","余杭区"]

dir2=["按行业划分数据集(方案1-汇总标准化)","按行业划分数据集(方案1-汇总标准化)_去异常",
       "按行业划分数据集(方案2-对每个企业标准化)","按行业划分数据集(方案2-对每个企业标准化)_去异常"]
h2=["采矿业","电力热力燃气及水生产和供应业","金融业与房地产业与租赁和商务服务业","建筑业","交通运输仓储和邮政业",
"教育","居民服务修理和其他服务业","科学研究和技术服务业与信息传输软件与信息技术服务业","农林牧渔业","批发和零售业"
    ,"水利环境和公共设施管理业","卫生和社会工作与公共管理社会保障和社会组织","文化体育和娱乐业"
    ,"制造业_机械电子制造业","制造业_轻纺工业","制造业_资源加工工业","住宿和餐饮业"]

dir3= ["按聚类划分数据集(方案1-汇总标准化)_c25"]
h3= ["class1", "class2", "class3", "class4", "class5", "class6", "class7", "class8", "class9", "class10",
           "class11", "class12", "class13", "class14", "class15", "class16", "class17", "class18", "class19", "class20",
           "class21", "class22", "class23", "class24", "class25"]
dir4= ["按聚类划分数据集(方案1-汇总标准化)_c3","按聚类划分数据集(方案2-对每个企业标准化)_c3"]
h4= ["class1", "class2", "class3"]

res=[]
for i in dir1:
    for j in h1:
        res.append([i,j])

for i in dir2:
    for j in h2:
        res.append([i,j])

for i in dir3:
    for j in h3:
        res.append([i,j])
for i in dir4:
    for j in h4:
        res.append([i,j])


print(res)#[['按地域划分数据集(方案1-汇总标准化)', '杭州市区'], ['按地域划分数据集(方案1-汇总标准化)', '淳安县'],
#存放分析结果
df_res = pd.DataFrame(columns = ['数据处理方式','地区类别企业','实时数据的正类样本数占比'
    ,'短期数据的正类样本数占比','长期数据的正类样本数占比','备注'])




def do(link1, link2):#['按地域划分数据集(方案1-汇总标准化)', '杭州市区']
    global df_res
    rt_p = 0  # 正
    rt_n = 0
    mt_p = 0
    mt_n = 0
    st_p = 0
    st_n = 0
    rtres=0
    mtres=0
    stres=0
    for root, dirs, filelist in os.walk("d:\\"+link1+"\\"+link2+"\\"):
        for i in filelist:
            if i == 'RT_data.csv':
                print(root)#d:\按地域划分数据集(方案2-对每个企业标准化)\临安区\临安区供电分公司3\141
                try:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                rt_p+=sum(RT_data["瞬时有功(kW)"]>=0)
                rt_n += sum(RT_data["瞬时有功(kW)"]<0)
            if i == 'MT_data.csv':
                #print(i)#RT_data.csv
                #print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    MT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    MT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')

                mt_p+=sum(MT_data["平均负荷(kW)"]>=0)
                mt_n += sum(MT_data["平均负荷(kW)"]<0)
            if i == 'ST_data.csv':
                # print(i)#RT_data.csv
                # print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                st_p += sum(ST_data["平均负荷(kW)"] >= 0)
                st_n += sum(ST_data["平均负荷(kW)"] < 0)
    if rt_n+rt_p==0:
        rtres="无法计算"
    else:
        rtres=rt_p/(rt_n+rt_p)
    if mt_n+mt_p==0:
        mtres="无法计算"
    else:
        mtres=mt_p/(mt_n+mt_p)
    if st_n+st_p==0:
        stres="无法计算"
    else:
        stres=st_p/(st_n+st_p)
    df_res = df_res.append([{'数据处理方式':link1,'地区类别企业':link2,'实时数据的正类样本数占比':rtres,
                            '短期数据的正类样本数占比':stres,'长期数据的正类样本数占比':mtres}])
#执行全部数据
for k in res:#[['按地域划分数据集(方案1-汇总标准化)', '杭州市区'], ['按地域划分数据集(方案1-
    do(k[0], k[1])
#do("按聚类划分数据集(方案1-汇总标准化)_去异常", "class15")



df_res.to_csv(path_or_buf='D:\\分析样本不平衡性的改善.csv', encoding="utf_8_sig",index=False)