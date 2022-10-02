from sklearn.preprocessing import StandardScaler
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
import warnings
import matplotlib
#from ShowapiRequest import ShowapiRequest
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
warnings.filterwarnings("ignore")
# df_res = pd.DataFrame(columns = ['id','公司名称'
#     ,'供电单位','所在地区','所属行业','聚类类型-实时','聚类类型-长期'])
try:
    data = pd.read_csv("D:\\实验记录\\重要结果文件\\企业信息查询表.csv", encoding='utf-8', sep=',')
except:
    data = pd.read_csv("D:\\实验记录\\重要结果文件\\企业信息查询表.csv", encoding='gbk', sep=',')
data['新聚类类型-实时']=0
data['新聚类类型-长期']=0
#写入字段：聚类类型-实时 聚类类型-长期
for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集(行业划分后聚类)\\"):
        for i in filelist:
            if i in ["RT_data.csv","ST_data.csv"]:
                print("正在写入：聚类类型-实时、聚类类型-长期")
                print(root)#D:\用电数据集\归一化之前的数据集\按聚类划分数据集_去异常\rtclass_1\滨江供电分公司\8
                id_now=root.split("\\")[-2]+"_"+root.split("\\")[-1] #临安区供电分公司_122
                Clustering_type=root.split("\\")[-3] #聚类类型
                if Clustering_type.startswith("rtclass"):
                    #print(root.split("\\")[-3]) rtclass交通运输仓储和邮政业1
                    data.loc[data[data['id'] == id_now].index,'新聚类类型-实时']=root.split("\\")[-3]
                    #print(data[data['id']==id_now]['新聚类类型-实时'])
                    # print(data[data['id']==id_now])
                    # 1/0
                    #df_res.loc[id_now,'聚类类型-实时'] = root.split("\\")[-3]
                if Clustering_type.startswith("stclass"):
                    data.loc[data[data['id'] == id_now].index,'新聚类类型-长期']=root.split("\\")[-3]
data.to_csv(path_or_buf='D:\\实验记录\\重要结果文件\\企业信息查询表2.csv', encoding="utf_8_sig",index=False)