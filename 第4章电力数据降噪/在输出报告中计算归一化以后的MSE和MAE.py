#!/usr/bin/python
# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt
import gc
import json
#from ShowapiRequest import ShowapiRequest
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
warnings.filterwarnings("ignore")

file_mapping={"RT":"rt_std_ssyg","MT":"mt_std_pjfh","ST":"st_std_pjfh"}
# with open('C:\\实验记录\\聚类kmeans对数据集做分类-标准化\\25分类数据集\\kmeans_for_classification.pk', 'rb') as f:
#     data_cluster_v25 = pickle.load(f)
#     print(data_cluster_v25)
#     1/0
with open('C:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按聚类划分数据集(频域分解动态规整)V2.pk', 'rb') as f:
    data_fdt_dtw = pickle.load(f)
f.close()

with open('C:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按负荷特性聚类划分数据集V2.pk', 'rb') as f:
    data_fhtx = pickle.load(f)
f.close()
#print(data_fhtx["class1"]["rt_std_ssyg"])

with open('C:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按地域划分数据集.pk', 'rb') as f:
    data_area = pickle.load(f)
f.close()

#print(data_area["海盐县"]["rt_std_ssyg"])

with open('C:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按行业划分数据集.pk', 'rb') as f:
    data_business = pickle.load(f)
f.close()
#print(data_business["科学研究和技术服务业与信息传输软件与信息技术服务业"]["rt_std_ssyg"])

with open('C:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按聚类划分数据集2V2.pk', 'rb') as f:
    data_cluster_from_business = pickle.load(f)
f.close()
print(data_cluster_from_business)
# 1/0
# print(data_cluster_from_business["rtclass制造业机械电子制造业1"])
# 1/0
#print(data_cluster_from_business["rtclass科学研究和技术服务业与信息传输软件与信息技术服务业1"]["rt_std_ssyg"])

for root, dirs, filelist in os.walk("C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\预测结果1次汇总\\"):
    for i in filelist:
        if i.endswith("csv"):
            print(root+i)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_2020—2021每月取3周数据\交通运输仓储和邮政业\临安区供电分公司\26
            # dirnew='D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集_特征增强\\'+root.split('\\')[-3]+'\\'+root.split('\\')[-2]+'\\'+root.split('\\')[-1]+'\\'
            # filenew=dirnew+i
            try:
                data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')

            for j, row in data.iterrows():  # 遍历每一行数据 j=index  row=data
                print(row["类别"])
                if row["数据类型"] in data_fdt_dtw:
                    #print(data_area[row["数据类型"]][file_mapping[row["类别"]]])
                    thisstd=data_fdt_dtw[row["数据类型"]][file_mapping[row["类别"]]]
                elif row["数据类型"] in data_fhtx:
                    thisstd = data_fhtx[row["数据类型"]][file_mapping[row["类别"]]]
                elif row["数据类型"] in data_area:
                    thisstd = data_area[row["数据类型"]][file_mapping[row["类别"]]]
                elif row["数据类型"] in data_business:
                    thisstd = data_business[row["数据类型"]][file_mapping[row["类别"]]]
                elif row["数据类型"] in data_cluster_from_business:
                    thisstd = data_cluster_from_business[row["数据类型"]][file_mapping[row["类别"]]]
                elif row["数据类型"]=="制造业机械电子制造业":
                    thisstd = data_business["制造业_机械电子制造业"][file_mapping[row["类别"]]]
                elif row["数据类型"]=="制造业资源加工工业":
                    thisstd = data_business["制造业_资源加工工业"][file_mapping[row["类别"]]]
                elif row["数据类型"]=="制造业轻纺工业":
                     thisstd = data_business["制造业_轻纺工业"][file_mapping[row["类别"]]]

                elif row["数据类型"]=="rtclass制造业机械电子制造业1":
                     thisstd = data_cluster_from_business["rtclass制造业_机械电子制造业1"][file_mapping[row["类别"]]]
                elif row["数据类型"] == "rtclass制造业机械电子制造业2":
                    thisstd = data_cluster_from_business["rtclass制造业_机械电子制造业2"][file_mapping[row["类别"]]]
                elif row["数据类型"] == "stclass制造业机械电子制造业1":
                    thisstd = data_cluster_from_business["stclass制造业_机械电子制造业1"][file_mapping[row["类别"]]]
                elif row["数据类型"] == "stclass制造业机械电子制造业2":
                    thisstd = data_cluster_from_business["stclass制造业_机械电子制造业2"][file_mapping[row["类别"]]]

                elif row["数据类型"]=="rtclass制造业资源加工工业1":
                     thisstd = data_cluster_from_business["rtclass制造业_资源加工工业1"][file_mapping[row["类别"]]]
                elif row["数据类型"] == "rtclass制造业资源加工工业2":
                    thisstd = data_cluster_from_business["rtclass制造业_资源加工工业2"][file_mapping[row["类别"]]]
                elif row["数据类型"] == "stclass制造业资源加工工业1":
                    thisstd = data_cluster_from_business["stclass制造业_资源加工工业1"][file_mapping[row["类别"]]]
                elif row["数据类型"] == "stclass制造业资源加工工业2":
                    thisstd = data_cluster_from_business["stclass制造业_资源加工工业2"][file_mapping[row["类别"]]]

                elif row["数据类型"]=="rtclass制造业轻纺工业1":
                     thisstd = data_cluster_from_business["rtclass制造业_轻纺工业1"][file_mapping[row["类别"]]]
                elif row["数据类型"] == "rtclass制造业轻纺工业2":
                    thisstd = data_cluster_from_business["rtclass制造业_轻纺工业2"][file_mapping[row["类别"]]]
                elif row["数据类型"] == "stclass制造业轻纺工业1":
                    thisstd = data_cluster_from_business["stclass制造业_轻纺工业1"][file_mapping[row["类别"]]]
                elif row["数据类型"] == "stclass制造业轻纺工业2":
                    thisstd = data_cluster_from_business["stclass制造业_轻纺工业2"][file_mapping[row["类别"]]]
                else:
                    print(row["数据类型"])

                data.loc[j,'标准差']=thisstd
                data.loc[j, 'MSE(反标准化)'] = row["MSE_score"]*thisstd*thisstd
                data.loc[j, 'MAE(反标准化)'] = row["MAE"]*thisstd
                data.loc[j, 'RMSE(反标准化)'] = row["RMSE_score"] * thisstd
            data.to_csv(path_or_buf=root+"\\"+i, encoding="utf_8_sig",index=False)
