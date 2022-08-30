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
import sys
import seaborn as sns
#设置需要保留的小数位数
pd.set_option('precision', 3)

# def read_excel(path):
#   data_xls = pd.ExcelFile(path)
#   print("这个excel文件有多少sheet页？")
#   print(data_xls.sheet_names)#['地区', '行业', '电力']
#   data={}
#   for name in data_xls.sheet_names:
#     df=data_xls.parse(name)
#     #print(df)
#     data[name]=df
#     # print(df)
#     # print(name)
#   return data

###########################################################
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
warnings.filterwarnings("ignore")
# 解决中文和负号显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 使显示图标自适应
plt.rcParams['figure.autolayout'] = True
###########################################################
# jpg_name="聚类个数与无效分类公司数的关系"
# sheet_name='聚类kmeans调参n_clusters结果(统计表) '
# frompath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\聚类kmeans调参n_clusters结果(统计表) .xls"
# topath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\"+jpg_name
# data=read_excel(frompath)
# print("打印有多少列")
# print(data[sheet_name].columns)########
# sns.set_style('darkgrid',{'font.sans-serif':['SimHei','Arial']})
# # plt.rcParams['figure.dpi'] = 1000
# sns.set_palette(palette="dark")
# x=data[sheet_name]["聚类个数"]########
# y=data[sheet_name]["无效分类的公司"]########
# #plt.xticks(rotation=20)#######倾斜角度
# plt.xlabel("聚类个数")
# plt.ylabel("无效分类的公司")
# # print(y.sort_values().index.value)
# #sns.barplot(x,y,order=x[y.sort_values(ascending=False).index])#降序
# sns.lineplot(x,y)
# plt.savefig(topath,dpi=1000)
# plt.show()
topath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第7章电力负荷预测系统的搭建\\"+"新息纳入算法的消融实验结果(误差比较)"
y1=[[0.102,0.048,0.089,0.056,0.066,0.065,0.083,0.054,0.090,0.083,0.047,0.064,0.071],
    [0.097,0.143,0.138,0.055,0.079,0.164,0.117,0.156,0.165,0.037,0.068,0.062,0.084],
    [0.141,0.056,0.121,0.068,0.086,0.058,0.101,0.064,0.163,0.043,0.080,0.086,0.122]]
plt.boxplot(y1,labels=["新息纳入+循环预测","一次预测多个时点","不纳入新息"],showmeans=1)
#plt.xlabel("s")
plt.grid(True)
plt.legend()
plt.ylabel("均方损失")
plt.savefig(topath,dpi=1000)
plt.show()