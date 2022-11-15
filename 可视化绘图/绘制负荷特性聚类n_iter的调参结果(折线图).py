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

def read_excel(path):
  data_xls = pd.ExcelFile(path)
  print("这个excel文件有多少sheet页？")
  print(data_xls.sheet_names)#['地区', '行业', '电力']
  data={}
  for name in data_xls.sheet_names:
    df=data_xls.parse(name)
    #print(df)
    data[name]=df
    # print(df)
    # print(name)
  return data

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
frompath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\聚类kmeans调参n_init结果.xls"
topath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\聚类kmeans调参n_init结果.jpg"
data=read_excel(frompath)
print("打印有多少列")
print(data["聚类kmeans调参n_init结果"].columns)########
sns.set_style('darkgrid',{'font.sans-serif':['SimHei','Arial']})
# plt.rcParams['figure.dpi'] = 1000
sns.set_palette(palette="dark")
x=data["聚类kmeans调参n_init结果"]["初始化次数"]########
y=data["聚类kmeans调参n_init结果"]["轮廓系数"]########
#plt.xticks(rotation=20)#######倾斜角度
plt.xlabel("初始化次数")
plt.ylabel("轮廓系数")
# print(y.sort_values().index.value)
#sns.barplot(x,y,order=x[y.sort_values(ascending=False).index])#降序
sns.lineplot(x,y)
plt.savefig(topath,dpi=1000)
plt.show()