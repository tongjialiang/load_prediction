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
frompath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\相关系数分析结果.xls"
topath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\地域相关因素与电力负荷的相关性3.jpg"
data=read_excel(frompath)
print("打印有多少列")
print(data["地区"].columns)########
# print(data["行业"].columns)########
# 1/0
sns.set_style('darkgrid',{'font.sans-serif':['SimHei','Arial']})
sns.set_palette(palette="dark")
x=data["地区"]["地区性相关因素"]########
y=data["地区"]["相关系数（全年用电量）"]########

#plt.bar(x,y,align="center",color='k',alpha=0.7)
plt.xticks(rotation=20)#######倾斜角度
plt.xlabel("地区相关因素")
plt.ylabel("相关系数（全年用电量）")
# print(y.sort_values().index.value)
sns.barplot(x,y,order=x[y.sort_values(ascending=False).index])#降序
plt.savefig(topath,dpi=1000)
plt.show()


###########################################################
#frompath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\相关系数分析结果.xlsm"
topath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\行业相关因素与电力负荷的相关性3.jpg"
#data=read_excel(frompath)
print("打印有多少列")
print(data["行业"].columns)########
sns.set_style('darkgrid',{'font.sans-serif':['SimHei','Arial']})
sns.set_palette(palette="dark")
x=data["行业"]["行业相关因素"].copy()########
y=data["行业"]["相关系数(全社会用电情况)"].copy()########
print(x,y)
#plt.bar(x,y,align="center",color='k',alpha=0.7)
plt.xticks(rotation=20)#######倾斜角度
plt.xlabel("行业相关因素")
plt.ylabel("相关系数（全社会用电情况）")
# print(y.sort_values().index.value)
#sns.barplot(x,y,order=x[y.sort_values(ascending=False).index])#降序
plt.bar(x,y)
# sns.barplot(x,y)#降序
plt.savefig(topath,dpi=1000)
plt.show()

###########################################################
#frompath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\相关系数分析结果.xlsm"
topath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\电力弹性系数相关因素与电力负荷的相关性3.jpg"
#data=read_excel(frompath)
print("打印有多少列")
print(data["电力"].columns)########
sns.set_style('darkgrid',{'font.sans-serif':['SimHei','Arial']})
sns.set_palette(palette="dark")
x=data["电力"]["电力相关因素"]########
y=data["电力"]["相关系数（电力消费量）"]########
#plt.bar(x,y,align="center",color='k',alpha=0.7)
plt.xticks(rotation=20)#######倾斜角度
plt.xlabel("电力弹性系数相关因素")
plt.ylabel("相关系数（电力消费量）")
# print(y.sort_values().index.value)
sns.barplot(x,y,order=x[y.sort_values(ascending=False).index])#降序
plt.savefig(topath,dpi=1000)
plt.show()