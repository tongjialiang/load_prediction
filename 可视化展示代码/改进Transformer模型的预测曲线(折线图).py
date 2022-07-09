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
import seaborn as sns#使用的数据按频域分解聚类划分数据集V2_方案1-汇总标准化_stclass_15_ST_data_70_20.pk
#使用的数据按频域分解聚类划分数据集V2_方案1-汇总标准化_stclass_15_ST_data_70_20.pk
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

topath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第8章基于改进transformer模型的负荷预测方法\\"+"改进Transformer模型的预测曲线2.jpg"

sns.set_style('darkgrid',{'font.sans-serif':['SimHei','Arial']})

resource="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第8章基于改进transformer模型的负荷预测方法\\绘制预测曲线与损失曲线2\\"
with open(resource+'_MT_真实值.pk', 'rb') as f:
  y = pickle.load(f)
f.close()
#print(y)
x=range(len(y))
print(len(y))
with open(resource+'_MT_Transformer预测值.pk', 'rb') as f:
  y1 = pickle.load(f)
f.close()
with open(resource+'_MT_SVR预测值.pk', 'rb') as f:
  y2 = pickle.load(f)
f.close()
plt.xlabel("时间")
plt.ylabel("电力负荷(归一化)")
# print(y.sort_values().index.value)
#sns.barplot(x,y,order=x[y.sort_values(ascending=False).index])#降序

sns.lineplot(x,y,ci='red',label="真实值")
sns.lineplot(x,y1,ci='black',label="改进Transformer模型预测结果")
#sns.lineplot(x,y2,ci='black',label="支持向量回归模型预测结果")
# sns.lineplot(x,y2,ci='grey',label="一次性多点预测")
# sns.lineplot(x,y3,ci='gold',label="不考虑新息数据")
plt.legend()
plt.savefig(topath,dpi=1000)

plt.show()