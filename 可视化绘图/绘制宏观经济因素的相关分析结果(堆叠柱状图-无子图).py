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
frompath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第四章电力相关因素的研究\\宏观经济特征分析结果.xls"
topath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第四章电力相关因素的研究\\宏观经济特征分析结果.jpg"
data=read_excel(frompath)
print("打印有多少列")
print(data["综合"].columns)########
# print(data["行业"].columns)########
# 1/0
sns.set_style('darkgrid',{'font.sans-serif':['SimHei','Arial']})
sns.set_palette(palette="dark")
fig=plt.figure()
# ax1 = fig.add_subplot()
# ax1.set_xlabel("地域、行业、电力相关因素")
# ax1.set_ylabel("相关系数")
#
# ax2 = ax1.twinx()
# ax2.set_ylabel("显著中度相关公司数")
x=data["综合"]["特征"]########
y1=data["综合"]["皮尔逊系数"]########
y2=data["综合"]["斯皮尔曼系数"]
y3=data["综合"]["二列相关系数"]
y4=data["综合"]["Xgboost特征重要性"]
#plt.bar(x,y,align="center",color='k',alpha=0.7)
plt.xticks(rotation=90)#######倾斜角度
# ax1.set_xticklabels(x,rotation = 90)#####子图旋转90度
plt.xlabel("地域、行业、电力相关因素")
plt.ylabel("相关系数")
plt.bar(x,y1,align="center",color='b',alpha=0.7,label="皮尔逊相关系数")
plt.bar(x,y2,bottom=y1,align="center",color='g',alpha=0.7,label="斯皮尔曼相关系数")
plt.bar(x,y3,bottom=y1+y2,align="center",color='m',alpha=0.7,label="二列相关系数")
plt.bar(x,y4,bottom=y1+y2+y3,align="center",color='k',alpha=0.7,label="Xgboost特征重要性")
# ax2.plot(x,y5,label="显著中度相关公司数—皮尔逊",color='b')
# ax2.plot(x,y6,label="显著中度相关公司数_斯皮尔曼",color='g')
# ax2.plot(x,y7,label="显著中度相关公司数_二列相关",color='m')
# print(y.sort_values().index.value)
#sns.barplot(x,y,order=x[y.sort_values(ascending=False).index])#降序
#sns.lineplot(x,y)

plt.legend()
# ax2.legend(loc="lower right",bbox_to_anchor=(1,0.7))
plt.savefig(topath,dpi=1000)
plt.show()

