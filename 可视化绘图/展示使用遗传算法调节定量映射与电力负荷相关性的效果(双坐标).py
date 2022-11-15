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
frompath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第四章电力相关因素的研究\\GA特征增强算法.xls"

data=read_excel(frompath)
print("打印有多少列")
print(data["星期映射"].columns)########
sns.set_style('darkgrid',{'font.sans-serif':['SimHei','Arial']})
# plt.rcParams['figure.dpi'] = 1000
sns.set_palette(palette="dark")
##############################################################################
topath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第四章电力相关因素的研究\\星期映射的优化过程.jpg"
fig=plt.figure()
ax1 = fig.add_subplot()
ax1.set_xlabel("迭代更新轮次")
ax1.set_ylabel("平均相关系数")
ax2 = ax1.twinx()
ax2.set_ylabel("显著中度相关公司数")
x=range(len(data["星期映射"][data["星期映射"]["该基因是否遗传"]=="是"]["迭代轮次"]))
y1=data["星期映射"][data["星期映射"]["该基因是否遗传"]=="是"]["平均相关系数"]########
y2=data["星期映射"][data["星期映射"]["该基因是否遗传"]=="是"]["显著中度相关公司数"]########
#plt.xticks(rotation=20)#######倾斜角度
#plt.ylabel("定量映射值")
# print(y.sort_values().index.value)
#sns.barplot(x,y,order=x[y.sort_values(ascending=False).index])#降序

sns.lineplot(x,y1,ax=ax1,label="平均相关系数",color='blue')
sns.lineplot(x,y2,ax=ax2,label="显著中度相关公司数",color='orange')
# plt.xlabel("更新轮次")
# plt.tight_layout
ax1.legend(loc="lower right",bbox_to_anchor=(1,0.07))
ax2.legend(loc="lower right")
# plt.legend()
plt.savefig(topath,dpi=1000)
plt.show()

#######################################################################
##############################################################################
topath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第四章电力相关因素的研究\\月份映射的优化过程.jpg"
fig=plt.figure()
ax1 = fig.add_subplot()
ax1.set_xlabel("迭代更新轮次")
ax1.set_ylabel("平均相关系数")
ax2 = ax1.twinx()
ax2.set_ylabel("显著中度相关公司数")
x=range(len(data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["迭代轮次"]))
y1=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["平均相关系数"]########
y2=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["显著中度相关公司数"]########
#plt.xticks(rotation=20)#######倾斜角度
#plt.ylabel("定量映射值")
# print(y.sort_values().index.value)
#sns.barplot(x,y,order=x[y.sort_values(ascending=False).index])#降序

sns.lineplot(x,y1,ax=ax1,label="平均相关系数",color='blue')
sns.lineplot(x,y2,ax=ax2,label="显著中度相关公司数",color='orange')
# plt.xlabel("更新轮次")
# plt.tight_layout
ax1.legend(loc="lower right",bbox_to_anchor=(1,0.07))
ax2.legend(loc="lower right")

# plt.legend()
plt.savefig(topath,dpi=1000)
plt.show()
#######################################################################
topath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第四章电力相关因素的研究\\星期映射值的迭代更新过程.jpg"
x=data["星期映射"][data["星期映射"]["该基因是否遗传"]=="是"]["迭代轮次"]
x=range(len(x))
y1=data["星期映射"][data["星期映射"]["该基因是否遗传"]=="是"]["周一"]########
y2=data["星期映射"][data["星期映射"]["该基因是否遗传"]=="是"]["周二"]########
y3=data["星期映射"][data["星期映射"]["该基因是否遗传"]=="是"]["周三"]#######
y4=data["星期映射"][data["星期映射"]["该基因是否遗传"]=="是"]["周四"]#######
y5=data["星期映射"][data["星期映射"]["该基因是否遗传"]=="是"]["周五"]#######
y6=data["星期映射"][data["星期映射"]["该基因是否遗传"]=="是"]["周六"]########
y7=data["星期映射"][data["星期映射"]["该基因是否遗传"]=="是"]["周日"]#######
#plt.xticks(rotation=20)#######倾斜角度
plt.xlabel("迭代更新轮次")
plt.ylabel("定量映射值")
# print(y.sort_values().index.value)
#sns.barplot(x,y,order=x[y.sort_values(ascending=False).index])#降序

sns.lineplot(x,y1,label="周一")
sns.lineplot(x,y2,label="周二")
sns.lineplot(x,y3,label="周三")
sns.lineplot(x,y4,label="周四")
sns.lineplot(x,y5,label="周五")
sns.lineplot(x,y6,label="周六")
sns.lineplot(x,y7,label="周日")
plt.legend(loc='best')
plt.savefig(topath,dpi=1000)
plt.show()
#######################################################################
topath="C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第四章电力相关因素的研究\\月份映射值的迭代更新过程.jpg"
x=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["迭代轮次"]
x=range(len(x))
y1=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["1月"]########
y2=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["2月"]########
y3=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["3月"]#######
y4=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["4月"]#######
y5=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["5月"]#######
y6=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["6月"]########
y7=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["7月"]#######
y8=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["8月"]
y9=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["9月"]
y10=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["10月"]
y11=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["11月"]
y12=data["月份映射"][data["月份映射"]["该基因是否遗传"]=="是"]["12月"]
#plt.xticks(rotation=20)#######倾斜角度
plt.xlabel("迭代更新轮次")
plt.ylabel("定量映射值")
# print(y.sort_values().index.value)
#sns.barplot(x,y,order=x[y.sort_values(ascending=False).index])#降序

sns.lineplot(x,y1,label="1月")
sns.lineplot(x,y2,label="2月")
sns.lineplot(x,y3,label="3月")
sns.lineplot(x,y4,label="4月")
sns.lineplot(x,y5,label="5月")
sns.lineplot(x,y6,label="6月")
sns.lineplot(x,y7,label="7月")
sns.lineplot(x,y8,label="8月")
sns.lineplot(x,y9,label="9月")
sns.lineplot(x,y10,label="10月")
sns.lineplot(x,y11,label="11月")
sns.lineplot(x,y12,label="12月")

plt.legend(loc="upper right")
plt.savefig(topath,dpi=1000)
plt.show()
