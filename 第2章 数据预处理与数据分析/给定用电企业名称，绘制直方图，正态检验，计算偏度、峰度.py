#!/usr/bin/python
# -*- coding: utf-8 -*-
from scipy.stats import norm
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
import seaborn as sns
import gc
import json
from scipy.stats import skew, kurtosis
#from ShowapiRequest import ShowapiRequest
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
warnings.filterwarnings("ignore")
root="c:\\用电数据集\\画正态分布\\建德市\\建德市供电分公司2\\147"

try:
    data = pd.read_csv(root + "\\" + "RT_data.csv", encoding='utf-8', sep=',')
except:
    data = pd.read_csv(root + "\\" + "RT_data.csv", encoding='gbk', sep=',')

data = data["瞬时有功(kW)"].sort_values()

plt.xlabel("瞬时有功(kW)")
plt.ylabel("频率")

plt.hist(data, density=True, edgecolor='w', label='直方图')
sns.kdeplot(data, label='频率图')
#plt.xlim(-1,4)
# 显示图例
plt.legend()
# 解决中文和负号显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 使显示图标自适应
plt.rcParams['figure.autolayout'] = True
plt.title('数据方差为0（建德市供电分公司2_147）')
# 显示图形

k2, p = scipy.stats.normaltest(data)
print('k2:',k2)
print('p:',p)
print(skew(data))#偏度 正态是0，负为左倾斜
print(kurtosis(data))#峰度
shapiro_test, shapiro_p = scipy.stats.shapiro(data)
print("Shapiro-Wilk Stat:",shapiro_test, " Shapiro-Wilk p-Value:", shapiro_p)
plt.savefig('C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第6章电力数据质量分析与异常监控系统\\正态检验QQ图6')
plt.show()
# C:\用电数据集\分类后\浙江省电力公司2021\杭州供电公司\临安区供电分公司2\81
# 0.05432334424843857
# C:\用电数据集\分类后\浙江省电力公司2021\杭州供电公司\余杭区供电分公司6\170
# 0.06072626025127996
# C:\用电数据集\分类后\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\33
# 0.0900069205494243
# C:\用电数据集\分类后\浙江省电力公司2021\杭州供电公司\嘉善县供电分公司1\154
# 0.1949134669301114
# C:\用电数据集\分类后\浙江省电力公司2021\杭州供电公司\嘉善县供电分公司1\40
# 0.8984880776387336