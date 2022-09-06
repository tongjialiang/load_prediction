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
#root="C:\\用电数据集\\分类后\\浙江省电力公司2021\\杭州供电公司\\嘉善县供电分公司1\\40"

df_res = pd.DataFrame()
for root, dirs, filelist in os.walk("C:\\用电数据集\\分类后\\浙江省电力公司2021\\"):
    for i in filelist:
        if i == 'RT_data.csv':
            print(root)
            try:
                data = pd.read_csv(root + "\\" + "RT_data.csv", encoding='utf-8', sep=',')
            except:
                data = pd.read_csv(root + "\\" + "RT_data.csv", encoding='gbk', sep=',')
            data = data["瞬时有功(kW)"][:1000].sort_values()
            k2, p = scipy.stats.normaltest(data)
            #w, p2 = scipy.stats.kstest(data,cdf=norm)
            #print('p:',p)
            df_res = df_res.append([{'root': root,'k2':k2,'p':p,'skew':skew(data),'kurtosis':kurtosis(data)}])
dir='C:\\实验记录\\重要结果文件\\'
if os.path.exists(dir) == False:
    os.makedirs(dir)
df_res.to_csv(path_or_buf=dir+"数据分布倾斜分析.csv", encoding="utf_8_sig",index=False)
#
# try:
#     data = pd.read_csv(root + "\\" + "RT_data.csv", encoding='utf-8', sep=',')
# except:
#     data = pd.read_csv(root + "\\" + "RT_data.csv", encoding='gbk', sep=',')
#
# data = data["瞬时有功(kW)"].sort_values()
#
#
#
# plt.hist(data, density=True, edgecolor='w', label='直方图')
# sns.kdeplot(data, label='频率图')
# #plt.xlim(-1,4)
# # 显示图例
# plt.legend()
# # 解决中文和负号显示问题
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# # 使显示图标自适应
# plt.rcParams['figure.autolayout'] = True
# plt.title('嘉善县供电分公司1_40负荷分布图')
# # 显示图形
# plt.show()
# k2, p = scipy.stats.normaltest(data)
# print('p:',p)
# shapiro_test, shapiro_p = scipy.stats.shapiro(data)
# print("Shapiro-Wilk Stat:",shapiro_test, " Shapiro-Wilk p-Value:", shapiro_p)

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