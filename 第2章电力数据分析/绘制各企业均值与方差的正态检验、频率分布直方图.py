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
#from ShowapiRequest import ShowapiRequest
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
warnings.filterwarnings("ignore")

with open('C:\\实验记录\\聚类dbscan做异常检测\\exception_detection_samples.pk', 'rb') as f:
    data = pickle.load(f)
f.close()
lmean=[]
lstd=[]
data_all=data['rt'][0]
# print(data_all)
for i in data_all:
    lmean.append(i[0])
    lstd.append(i[1])
# print(len(lmean))
# print(len(lstd))
# lmean=pd.Series(lmean).sort_values()[1000:-1000]
# lstd=pd.Series(lstd).sort_values()[1000:1000]
aa=(pd.Series(lstd)/pd.Series(lmean)).sort_values()[1000:3000]
#lmean=(lmean-np.mean(lmean))/np.std(lmean)
#lstd=(lstd-np.mean(lstd))/np.std(lstd)
# print(lmean)

# print(lmean.min())
# print(lmean.max())
# bin_val_mean = np.arange(start= lmean.min(), stop= lmean.max(), step = 0.1)
# mu_mean, std_mean = np.mean(lmean), np.std(lmean)
# print(mu_mean)
# print(std_mean)
# #
# #
# p = norm.pdf(lmean, mu_mean, std_mean)
# print(p)
# #
# #
# plt.hist(lmean,bins = bin_val_mean,density=True, stacked=False)
# plt.plot(lmean, p, color = 'red')
# #plt.xlim(-1,3)
# # plt.xticks(np.arange(-0.4,2,5),rotation=90)
# #plt.xlabel('Human Body Temperature Distributions')
# #plt.xlabel('human body temperature')
# plt.show()
#
#
#print('Average (Mu): '+ str(mu) + ' / ' 'Standard Deviation: '+str(std))

# data = pd.Series(lmean)  # 将数据由数组转换成series形式
# plt.hist(data, density=True, edgecolor='w', label='直方图')
# data.plot(kind='kde', label='密度图')
#
# # 显示图例
# plt.legend()
# # 显示图形
# plt.show()


##################################
data = aa
##################################
# bin_val_mean = np.arange(start= data.min(), stop= data.max(), step = 1)
# plt.hist(data, density=True, bins=bin_val_mean,edgecolor='w', label='直方图')
# sns.kdeplot(data, label='密度图')
# plt.xlim(-1,4)
# # 显示图例
# plt.legend()
# # 解决中文和负号显示问题
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# # 使显示图标自适应
# plt.rcParams['figure.autolayout'] = True
# # 显示图形
# plt.show()
##############################


k2, p = scipy.stats.normaltest(data)
print(p)
shapiro_test, shapiro_p = scipy.stats.shapiro(data)
print("Shapiro-Wilk Stat:",shapiro_test, " Shapiro-Wilk p-Value:", shapiro_p)