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
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square
from skopt import BayesSearchCV  # pip install scikit-optimize
from sklearn.linear_model import Ridge
import use_save_csv
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
from sklearn.neighbors import KNeighborsRegressor
import gc
import auto_search
from lightgbm import LGBMRegressor
import xgboost as xgb
import sklearn.linear_model as sl
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
with open('C:\\实验记录\\实验结果\\SVR_v3\\SVR_v3.pk', 'rb') as f:
    mymodel = pickle.load(f)
f.close()
model_area=mymodel["按地域划分数据集V2_方案1-汇总标准化_萧山区_RT_data_96_20.pk"][0]

#print(model_area)
with open('C:\\数据采样完成new\\区间估计_方案1_汇总标准化_萧山区_RT_data_96_20.pk', 'rb') as f:
    data_area = pickle.load(f)
f.close()


#######################################################################################

X_test1 = data_area[0][:, :, 3]
#print(X_test2.shape)  #(2945, 96)
X_test1 = X_test1.reshape(len(X_test1), -1)
Y_test1 = data_area[1][:, :, 3]
Z_test1 = data_area[1][:, :, 4:]  # 存储除了用电负荷以外的特征
X_test1 = np.concatenate((X_test1, Z_test1[:, 0, :]), axis=1)
y_predict_area=model_area.predict(X_test1)
#print(y_predict_area.shape)#(2945,)
print("地域MAE")
MAE_area = mean_absolute_error(Y_test1[:,0], y_predict_area)
print(MAE_area)


#误差的正态性拟合检验


# k2, p = scipy.stats.normaltest(y_predict_area)
# print(k2,p)

data=Y_test1[:,0]-y_predict_area
print(data.shape)
data=list(data)

# print(np.isnan(data))
# 1/0

# k2, p = scipy.stats.kstest(data,'norm')
# print(k2,p)
# k2, p = scipy.stats.shapiro(data)
# print(k2,p)
#
# res=scipy.stats.chi2_contingency(data)
# print(res)

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
#plt.title('误差频率分布直方图（萧山区供电分公司4_98）')
# 显示图形
plt.savefig("C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第9章不同数据集划分方式的融合与区间估计\\误差频率分布直方图(萧山区供电分公司4_98).jpg",dpi=1000)
plt.show()
print(np.std(data)**2)
print(np.mean(data))


plt.plot(Y_test1[:,0][:100],label='真实值')
plt.plot((y_predict_area+2*np.std(y_predict_area))[:100],linestyle='--',label='预测上界')
plt.plot((y_predict_area-2*np.std(y_predict_area))[:100],linestyle='--',label='预测下界')
#plt.title("保守的区间估计(置信水平0.95)")
plt.xlabel("时间")
plt.ylabel("负荷值(标准化)")
plt.legend()
plt.savefig("C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第9章不同数据集划分方式的融合与区间估计\\保守的区间估计(置信水平0.95).jpg",dpi=1000)
plt.show()

plt.plot(Y_test1[:,0][:100],label='真实值')
plt.plot((y_predict_area+np.std(y_predict_area))[:100],linestyle='--',label='预测上界')
plt.plot((y_predict_area-np.std(y_predict_area))[:100],linestyle='--',label='预测下界')
#plt.title("积极的区间估计(置信水平0.68)")
plt.xlabel("时间")
plt.ylabel("负荷值(标准化)")
plt.legend()
plt.savefig("C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第9章不同数据集划分方式的融合与区间估计\\积极的区间估计(置信水平0.68).jpg",dpi=1000)
plt.show()

