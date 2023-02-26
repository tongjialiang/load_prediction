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
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

warnings.filterwarnings("ignore")
# 解决中文和负号显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 使显示图标自适应
plt.rcParams['figure.autolayout'] = True
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
with open('d:\\实验记录\\实验结果\\SVR_v3\\SVR_v3.pk', 'rb') as f:
    mymodel = pickle.load(f)
f.close()
with open('d:\\实验记录\\实验结果\\Xgboost_v3-res\\Xgboost_v3.pk', 'rb') as f:
    mymodel2 = pickle.load(f)
f.close()
model_area=mymodel["按地域划分数据集V2_方案1-汇总标准化_海盐县_RT_data_96_20.pk"][0]
model_business=mymodel["按行业划分数据集V2_方案1-汇总标准化_科学研究和技术服务业与信息传输软件与信息技术服务业_RT_data_96_20.pk"][0]
model_cluster=mymodel["按聚类划分数据集V2_方案2-行业划分后聚类_rtclass科学研究和技术服务业与信息传输软件与信息技术服务业1_RT_data_96_20.pk"][0]
model_xgb=mymodel2["按聚类划分数据集V2_方案2-行业划分后聚类_rtclass科学研究和技术服务业与信息传输软件与信息技术服务业1_RT_data_96_20.pk"][0]
#print(model_area)
with open('d:\\数据采样完成new\\组合建模地域_方案1_汇总标准化_海盐县_RT_data_96_20.pk', 'rb') as f:
    data_area = pickle.load(f)
f.close()
with open('d:\\数据采样完成new\\组合建模聚类_方案1_汇总标准化_rtclass科学研究和技术服务业与信息传输软件与信息技术服务业1_RT_data_96_20.pk', 'rb') as f2:
    data_cluster = pickle.load(f2)
f2.close()
with open('d:\\数据采样完成new\\组合建模行业_方案1_汇总标准化_科学研究和技术服务业与信息传输软件与信息技术服务业_RT_data_96_20.pk', 'rb') as f3:
    data_business = pickle.load(f3)
f3.close()

#######################################################################################

X_test1 = data_area[0][:, :, 3]
#print(X_test2.shape)  #(2945, 96)
X_test1 = X_test1.reshape(len(X_test1), -1)
Y_test1 = data_area[1][:, :, 3]
Z_test1 = data_area[1][:, :, 4:]  # 存储除了用电负荷以外的特征
X_test1 = np.concatenate((X_test1, Z_test1[:, 0, :]), axis=1)
#print(X_test2.shape)  #(2945, 96)
#print(Y_test2.shape)  #(2945, 96)
# print(Z_test.shape)  #(2945, 20, 4)
# print(X_test.shape)#(2945, 100)
y_predict_area=model_area.predict(X_test1)
#print(y_predict_area.shape)#(2945,)
print("地域MAE")
# print(Y_test2[:,0].shape)
# print(y_predict_area.shape)

MAE_area = mean_absolute_error(Y_test1[:,0], y_predict_area)
print(MAE_area)

#######################################################################################
X_test2 = data_business[0][:, :, 3]
X_test2 = X_test2.reshape(len(X_test2), -1)
Y_test2 = data_business[1][:, :, 3]

Z_test2 = data_business[1][:, :, 4:]  # 存储除了用电负荷以外的特征
X_test2 = np.concatenate((X_test2, Z_test2[:, 0, :]), axis=1)
# print(X_test.shape)  #(2945, 96)
# print(Y_test.shape)  #(2945, 96)
# print(Z_test.shape)  #(2945, 20, 4)
# print(X_test.shape)#(2945, 100)
y_predict_business=model_business.predict(X_test2)
#print(y_predict_business.shape)#(2945,)

print("行业MAE")
MAE_business = mean_absolute_error(Y_test2[:,0], y_predict_business)
print(MAE_business)

###############
X_test3 = data_cluster[0][:, :, 3]
X_test3 = X_test3.reshape(len(X_test3), -1)
Y_test3 = data_cluster[1][:, :, 3]
Z_test3 = data_cluster[1][:, :, 4:]  # 存储除了用电负荷以外的特征
X_test3 = np.concatenate((X_test3, Z_test3[:, 0, :]), axis=1)
# print(X_test.shape)  #(2945, 96)
# print(Y_test.shape)  #(2945, 96)
# print(Z_test.shape)  #(2945, 20, 4)
# print(X_test.shape)#(2945, 100)
y_predict_cluster=model_cluster.predict(X_test3)
# y_predict_cluster=model_business.predict(X_test)
# y_predict_cluster=model_area.predict(X_test)
#print(y_predict_cluster.shape)#(2945,)

print("聚类MAE")
MAE_cluster = mean_absolute_error(Y_test3[:,0], y_predict_cluster)
print(MAE_cluster)
###################################################################
X_test4 = data_cluster[0][:, :, 3]
X_test4 = X_test3.reshape(len(X_test3), -1)
Y_test4 = data_cluster[1][:, :, 3]
Z_test4 = data_cluster[1][:, :, 4:]  # 存储除了用电负荷以外的特征
#X_test4 = np.concatenate((X_test4, Z_test4[:, 0, :]), axis=1)
# print(X_test.shape)  #(2945, 96)
# print(Y_test.shape)  #(2945, 96)
# print(Z_test.shape)  #(2945, 20, 4)
# print(X_test.shape)#(2945, 100)
y_xgb=model_xgb.predict(X_test4)
# y_predict_cluster=model_business.predict(X_test)
# y_predict_cluster=model_area.predict(X_test)
#print(y_predict_cluster.shape)#(2945,)

print("xgn的MAE")
MAE_xgb = mean_absolute_error(Y_test4[:,0], y_xgb)
print(MAE_xgb)

###################################################################
y_real=Y_test1[:,0]*395.5939905272621+131.89831319910814
###################################################################3
#等权输出
# Y_test4=(Y_test1[:,0]+Y_test2[:,0]+Y_test3[:,0])/3
# MAE_equal_weight = mean_absolute_error(Y_test1[:,0], Y_test4)
# print("MAE等权组合")
# print(MAE_equal_weight)
y_predict_equal_weight=(y_predict_area*395.5939
                        +y_predict_business*534.8658
                        +y_predict_cluster*181.4467
                        +y_xgb*181.4467
                        +131.8983+151.0331+78.0661+78.0661)/4
# MAE_equal_weight = mean_absolute_error(Y_test1[:,0], Y_test4)
print("MAE等权组合")
MAE_equal_weight = mean_absolute_error(y_predict_equal_weight,y_real)
print(MAE_equal_weight)
# print(Y_test1[:,0][0])
# print(Y_test1[:,0]*395.5939905272621+131.89831319910814)
# print(Y_test2[:,0][0])
# print(Y_test2[:,0]*534.8658601611729+151.03311551889857)
# print(Y_test3[:,0][0])
# print(Y_test3[:,0]*181.44672546828272+78.06619392714073)
###################################################################3
#最优化方法
h11=np.sum((y_predict_area*395.5939905272621-Y_test1[:,0]*395.5939905272621)**2)
h22=np.sum((y_predict_business*534.8658-Y_test2[:,0]*534.8658)**2)
h33=np.sum((y_predict_cluster*181.44672-Y_test3[:,0]*181.44672)**2)
h44=np.sum((y_xgb*181.44672-Y_test4[:,0]*181.44672)**2)
h_all=1/h11+1/h22+1/h33+1/h33+1/h44
w1=1/(h11*h_all)
w2=1/(h22*h_all)
w3=1/(h33*h_all)
w4=1/(h44*h_all)
print("w1,w2,w3,w4")
print(w1)
print(w2)
print(w3)
print(w4)
y_predict_optimize=((y_predict_area*395.5939+131.8983)*w1
                    +(y_predict_business*534.8658+151.0331)*w2
                    +(y_predict_cluster*181.4467+78.0661)*w3
                    +(y_xgb*181.4467+78.0661)*w4)
# MAE_equal_weight = mean_absolute_error(Y_test1[:,0], Y_test4)
print("MAE最优化方法")
MAE_optimize = mean_absolute_error(y_predict_optimize,y_real)
print(MAE_optimize)

#线性回归###################################################################
#weight_vector=np.array([0.334,0.333,0.333])
x=np.array([y_predict_area*395.5939+131.8983,
            y_predict_business*534.8658+151.0331,
            y_predict_cluster*181.4467+78.0661,
            y_xgb*181.4467+78.0661])
#print(weight_vector.shape)
#print(x.shape)
#print(np.dot(weight_vector.T,x))
y_real=np.array(y_real)
model=sl.LinearRegression()
print(x.shape)#(3, 2953)
print(y_real.shape)#(2953,)
model.fit(x.T,y_real)
y_predict_lr=model.predict(x.T)
print(model.coef_)#[ 0.7074958  -2.46706303  1.94098542  0.73871939]
y_predict_lr=(y_predict_area*395.5939+131.8983)*0.7074958+(y_predict_business*534.8658+151.0331)*-2.46706303+(y_predict_cluster*181.4467+78.0661)*1.94098542+(y_xgb*181.4467+78.0661)*0.73871939
print("MAE线性回归方法")
MAE_lr = mean_absolute_error(y_predict_lr,y_real)
print(MAE_lr)
##########################################################################

#模型友好分类器
#打分类器标签
x=x.T
y_real_2=np.expand_dims(y_real,axis=1)
#print(y_real_2.shape)#(2953,1)
y_real_repeat=y_real_2.repeat(4, axis=1)
#print(y_real_repeat.shape)#(2953, 4)
diff_predict_real=(abs(x-y_real_repeat))#计算误差
print(diff_predict_real[:2])

x_sort=np.argsort(diff_predict_real, axis=1)#排序
friend_y_real0=x_sort[:,0]#标签：最优模型
friend_y_real1=x_sort[:,1]#标签：次优模型
friend_y_real2=x_sort[:,2]#标签：次次优模型
friend_y_real3=x_sort[:,3]#标签：最弱模型

#根据基模型预测值与真实值的误差的绝对值计算组合权重
sum0=diff_predict_real[[np.arange(len(x))],[friend_y_real0]][0].sum()
sum1=diff_predict_real[[np.arange(len(x))],[friend_y_real1]][0].sum()
sum2=diff_predict_real[[np.arange(len(x))],[friend_y_real2]][0].sum()
sum3=diff_predict_real[[np.arange(len(x))],[friend_y_real3]][0].sum()
sum0=sum3-sum0
sum1=sum3-sum1
sum2=sum3-sum2
sum3=sum3-sum3
sum_all=sum1+sum2+sum3+sum0
w_0=(sum0)/sum_all
w_1=(sum1)/sum_all
w_2=(sum2)/sum_all
w_3=(sum3)/sum_all
print(w_0,w_1,w_2,w_3)#0.4924325067089302 0.32009411383877784 0.18747337945229203 0.0

#对4个基模型构建4个模型友好分类器，预测样本对基模型在集成系统中的重要性
xgb_m0=RandomForestClassifier()
xgb_m0.fit(x,friend_y_real0)
friend_y0=xgb_m0.predict(x)
#print(friend_y0[:-5])#(2953,)

xgb_m1=RandomForestClassifier()
xgb_m1.fit(x,friend_y_real1)
friend_y1=xgb_m0.predict(x)

xgb_m2=RandomForestClassifier()
xgb_m2.fit(x,friend_y_real2)
friend_y2=xgb_m0.predict(x)

xgb_m3=RandomForestClassifier()
xgb_m3.fit(x,friend_y_real3)
friend_y3=xgb_m3.predict(x)

#计算集成系统的预测值
y_friend0=x[[np.arange(len(x))],[friend_y0]][0]
y_friend1=x[[np.arange(len(x))],[friend_y1]][0]
y_friend2=x[[np.arange(len(x))],[friend_y2]][0]
y_friend3=x[[np.arange(len(x))],[friend_y3]][0]
#print(y_friend)
MAE_friend = mean_absolute_error(y_friend0*0.4924+y_friend1*0.3200+y_friend2*0.1874,y_real)
print(MAE_friend)#2.1824

##########################################################################
# 画图
sns.set_palette(palette="dark")
x=range(len(y_real))
y1=y_real
y2=y_predict_area
y3=y_predict_business
y4=y_predict_cluster
y5=y_predict_equal_weight
y6=y_predict_lr
y7=y_predict_optimize
y8=y_xgb
y9=y_friend0*0.4924+y_friend1*0.3200+y_friend2*0.1874
#plt.xticks(rotation=20)#######倾斜角度
plt.xlabel("时间步")
plt.ylabel("负荷预测值")
f=1000
t=1050
sns.lineplot(x[f:t],y1[f:t],label="真实值",linestyle='--')
sns.lineplot(x[f:t],y2[f:t]*395.5939+131.8983,label="地域划分数据集的预测值(SVR)",alpha=0.3)
sns.lineplot(x[f:t],y3[f:t]*534.8658+151.0331,label="行业划分数据集的预测值(SVR)",alpha=0.3)
sns.lineplot(x[f:t],y4[f:t]*181.4467+78.0661,label="聚类划分数据集的预测值(SVR)",alpha=0.3)
sns.lineplot(x[f:t],y8[f:t]*181.4467+78.0661,label="聚类划分数据集的预测值(Xgboost)",alpha=0.3)
sns.lineplot(x[f:t],y5[f:t],label="等权融合的预测值",alpha=0.3)
sns.lineplot(x[f:t],y6[f:t],label="线性回归法融合的预测值",alpha=0.3)
sns.lineplot(x[f:t],y7[f:t],label="方差分析法融合的预测值",alpha=0.3)
sns.lineplot(x[f:t],y9[f:t],label="基于模型友好分类器的预适应的预测值",linestyle='--',color='red')
plt.legend(loc="lower right")
plt.savefig("d:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\\第8章数据集融合、模型组合、区间估计\\不同融合方式的预测结果-DA.jpg",dpi=1000)

plt.show()