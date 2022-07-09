from matplotlib import pyplot as plt
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
warnings.filterwarnings("ignore")
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# 设置参数搜索范围
param_dist = {
'max_depth': range(3, 10, 1),
'num_leaves': [20, 50, 100, 200, 350, 500],  #
'min_child_weight': [1, 2, 3, 4, 5, 6],
'subsample': [0.6, 0.7, 0.8, 0.9],
'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
'reg_alpha': [0.05, 0.1, 1, 2, 3],
'reg_lambda': [0.05, 0.1, 1, 2, 3],
'lambda': [0, 0.1, 0.5, 1],
'learning_rate': [0.05, 0.07, 0.1, 0.2, 0.3],
'n_estimators': [400, 500, 600],
'gamma': [0.2, 0.3, 0.4, 0.5, 0.6],
'verbosity': [0],
'tree_method': ['gpu_hist']
        }#c'n_jobs':[1]


# 设置模型
model = xgb.XGBRegressor()

try:
    data = pd.read_csv('D:\\用电数据集\\特征工程加强\\特征选择数据集_for_xgboost_汇总(最终版本).csv', encoding='utf-8', sep=',')
except:
    data = pd.read_csv('D:\\用电数据集\\特征工程加强\\特征选择数据集_for_xgboost_汇总(最终版本).csv', encoding='gbk', sep=',')
print(data.columns)
y=data['平均负荷(kW)']
print(y[:5])
x=data[['受电容量(KVA)','max_temperature', 'min_temperature', '平均气温', '是否工作日', 'weather_中雨',
       'weather_多云', 'weather_大雨', 'weather_小雨', 'weather_晴', 'weather_暴雨',
       'weather_阴', 'weather_阵雨', 'weather_雪', '星期_0', '星期_1', '星期_2', '星期_3',
       '星期_4', '星期_5', '星期_6', '月份_1', '月份_2', '月份_3', '月份_4', '月份_5', '月份_6',
       '月份_7', '月份_8', '月份_9', '月份_10', '月份_11', '月份_12', 'year_2016',
       'year_2017', 'year_2018', 'year_2019', 'year_2020', 'year_2021',
       '小到中雨及下雪天', '大雨暴雨天', '不下雨', '天气映射', '周末', '星期映射', '春节期间', '春', '夏', '秋',
       '冬', '星期特征_遗传算法','月份映射_遗传算法']]
print(x[:5])
#随机搜索参数
grid = RandomizedSearchCV(model, param_dist, cv=None, scoring='neg_mean_squared_error', n_iter=1)
grid.fit(x,y)

best_estimator = grid.best_estimator_
best_estimator.fit(x, y)
y_p=best_estimator.predict(x)
MSE_score = mean_squared_error(y, y_p)
print(MSE_score)
best_params = str(grid.best_params_)

fig,ax = plt.subplots(figsize=(35,15))
# 解决中文和负号显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 使显示图标自适应
plt.rcParams['figure.autolayout'] = True
xgb.plot_importance(best_estimator,height=0.5,ax=ax,max_num_features=64)

with open('D:\\实验记录\\pk\\用于特征选择的xgb模型_日期天气(最终版本)2.pk', 'wb+') as f:
    pickle.dump(best_estimator, f)

plt.savefig('D:\\用电数据集\\特征工程加强\\Xgboost分析天气日期特征重要性(最终版本)2.png', format='png')
