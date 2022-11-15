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

# try:
#     data = pd.read_csv('D:\\用电数据集\\特征工程加强\\宏观经济数据-地区.csv', encoding='utf-8', sep=',')
# except:
#     data = pd.read_csv('D:\\用电数据集\\特征工程加强\\宏观经济数据-地区.csv', encoding='gbk', sep=',')

# try:
#     data = pd.read_csv('D:\\用电数据集\\特征工程加强\\宏观经济数据-行业.csv', encoding='utf-8', sep=',')
# except:
#     data = pd.read_csv('D:\\用电数据集\\特征工程加强\\宏观经济数据-行业.csv', encoding='gbk', sep=',')

try:
    data = pd.read_csv('D:\\用电数据集\\特征工程加强\\宏观经济数据-电力弹性系数.csv', encoding='utf-8', sep=',')
except:
    data = pd.read_csv('D:\\用电数据集\\特征工程加强\\宏观经济数据-电力弹性系数.csv', encoding='gbk', sep=',')
# print(data.columns)
# 1/0

# y=data['全年用电量_百亿千瓦时_p529_p571']     ###地区
# y=data['按行业分全社会用电情况_亿千瓦时_p305']     ###行业
y=data['全省电力消费量(百亿千瓦小时)']     ###电力弹性系数
print(y[:5])
# x=data[['按地区分组的法人单位数(万人)_p35', '总人口数(万人)_p46',
#        '各市规模以上企业年末单位就业人员(万人)_p72', '各市、县居民消费价格指数_p144', '土地面积(万平方公里)_p539',
#        '生产总值(百亿元)_p518_p539', '全年用电量_百亿千瓦时_p529_p571',
#        '城镇居民人均可支配收入(万元)_p537_p539', '第一产业(百亿元)_p537_p539',
#        '第二产业(百亿元)_p537_p539', '第三产业(百亿元)_p537_p539']]

# x=data[['按行业分的法人单位数_p29', '按产业分的全省生产总值_亿元_p18', '按行业分的全省生产总值_亿元_p18',
#        '总支出_亿元_p22', '按行业和经济类型分的就业人员总数_非私营与规上私营之和_年末数_万人_p61_p71',
#        '项目建成投产率_p84', '分行业全社会单位就业人员年平均工资_p163', '按行业分全社会用电情况_亿千瓦时_p305',
#        '按产业分用电合计_亿千瓦时_p305']]

x=data[['全省能源生产量(百万吨标准煤)', '全省电力生产量(百亿千瓦小时)', '能源生产比上年增长(%)',
       '电力生产比上年增长(%)', '生产总值比上年增长\n', '能源生产弹性系数', '电力生产弹性系数',
       '全省能源消费量(百万吨标准煤)', '全省电力消费量(百亿千瓦小时)', '能源消费比上年增长(%)', '电力消费比上年增长(%)',
       '能源消费弹性系数', '电力消费弹性系数']]
print(x[:5])
#随机搜索参数
grid = RandomizedSearchCV(model, param_dist, cv=None, scoring='neg_mean_squared_error', n_iter=50)
grid.fit(x,y)

best_estimator = grid.best_estimator_
best_estimator.fit(x, y)
y_p=best_estimator.predict(x)
MSE_score = mean_squared_error(y, y_p)
print(MSE_score)
best_params = str(grid.best_params_)

fig,ax = plt.subplots(figsize=(35,8))
# 解决中文和负号显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 使显示图标自适应
plt.rcParams['figure.autolayout'] = True
xgb.plot_importance(best_estimator,height=0.8,ax=ax,max_num_features=64)

# with open('D:\\实验记录\\pk\\Xgboost分析宏观经济数据-地区特征重要性.pk', 'wb+') as f:
#     pickle.dump(best_estimator, f)
# plt.savefig('D:\\用电数据集\\特征工程加强\\Xgboost分析宏观经济数据-地区特征重要性.png', format='png')

# with open('D:\\实验记录\\pk\\Xgboost分析宏观经济数据-行业特征重要性.pk', 'wb+') as f:
#     pickle.dump(best_estimator, f)
# plt.savefig('D:\\用电数据集\\特征工程加强\\Xgboost分析宏观经济数据-行业特征重要性.png', format='png')

with open('D:\\实验记录\\pk\\Xgboost分析宏观经济数据-电力弹性系数特征重要性.pk', 'wb+') as f:
    pickle.dump(best_estimator, f)
plt.savefig('D:\\用电数据集\\特征工程加强\\Xgboost分析宏观经济数据-电力弹性系数特征重要性.png', format='png')
