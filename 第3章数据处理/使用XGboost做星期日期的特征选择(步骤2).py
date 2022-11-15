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
from ShowapiRequest import ShowapiRequest
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

warnings.filterwarnings("ignore")
# month_map=[0.048,0.28,0.005,0.008,0.015,0.056,0.036,0.310,0.0097,0.003,0.004,0.023]
month_map=[0.135, -0.062, 0.016, -0.039, 0.013, 0.104, 0.156, 0.355, 0.051, -0.036, -0.033, 0.124]
dir='D:\\用电数据集\\特征工程加强\\特征选择数据集_for_xgboost_汇总.csv'
dir_new='D:\\用电数据集\\特征工程加强\\特征选择数据集_for_xgboost_汇总(最终版本).csv'
try:
    ST_data = pd.read_csv(dir, encoding='utf-8', sep=',')
except:
    ST_data = pd.read_csv(dir, encoding='gbk', sep=',')

print(ST_data[:3])
ST_data['月份映射_遗传算法']=(ST_data['月份_1']*0.135+ST_data['月份_2']*-0.062+ST_data['月份_3']*0.016+ST_data['月份_4']*-0.039+
ST_data['月份_5']*0.013+ST_data['月份_6']*0.104+ST_data['月份_7']*0.156+ST_data['月份_8']*0.355+
ST_data['月份_9']*0.051+ST_data['月份_10']*-0.036+ST_data['月份_11']*-0.033+ST_data['月份_12']*0.124)

ST_data.to_csv(path_or_buf=dir_new, encoding="utf_8_sig",index=False)