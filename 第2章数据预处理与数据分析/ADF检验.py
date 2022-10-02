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
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import  *
#from ShowapiRequest import ShowapiRequest
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
warnings.filterwarnings("ignore")
root="C:\\用电数据集\\分类后\\浙江省电力公司2021\\杭州供电公司\\建德市供电分公司2\\136"

try:
    data = pd.read_csv(root + "\\" + "RT_data.csv", encoding='utf-8', sep=',')
except:
    data = pd.read_csv(root + "\\" + "RT_data.csv", encoding='gbk', sep=',')

data = data["瞬时有功(kW)"].sort_values()
#data=data.dropna()
dftest=stattools.adfuller(data)
dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
sdracf=stattools.acf(data)
plot_acf(data,use_vlines=True,lags=60)#自相关
plt.show()
sdracf=stattools.pacf(data)
plot_pacf(data,use_vlines=True,lags=60)#偏自相关
plt.show()
