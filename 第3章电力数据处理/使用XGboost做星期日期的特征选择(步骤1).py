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
import scipy.stats as stats
import gc
import json
from ShowapiRequest import ShowapiRequest

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

res=pd.DataFrame()
count_i=0

for root, dirs, filelist in os.walk("D:\\用电数据集\\特征工程加强\\用于特征选择的行业数据集_2020—2021每月取3周数据_星期增强\\"):
        for i in filelist:
            print(root)
            if i == 'ST_data.csv':
                count_i+=1

                try:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                res=res.append(ST_data)
            if len(res)>252*500:
                res.to_csv(path_or_buf='D:\\用电数据集\\特征工程加强\\特征选择数据集_for_xgboost'+'_'+str(count_i)+'.csv', encoding="utf_8_sig",index=False)
                res = pd.DataFrame()#清空
res.to_csv(path_or_buf='D:\\用电数据集\\特征工程加强\\特征选择数据集_for_xgboost'+'_'+str(count_i)+'.csv', encoding="utf_8_sig",index=False)
print(len(res))