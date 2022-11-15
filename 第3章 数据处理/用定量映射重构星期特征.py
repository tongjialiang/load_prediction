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

week_map_all=[]
week_map=[0.191, 0.135, 0.132, 0.151, 0.184, 0.793, 0.950]
for i in range(12*3):#7*12*3
    week_map_all.extend(week_map)

for root, dirs, filelist in os.walk("D:\\用电数据集\\特征工程加强\\用于特征选择的行业数据集_2020—2021每月取3周数据\\"):
        for i in filelist:
            if i == 'ST_data.csv':
                print(root)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_2020—2021每月取3周数据\交通运输仓储和邮政业\临安区供电分公司\26
                dirnew='D:\\用电数据集\\特征工程加强\\'+'用于特征选择的行业数据集_2020—2021每月取3周数据_星期增强\\'+root.split('\\')[-3]+'\\'+root.split('\\')[-2]+'\\'+root.split('\\')[-1]+'\\'
                filenew=dirnew+i
                try:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                ST_data['星期特征_遗传算法']=week_map_all
                if os.path.exists(dirnew) == False:
                    os.makedirs(dirnew)
                ST_data.to_csv(path_or_buf=filenew, encoding="utf_8_sig",index=False)