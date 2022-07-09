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


# week_map_all=[]
# week_map=[0.191, 0.135, 0.132, 0.151, 0.184, 0.793, 0.950]
month_map_all=[]
month_map=[0.135, -0.062, 0.016, -0.039, 0.013, 0.104, 0.156, 0.355, 0.051, -0.036, -0.033, 0.124]
# for i in range(12*3):#7*12*3
#     week_map_all.extend(week_map)
def extend_map():
    for i in range(21):#7*12*3
        month_map_all.extend([month_map[6]])#extend() 向列表尾部追加一个列表，将列表中的每个元素都追加进来，在原有列表上增加
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[7]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[8]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[9]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[10]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[11]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[0]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[1]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[2]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[3]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[4]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[5]])

extend_map()
for root, dirs, filelist in os.walk("D:\\用电数据集\\特征工程加强\\用于特征选择的行业数据集_2020—2021每月取3周数据_星期增强\\"):
        for i in filelist:
            if i == 'ST_data.csv':
                print(root)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_2020—2021每月取3周数据\交通运输仓储和邮政业\临安区供电分公司\26
                dirnew='D:\\用电数据集\\特征工程加强\\'+'用于特征选择的行业数据集_2020—2021每月取3周数据_星期增强_月份增强\\'+root.split('\\')[-3]+'\\'+root.split('\\')[-2]+'\\'+root.split('\\')[-1]+'\\'
                filenew=dirnew+i
                try:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                ST_data['月份特征_遗传算法']=month_map_all
                if os.path.exists(dirnew) == False:
                    os.makedirs(dirnew)
                ST_data.to_csv(path_or_buf=filenew, encoding="utf_8_sig",index=False)