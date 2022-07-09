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
#from ShowapiRequest import ShowapiRequest
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

file_name="D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集(行业划分后聚类)V2\\"
for root, dirs, filelist in os.walk(file_name):
    for i in filelist:
        if i in ['ST_data.csv']:
            #print(root)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_2020—2021每月取3周数据\交通运输仓储和邮政业\临安区供电分公司\26
            try:
                ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            #print(ST_data.columns)
            order=ST_data.columns
            #print(order)
            order=order.insert(1,order[3])
            order=order.delete(4)
            ST_data=ST_data[order]
            ST_data.to_csv(path_or_buf=root+'\\'+i, encoding="utf_8_sig",index=False)
        if i in ['RT_data.csv']:
            print(root)  # D:\用电数据集\特征工程加强\用于特征选择的行业数据集_2020—2021每月取3周数据\交通运输仓储和邮政业\临安区供电分公司\26
            # dirnew = 'D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集_特征增强\\' + root.split('\\')[-3] + '\\' + root.split('\\')[
            #     -2] + '\\' + root.split('\\')[-1] + '\\'
            # filenew = dirnew + i
            try:
                RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            #print(RT_data.columns)
            order=RT_data.columns
            #print(order)
            order=order.insert(1,order[2])
            order=order.delete(3)
            #print(order)
            RT_data=RT_data[order]
            RT_data.to_csv(path_or_buf=root+'\\'+i, encoding="utf_8_sig",index=False)

        if i in ['MT_data.csv']:
            print(root)  # D:\用电数据集\特征工程加强\用于特征选择的行业数据集_2020—2021每月取3周数据\交通运输仓储和邮政业\临安区供电分公司\26
            # dirnew = 'D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集_特征增强\\' + root.split('\\')[-3] + '\\' + root.split('\\')[
            #     -2] + '\\' + root.split('\\')[-1] + '\\'
            # filenew = dirnew + i
            try:
                MT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                MT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            #print(MT_data.columns)
            order=MT_data.columns
            #print(order)
            order=order.insert(1,order[3])
            order=order.delete(4)
            #print(order)
            MT_data = MT_data[order]
            MT_data.to_csv(path_or_buf=root+'\\'+i, encoding="utf_8_sig", index=False)