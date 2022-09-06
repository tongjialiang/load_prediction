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

for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按地域划分数据集_特征增强\\"):
    for i in filelist:
        if i == 'ST_data.csv':
            try:
                ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(ST_data.columns)
            # 1/0
            del ST_data['平均气温']
            del ST_data['天气映射']
            del ST_data['按地区分组的法人单位数(万人)_p35']
            del ST_data['总人口数(万人)_p46']
            del ST_data['各市规模以上企业年末单位就业人员(万人)_p72']
            del ST_data['生产总值(百亿元)_p518_p539']
            del ST_data['全年用电量_百亿千瓦时_p529_p571']
            del ST_data['第一产业(百亿元)_p537_p539']
            del ST_data['第二产业(百亿元)_p537_p539']
            del ST_data['第三产业(百亿元)_p537_p539']
            ST_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)
        if i == 'MT_data.csv':
            try:
                MT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                MT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(MT_data.columns)
            # 1/0
            del MT_data['按地区分组的法人单位数(万人)_p35']
            del MT_data['总人口数(万人)_p46']
            del MT_data['各市规模以上企业年末单位就业人员(万人)_p72']
            del MT_data['生产总值(百亿元)_p518_p539']
            del MT_data['全年用电量_百亿千瓦时_p529_p571']
            del MT_data['第一产业(百亿元)_p537_p539']
            del MT_data['第二产业(百亿元)_p537_p539']
            del MT_data['第三产业(百亿元)_p537_p539']
            MT_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)

        if i == 'RT_data.csv':
            try:
                RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(RT_data.columns)
            # 1/0
            del RT_data['平均气温']
            del RT_data['天气映射']
            del RT_data['按地区分组的法人单位数(万人)_p35']
            del RT_data['总人口数(万人)_p46']
            del RT_data['各市规模以上企业年末单位就业人员(万人)_p72']
            del RT_data['生产总值(百亿元)_p518_p539']
            del RT_data['全年用电量_百亿千瓦时_p529_p571']
            del RT_data['第一产业(百亿元)_p537_p539']
            del RT_data['第二产业(百亿元)_p537_p539']
            del RT_data['第三产业(百亿元)_p537_p539']
            RT_data.to_csv(path_or_buf=root+'\\'+i, encoding="utf_8_sig", index=False)


for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_特征增强\\"):
    for i in filelist:
        if i == 'ST_data.csv':
            try:
                ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(ST_data.columns)
            # 1/0
            del ST_data['按行业分的法人单位数_p29']
            del ST_data['按行业分的全省生产总值_亿元_p18']
            del ST_data['分行业全社会单位就业人员年平均工资_p163']
            del ST_data['按行业分全社会用电情况_亿千瓦时_p305']
            ST_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)
        if i == 'MT_data.csv':
            try:
                MT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                MT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(MT_data.columns)
            # 1/0
            del MT_data['按行业分的法人单位数_p29']
            del MT_data['按行业分的全省生产总值_亿元_p18']
            del MT_data['分行业全社会单位就业人员年平均工资_p163']
            del MT_data['按行业分全社会用电情况_亿千瓦时_p305']
            MT_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)

        if i == 'RT_data.csv':
            try:
                RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(RT_data.columns)
            # 1/0
            del RT_data['按行业分的法人单位数_p29']
            del RT_data['按行业分的全省生产总值_亿元_p18']
            del RT_data['分行业全社会单位就业人员年平均工资_p163']
            del RT_data['按行业分全社会用电情况_亿千瓦时_p305']
            RT_data.to_csv(path_or_buf=root+'\\'+i, encoding="utf_8_sig", index=False)

for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之后的数据集-待采样(多特征-多点预测)\\按聚类划分数据集_方案2-行业划分后聚类\\"):
    for i in filelist:
        if i == 'ST_data.csv':
            try:
                ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(ST_data.columns)
            # 1/0
            del ST_data['按行业分的法人单位数_p29']
            del ST_data['按行业分的全省生产总值_亿元_p18']
            #del ST_data['分行业全社会单位就业人员年平均工资_p163']
            del ST_data['按行业分全社会用电情况_亿千瓦时_p305']
            ST_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)
        if i == 'MT_data.csv':
            try:
                MT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                MT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(MT_data.columns)
            # 1/0
            del MT_data['按行业分的法人单位数_p29']
            del MT_data['按行业分的全省生产总值_亿元_p18']
            #del MT_data['分行业全社会单位就业人员年平均工资_p163']
            del MT_data['按行业分全社会用电情况_亿千瓦时_p305']
            MT_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)

        if i == 'RT_data.csv':
            try:
                RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(RT_data.columns)
            # 1/0
            del RT_data['按行业分的法人单位数_p29']
            del RT_data['按行业分的全省生产总值_亿元_p18']
            #del RT_data['分行业全社会单位就业人员年平均工资_p163']
            del RT_data['按行业分全社会用电情况_亿千瓦时_p305']
            RT_data.to_csv(path_or_buf=root+'\\'+i, encoding="utf_8_sig", index=False)