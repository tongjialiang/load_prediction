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
for root, dirs, filelist in os.walk("c:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集(负荷特性)\\"):
    for i in filelist:
        if i == 'MT_data.csv':
            print(root)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_2020—2021每月取3周数据\交通运输仓储和邮政业\临安区供电分公司\26
            root_new=root.replace("按聚类划分数据集(负荷特性)","按聚类划分数据集(负荷特性)V2")
            print(root_new)
            try:
                data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            data=data[:]
            if os.path.exists(root_new) == False:
                os.makedirs(root_new)
            data.to_csv(path_or_buf=root_new+ "\\"+i, encoding="utf_8_sig",index=False)

        if i == 'ST_data.csv':
            print(root)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_2020—2021每月取3周数据\交通运输仓储和邮政业\临安区供电分公司\26
            root_new=root.replace("按聚类划分数据集(负荷特性)","按聚类划分数据集(负荷特性)V2")
            print(root_new)
            try:
                data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            data=data[-365:]
            if os.path.exists(root_new) == False:
                os.makedirs(root_new)
            data.to_csv(path_or_buf=root_new+ "\\"+i, encoding="utf_8_sig",index=False)

        if i == 'RT_data.csv':
            print(root)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_2020—2021每月取3周数据\交通运输仓储和邮政业\临安区供电分公司\26
            root_new=root.replace("按聚类划分数据集(负荷特性)","按聚类划分数据集(负荷特性)V2")
            print(root_new)
            try:
                data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            data=data[-96*4:]
            if os.path.exists(root_new) == False:
                os.makedirs(root_new)
            data.to_csv(path_or_buf=root_new+ "\\"+i, encoding="utf_8_sig",index=False)
