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
#from ShowapiRequest import ShowapiRequest
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
warnings.filterwarnings("ignore")

for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之后的数据集-待采样\\按频域分解聚类划分数据集V2_领域自适应-汇总标准化\\"):
    for i in filelist:
        if i == 'RT_data.csv':
            class_cluster=root.split("\\")[-3]
            try:
                data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')

            if class_cluster in ["rtclass_1", "rtclass_10", "rtclass_12", "rtclass_15", "rtclass_19"
                                 , "rtclass_24", "rtclass_3", "rtclass_6", "rtclass_91"]:
                data["聚类领域类别"] = 0
            else:
                data["聚类领域类别"] = 1
            data.to_csv(path_or_buf=root+"\\"+i, encoding="utf_8_sig",index=False)

        if i in ['ST_data.csv','MT_data.csv']:
            class_cluster=root.split("\\")[-3]
            try:
                data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')

            if class_cluster in ["stclass_1", "stclass_11", "stclass_12", "stclass_14", "stclass_16", "stclass_2",
                                 "stclass_23","stclass_26","stclass_24","stclass_29","stclass_32","stclass_40","stclass_6"]:
                data["聚类领域类别"] = 0
            else:
                data["聚类领域类别"] = 1
            data.to_csv(path_or_buf=root+"\\"+i, encoding="utf_8_sig",index=False)