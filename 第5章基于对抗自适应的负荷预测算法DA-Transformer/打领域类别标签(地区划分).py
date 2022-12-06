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

try:
    table = pd.read_csv("D:\\实验记录\\重要结果文件\\企业信息查询表2.csv", encoding='utf-8', sep=',')
except:
    table = pd.read_csv("D:\\实验记录\\重要结果文件\\企业信息查询表2.csv" + "\\" + i, encoding='gbk', sep=',')
#print(table)

for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之后的数据集-待采样\\按频域分解聚类划分数据集V2_领域自适应-汇总标准化\\"):
    for i in filelist:
        if i in ['RT_data.csv','ST_data.csv','MT_data.csv']:
            #class_cluster=root.split("\\")[-3]
            try:
                data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            class_business=data["id"][0]
            area_name=table[table["id"]==class_business]["所在地区"].array[0]
            if area_name in ['萧山区','余杭区','海宁市','海宁市','海宁市','平湖市','嘉兴市桐乡市','嘉善县']:
                data["地区领域类别"] = 0
            else:
                data["地区领域类别"] = 1
                #print("111")
            data.to_csv(path_or_buf=root+"\\"+i, encoding="utf_8_sig",index=False)