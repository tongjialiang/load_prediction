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

# with open('C:\\实验记录\\聚类dbscan做异常检测\\exception_detection_samples.pk', 'rb') as f:
#     data = pickle.load(f)
# f.close()
# print(data['rt'][0])





with open('C:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按地域划分数据集.pk', 'rb') as f:
    data = pickle.load(f)
f.close()
#print(data["海盐县"]["rt_std_ssyg"])

with open('C:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按行业划分数据集.pk', 'rb') as f:
    data = pickle.load(f)
f.close()
#print(data["科学研究和技术服务业与信息传输软件与信息技术服务业"]["rt_std_ssyg"])

with open('C:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按聚类划分数据集2.pk', 'rb') as f:
    data = pickle.load(f)
f.close()
#print(data["rtclass科学研究和技术服务业与信息传输软件与信息技术服务业1"]["rt_std_ssyg"])

with open('C:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_不划分数据集V2.pk', 'rb') as f:
    data = pickle.load(f)
f.close()
print(data)