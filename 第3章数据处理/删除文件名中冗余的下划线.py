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

for root, dirs, filelist in os.walk("D:\\数据采样完成new\\"):#数据采样完成new
    for i in filelist:
        if i.startswith('按聚类划分数据集'):
            print(i)#按聚类划分数据集_方案1-汇总标准化_rtclass_26_RT_data_96_20.pk
            #print(i.split('_')[:3]+i.split('_')[3:])
            j='_'.join(i.split('_')[:3])+i.split('_')[3]+'_'+'_'.join(i.split('_')[4:])
            print(j)

            os.rename(root+'\\'+i, root+'\\'+j)
# with open("D:\\1\\按聚类划分数据集_方案1-汇总标准化_stclass16_MT_data_6_2.pk", 'rb') as f:
#     data = pickle.load(f)
# f.close()
# print(data[0])
# print(data[1])
