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
disk="D:\\用电数据集\\Informer对比实验用的数据集\\"
files=os.listdir(disk)
print(files)
for k in files:
    res_data=''
    #print(len(res_data))
    for root, dirs, filelist in os.walk(disk+k):
        for i in filelist:
            if i in  ['ST_data.csv','RT_data.csv','MT_data.csv']:
                #dirnew=disk
                #filenew=dirnew+i
                try:
                    data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                if len(res_data)==0:
                    res_data=data
                else:
                    res_data=pd.concat([res_data, data], axis=0)
                #ST_data.to_csv(path_or_buf=root+"\\"+i, encoding="utf_8_sig",index=False)
    print(len(res_data))
    if res_data.columns.__contains__("日期"):
        res_data["date"] = res_data["日期"]
        del res_data["日期"]
    if res_data.columns.__contains__("数据时间"):
        res_data["date"] = res_data["数据时间"]
        del res_data["数据时间"]
    if res_data.columns.__contains__("瞬时有功(kW)"):
        res_data["OT"] = res_data["瞬时有功(kW)"]
        del res_data["瞬时有功(kW)"]
    if res_data.columns.__contains__("平均负荷(kW)"):
        res_data["OT"] = res_data["平均负荷(kW)"]
        del res_data["平均负荷(kW)"]
    for f in res_data.columns:
        if f not in ["date","OT"]:
            del res_data[f]
    res_data.to_csv(path_or_buf=disk+k+".csv", encoding="utf_8_sig",index=False)
    # print(res_data[:3])
