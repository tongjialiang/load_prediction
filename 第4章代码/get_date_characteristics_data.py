# python3.6.5
# 需要引入requests包 ：运行终端->进入python/Scripts ->输入：pip install requests
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
import datetime
from chinese_calendar import is_workday
from chinese_calendar import is_holiday
from chinese_calendar import is_in_lieu

#读天气数据文件
try:
    weather_data = pd.read_csv('D:\\用电数据集\\嘉兴地区天气数据.csv', encoding='utf-8', sep=',')
except:
    weather_data = pd.read_csv('D:\\用电数据集\\嘉兴地区天气数据.csv', encoding='gbk', sep=',')

for i,row in weather_data.iterrows():
    #print(row['time'])#20160101
    rowtime=str(row['time'])
    print(rowtime)#20160101
    rowtime=list(rowtime)
    print(rowtime)#['2', '0', '1', '6', '0', '1', '0', '1']
    rowtime.insert(4,'-')
    print(rowtime)#['2', '0', '1', '6', '-', '0', '1', '0', '1']
    rowtime.insert(-2, '-')
    print(rowtime)#['2', '0', '1', '6', '-', '0', '1', '-', '0', '1']
    rowtime=''.join(rowtime)
    print(rowtime)#2016-01-01
    date_now = datetime.datetime.strptime(rowtime, "%Y-%m-%d")
    print(is_workday(date_now)) #是否是工作日
    weather_data.loc[i,'是否工作日'] = is_workday(date_now)*1
    # print(date_now.weekday())  # 0星期一
    # row['星期']=date_now.weekday()
    weather_data.loc[i, '星期'] = date_now.weekday()
    # print(date_now.month) #月份
    weather_data.loc[i, '月份'] = date_now.month
    # row['月份']=date_now.month
    # print(date_now.year)#年
    # row['年份']=date_now.year
    weather_data.loc[i, '年份'] = date_now.year
weather_data.to_csv(path_or_buf='D:\\用电数据集\\嘉兴地区天气和节假日数据.csv', encoding="utf_8_sig",index=False)











