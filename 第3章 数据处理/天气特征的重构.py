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
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

root='D:\\用电数据集\\特征工程加强\\嘉兴地区天气和节假日数据.csv'
try:
    weather_data = pd.read_csv(root , encoding='utf-8', sep=',')
except:
    weather_data = pd.read_csv(root , encoding='gbk', sep=',')
print(weather_data[:5])
#对字段天气、星期、月份、年份进行独热编码
weather_data_onehot=pd.get_dummies(weather_data,columns=['weather','星期','月份','year'])
print(weather_data_onehot[:5])
#天气字段重采样
print(weather_data['weather'].unique())#['多云' '小雨' '中雨' '阴' '雪' '晴' '阵雨' '大雨' '暴雨']
weather_data_onehot['小到中雨及下雪天']=((weather_data.loc[:,'weather'] == '小雨')|
                                (weather_data.loc[:,'weather'] == '中雨')|
                               (weather_data.loc[:,'weather'] == '中雨')|
                               (weather_data.loc[:,'weather'] == '雪'))*1
weather_data_onehot['大雨暴雨天']=((weather_data.loc[:,'weather'] == '大雨')|
                               (weather_data.loc[:,'weather'] == '暴雨'))*1
weather_data_onehot['不下雨']=((weather_data.loc[:,'weather'] == '多云')|
                            (weather_data.loc[:,'weather'] == '晴')|
                            (weather_data.loc[:,'weather'] == '阴'))*1

weather_data_onehot['天气映射']=(weather_data.loc[:,'weather'] == '晴')*0.01+(weather_data.loc[:,'weather'] == '多云')*0.05\
                            +(weather_data.loc[:,'weather'] == '阴')*0.1+(weather_data.loc[:,'weather'] == '小雨')*0.3\
                            +(weather_data.loc[:,'weather'] == '阵雨')*0.38+(weather_data.loc[:,'weather'] == '中雨')*0.5\
                            +(weather_data.loc[:,'weather'] == '雪')*0.4+(weather_data.loc[:,'weather'] == '大雨')*0.75\
                            +(weather_data.loc[:,'weather'] == '暴雨')*0.99

#星期字段重采样
weather_data_onehot['周末']=((weather_data.loc[:,'星期'] == 6)|
                            (weather_data.loc[:,'星期'] == 5))*1
weather_data_onehot['星期映射']=(weather_data.loc[:,'星期'] == 0)*0.07+(weather_data.loc[:,'星期'] == 1)*0.14\
                            +(weather_data.loc[:,'星期'] == 2)*0.13+(weather_data.loc[:,'星期'] == 3)*0.14\
                            +(weather_data.loc[:,'星期'] == 4)*0.18+(weather_data.loc[:,'星期'] == 5)*0.81\
                            +(weather_data.loc[:,'星期'] == 6)*0.91
weather_data_onehot['星期映射_遗传算法']=(weather_data.loc[:,'星期'] == 0)*0.191+(weather_data.loc[:,'星期'] == 1)*0.135\
                            +(weather_data.loc[:,'星期'] == 2)*0.132+(weather_data.loc[:,'星期'] == 3)*0.151\
                            +(weather_data.loc[:,'星期'] == 4)*0.184+(weather_data.loc[:,'星期'] == 5)*0.793\
                            +(weather_data.loc[:,'星期'] == 6)*0.95
#0.08 0.15 0.14 0.15 0.19 0.8  0.9
#月份字段重采样
weather_data_onehot['春节期间']=((weather_data.loc[:,'月份'] == 1)|
                            (weather_data.loc[:,'月份'] == 2))*1

weather_data_onehot['春']=((weather_data.loc[:,'月份'] == 1)|
                            (weather_data.loc[:,'月份'] == 2)|
                            (weather_data.loc[:,'月份'] == 3))*1

weather_data_onehot['夏']=((weather_data.loc[:,'月份'] == 4)|
                            (weather_data.loc[:,'月份'] == 5)|
                            (weather_data.loc[:,'月份'] == 6))*1
weather_data_onehot['秋']=((weather_data.loc[:,'月份'] == 7)|
                            (weather_data.loc[:,'月份'] == 8)|
                            (weather_data.loc[:,'月份'] == 9))*1
weather_data_onehot['冬']=((weather_data.loc[:,'月份'] == 10)|
                            (weather_data.loc[:,'月份'] == 11)|
                            (weather_data.loc[:,'月份'] == 12))*1
print(weather_data_onehot[:5])
print("打印是否有空值")
print(weather_data_onehot.isnull().any())
weather_data_onehot.to_csv(path_or_buf='D:\\用电数据集\\嘉兴地区天气和节假日数据_增强.csv', encoding="utf_8_sig",index=False)
