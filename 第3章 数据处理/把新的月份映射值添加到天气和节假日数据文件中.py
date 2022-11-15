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
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

try:
    jx_data = pd.read_csv('D:\\用电数据集\\特征工程加强\\嘉兴地区天气和节假日数据_增强.csv', encoding='utf-8', sep=',')
except:
    jx_data = pd.read_csv('D:\\用电数据集\\特征工程加强\\嘉兴地区天气和节假日数据_增强.csv', encoding='gbk', sep=',')
jx_data['月份映射_遗传算法'] = (jx_data['月份_1']*0.135+jx_data['月份_2']*-0.062+jx_data['月份_3']*0.016+jx_data['月份_4']*-0.039+
jx_data['月份_5']*0.013+jx_data['月份_6']*0.104+jx_data['月份_7']*0.156+jx_data['月份_8']*0.355+
jx_data['月份_9']*0.051+jx_data['月份_10']*-0.036+jx_data['月份_11']*-0.033+jx_data['月份_12']*0.124)

jx_data.to_csv(path_or_buf='D:\\用电数据集\\特征工程加强\\嘉兴地区天气和节假日数据_增强.csv', encoding="utf_8_sig", index=False)

try:
    hz_data = pd.read_csv('D:\\用电数据集\\特征工程加强\\杭州地区天气和节假日数据_增强.csv', encoding='utf-8', sep=',')
except:
    hz_data = pd.read_csv('D:\\用电数据集\\特征工程加强\\杭州地区天气和节假日数据_增强.csv', encoding='gbk', sep=',')
hz_data['月份映射_遗传算法'] = (hz_data['月份_1']*0.135+hz_data['月份_2']*-0.062+hz_data['月份_3']*0.016+hz_data['月份_4']*-0.039+
hz_data['月份_5']*0.013+hz_data['月份_6']*0.104+hz_data['月份_7']*0.156+hz_data['月份_8']*0.355+
hz_data['月份_9']*0.051+hz_data['月份_10']*-0.036+hz_data['月份_11']*-0.033+hz_data['月份_12']*0.124)

hz_data.to_csv(path_or_buf='D:\\用电数据集\\特征工程加强\\杭州地区天气和节假日数据_增强.csv', encoding="utf_8_sig", index=False)