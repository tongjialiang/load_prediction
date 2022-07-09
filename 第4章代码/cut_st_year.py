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
for root, dirs, filelist in os.walk("D:\\用电数据集\\特征工程加强\\用于特征选择的行业数据集_字段已添加"):
        for i in filelist:
            #filename=root+"\\"+i
            #print(filename)
            if i == 'ST_data.csv':
                print(root)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_字段已添加\交通运输仓储和邮政业\临安区供电分公司\122
                dir_new='D:\\用电数据集\\特征工程加强\\用于特征选择的行业数据集_2020—2021每月取3周数据\\'+\
                                                                root.split('\\')[-3]+'\\'+root.split('\\')[-2]+'\\'+root.split('\\')[-1]
                file_new=dir_new+'//'+i
                try:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                #print(ST_data)
                ST_data['数据时间'] = ST_data['数据时间'].str.strip('\t')  # 删除末尾换行符
                #ST_data['数据时间'] = pd.to_datetime(ST_data['数据时间'], format='%Y-%m-%d')  # 转datetime
                #print(ST_data[:5])
                #print(ST_data[ST_data['数据时间'] == '2020-06-29'])

                df_res_temp = pd.DataFrame()  # 存放临时结果集
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-06'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-07'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-08'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-09'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-10'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-11'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-12'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-13'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-14'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-15'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-16'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-17'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-18'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-19'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-20'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-21'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-22'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-23'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-24'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-25'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-07-26'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-03'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-04'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-05'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-06'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-07'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-08'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-09'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-10'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-11'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-12'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-13'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-14'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-15'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-16'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-17'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-18'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-19'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-20'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-21'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-22'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-08-23'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-07'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-08'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-09'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-10'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-11'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-12'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-13'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-14'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-15'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-16'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-17'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-18'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-19'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-20'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-21'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-22'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-23'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-24'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-25'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-26'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-09-27'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-05'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-06'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-07'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-08'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-09'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-10'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-11'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-12'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-13'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-14'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-15'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-16'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-17'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-18'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-19'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-20'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-21'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-22'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-23'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-24'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-10-25'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-02'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-03'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-04'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-05'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-06'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-07'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-08'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-09'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-10'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-11'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-12'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-13'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-14'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-15'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-16'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-17'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-18'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-19'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-20'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-21'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-11-22'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-07'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-08'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-09'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-10'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-11'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-12'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-13'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-14'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-15'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-16'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-17'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-18'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-19'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-20'])

                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-21'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-22'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-23'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-24'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-25'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-26'])
                df_res_temp=df_res_temp.append(ST_data[ST_data['数据时间'] == '2020-12-27'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-04'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-05'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-06'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-07'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-08'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-09'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-10'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-11'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-12'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-13'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-14'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-15'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-16'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-17'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-18'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-19'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-20'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-21'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-22'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-23'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-01-24'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-01'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-02'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-03'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-04'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-05'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-06'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-07'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-08'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-09'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-10'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-11'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-12'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-13'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-14'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-15'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-16'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-17'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-18'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-19'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-20'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-02-21'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-01'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-02'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-03'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-04'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-05'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-06'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-07'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-08'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-09'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-10'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-11'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-12'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-13'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-14'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-15'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-16'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-17'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-18'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-19'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-20'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-03-21'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-05'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-06'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-07'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-08'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-09'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-10'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-11'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-12'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-13'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-14'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-15'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-16'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-17'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-18'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-19'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-20'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-21'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-22'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-23'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-24'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-04-25'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-10'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-11'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-12'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-13'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-14'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-15'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-16'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-17'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-18'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-19'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-20'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-21'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-22'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-23'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-24'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-25'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-26'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-27'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-28'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-29'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-05-30'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-07'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-08'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-09'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-10'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-11'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-12'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-13'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-14'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-15'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-16'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-17'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-18'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-19'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-20'])

                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-21'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-22'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-23'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-24'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-25'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-26'])
                df_res_temp = df_res_temp.append(ST_data[ST_data['数据时间'] == '2021-06-27'])

                if len(df_res_temp)==7*12*3:
                    #print(df_res_temp)
                    #print(df_res_temp.loc[:,['数据时间','星期_0','星期_1','星期_2','星期_3','星期_4','星期_5','星期_6']])
                    #print(df_res_temp.columns)

                    # 建立目录
                    if os.path.exists(dir_new) == False:
                        os.makedirs(dir_new)
                    df_res_temp.to_csv(path_or_buf=file_new, encoding="utf_8_sig",index=False)
                    #print(df_res)


                # print(df_res)
                # print(len(df_res))
                # print(df_res_temp)
                # print(len(df_res_temp))


