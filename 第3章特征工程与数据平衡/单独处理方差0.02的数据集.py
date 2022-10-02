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
from ShowapiRequest import ShowapiRequest
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
warnings.filterwarnings("ignore")
df_res = pd.DataFrame()
for root, dirs, filelist in os.walk("D:\用电数据集\归一化之后的数据集-待采样\\按频域分解聚类划分数据集V2_方案1-汇总标准化"):
        for i in filelist:
            if i in ['ST_data.csv','MT_data.csv']:
                # print(root)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_2020—2021每月取3周数据\交通运输仓储和邮政业\临安区供电分公司\26
                # dirnew='D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_特征增强_去除问题数据\\'+root.split('\\')[-3]+'\\'+root.split('\\')[-2]+'\\'+root.split('\\')[-1]+'\\'
                try:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                variance=math.pow(np.std(ST_data['平均负荷(kW)']),2)
                if variance<0.02:
                    path=(root+'\\'+i)
                    id=root.split('\\')[-2]+'_'+root.split('\\')[-1]
                    print(id)#id
                    mean=np.mean(ST_data['平均负荷(kW)'])
                    df_res=df_res.append([{'id':id,'file':i,'方差':variance,'预测值':mean}])
                    os.remove(path)
                # if np.mean(ST_data['平均负荷(kW)'])<1:
                #     print(root+'//'+i)
                #     print(np.std(ST_data['平均负荷(kW)']))
            if i in ['RT_data.csv']:
                # print(root)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_2020—2021每月取3周数据\交通运输仓储和邮政业\临安区供电分公司\26
                # dirnew='D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_特征增强_去除问题数据\\'+root.split('\\')[-3]+'\\'+root.split('\\')[-2]+'\\'+root.split('\\')[-1]+'\\'

                try:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                variance=math.pow(np.std(RT_data['瞬时有功(kW)']),2)
                if variance<0.02:
                    path=root + '\\' + i
                    id=root.split('\\')[-2] + '_' + root.split('\\')[-1]
                    print(id)
                    mean = np.mean(RT_data['瞬时有功(kW)'])
                    df_res=df_res.append([{'id':id,'file':i,'方差':variance,'预测值':mean}])
                    os.remove(path)
df_res.to_csv(path_or_buf='D:\\实验记录\\实验结果分析\\方差极小数据的预测_按频域分解聚类划分数据集.csv', encoding="utf_8_sig",index=False)
                #     if os.path.exists(dirnew) == False:
                #     os.makedirs(dirnew)
                # ST_data.to_csv(path_or_buf=filenew, encoding="utf_8_sig",index=False)