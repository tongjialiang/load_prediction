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
df_res = pd.DataFrame()
for root, dirs, filelist in os.walk("D:\\数据采样完成new\\"):#数据采样完成new
    for i in filelist:
        if i .endswith('pk'):
            delindex=[]#待删除的下标
            print(i)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_2020—2021每月取3周数据\交通运输仓储和邮政业\临安区供电分公司\26
            # dirnew='D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集_特征增强\\'+root.split('\\')[-3]+'\\'+root.split('\\')[-2]+'\\'+root.split('\\')[-1]+'\\'
            # filenew=dirnew+i
            with open(root+'\\'+i, 'rb') as f:
                data = pickle.load(f)
            f.close()
            # try:
            #     data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            # except:
            #     data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # ST_data['星期特征_遗传算法']=week_map_all
            # if os.path.exists(dirnew) == False:
            #     os.makedirs(dirnew)
            # ST_data.to_csv(path_or_buf=filenew, encoding="utf_8_sig",index=False)
            #print(data[1].shape)#(55, 6, 10) (2904, 96, 12) (1250, 70, 17)
            res = data[0][0, 0, :]
            #print(data[0].shape[0])
            lenx=data[0].shape[0]
            print('数据量',lenx)
            for k in range(lenx):
                #print(k)
                if k ==0:
                    continue
                #print(data[0][k,0,3:])#012
                if lenx>10000:
                    #print("big")
                    if k%(lenx//100)!=0:
                        continue
                res=np.vstack((res,data[0][k,0,:]))
            #np.vstack
            for j in range(res.shape[1]):
                if j in [0,1,2]:
                    continue
                #print(np.std(res[:,j]))
                if np.std(res[[0,-1,1,-2,2],j])>0.1:#为了提高性能，如果这些样本标准差比较大，不计算全量 的
                    continue
                if np.std(res[:,j])<0.01:
                    #print(j)
                    delindex.append(j)
            print(delindex)
            # print(len(delindex))
            # 1/0

            # print(data[0].shape)
            # print(data[1].shape)
            df_res = df_res.append([{'样本名称': str(i), '数据量':lenx, '删除的字段':str(delindex)}])
            if len(delindex)>0:
                data[0] = np.delete(data[0], delindex, axis=2)
                data[1] = np.delete(data[1], delindex, axis=2)
                with open('F:\\采样完成的数据集(去掉方差为0的数据)\\'+i, 'wb+') as f:
                    pickle.dump(data, f)
                f.close()
            del data
            #delindex=[]
            res=[]
            gc.collect()
# if os.path.exists(a) == False:
#     os.makedirs(a)
df_res.to_csv(path_or_buf='D:\\实验记录\\重要结果文件\\清除标准差为0的字段记录-行业再聚类(采样).csv', encoding="utf_8_sig", index=False)

            # 1/0
            # for d in range(data[0].shape[0]):
            #     np.delete(data[0], delindex, axis=1)
            #     print(data[0].shape)
            #     print(data[0][d])
            #     1/0
            #0000
