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
import gc
warnings.filterwarnings("ignore")
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

# with open('d://按行业划分数据集_方案2-对每个企业标准化_文化体育和娱乐业_MT_data_3_1.pk', 'rb') as f:
#     data1 = pickle.load(f)
#     print(data1[0].shape)
#     print(data1[1].shape)
#     print(data1[0][:6])
dir_input="D:\\数据采样完成\\"
dir_output="D:\\数据采样完成-完成供电容量归一化\\"
#处理MT的供电容量
def do_norm_power_supply_capacity(dir_input,dir_output):
    for root, dirs, filelist in os.walk(dir_input):
            for i in filelist:
                if not i.endswith('pk'):
                    continue
                if i.split("_")[-4] == 'MT':
                    print(root+i)
                    with open(root+i, 'rb') as f1:
                        data = pickle.load(f1)
                        print(data[0].shape)#(38095, 12, 11)
                        print(data[0][:,:,2])
                        capacity=data[0][:, :, 2]
                        print(capacity.shape)#(38095, 12)
                        #均值
                        # capacity_max=np.max(capacity)
                        capacity_mean = np.mean(capacity,axis=0)[0]
                        print(capacity_mean)
                        #print(capacity_mean)
                        #方差
                        capacity_std = np.std(capacity.astype(np.float),axis=0)[0]
                        print(capacity_std)
                        #print(capacity_std)

                        # capacity_sum = np.sum(capacity, axis=0)
                        # print(capacity_sum)
                        # for i,j in enumerate(range(len(capacity))):
                        #     if len(np.unique(capacity[j,:]))==3:
                        #         print(i)
                        #         print(len(np.unique(capacity[j,:])))
                        #         print(data[0][i])
                        # print(capacity_max)
                        # print(capacity_min)

                        if capacity_std!=0:
                            data[0][:, :, 2]= (data[0][:, :, 2]-capacity_mean)/capacity_std
                            print(data[0].shape)#(38095, 12, 11)
                        if capacity_std==0:
                            data[0][:, :, 2]= 0.00
                            print(data[0].shape)#(38095, 12, 11)
                        # print(data[0][:10])
                        with open(dir_output+i, 'wb+') as f:
                            pickle.dump(data, f)
                            f.close()
                            del data
                            gc.collect()
                        f1.close()
                        gc.collect()
#处理ST的供电容量
    for root, dirs, filelist in os.walk(dir_input):
            for i in filelist:
                if not i.endswith('pk'):
                    continue
                if i.split("_")[-4] == 'ST':
                    print(root+i)
                    with open(root+i, 'rb') as f1:
                        data = pickle.load(f1)

                        print(data[0].shape)#(38095, 12, 11)
                        # print(data[0][0])
                        #print(data[0][:,:,6])
                        capacity=data[0][:, :, 6]
                        #均值
                        capacity_mean = np.mean(capacity,axis=0)[0]
                        print(capacity_mean)
                        #方差
                        capacity_std = np.std(capacity.astype(np.float),axis=0)[0]
                        print(capacity_std)
                        print("___________")
                        if capacity_std!=0:
                            data[0][:, :, 6]= (data[0][:, :, 6]-capacity_mean)/capacity_std
                            print(data[0].shape)#(38095, 12, 11)
                        if capacity_std == 0:
                            data[0][:, :, 6] = 0.00
                            print(data[0].shape)  # (38095, 12, 11)

                        # print(data[0][:10])
                        with open(dir_output+i, 'wb+') as f:
                            pickle.dump(data, f)
                            f.close()
                            del data
                            gc.collect()
                        f1.close()
                        gc.collect()

do_norm_power_supply_capacity(dir_input,dir_output)


