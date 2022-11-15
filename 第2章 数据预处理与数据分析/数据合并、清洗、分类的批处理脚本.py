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
dir="D:\数据预处理完成后的样本"
for root, dirs, filelist in os.walk(dir):
    for i in filelist:
        print(root+"\\"+i)
        print(i.split("-")[-1])
        # 处理RT_data_48_4.pk
        if i.split("-")[-1]=="RT_data_48_4.pk":
            with open(root+"\\"+i, 'rb') as f:
                data = pickle.load(f)
                #print(data[0].shape)#(8953, 48, 4) 第一维：样本数量 第二维：一个样本包含几个时间点数据 第三维：特征数量
                #print(data[1].shape)#(8953, 4, 4)  第一维：标签数量 第二维：一个标签包含几个时间点数据 第三维：特征数量
                x=data[0][:,:,[1]]
                #print(x.shape)#(8953, 48, 1)
                xshape=x.shape
                x=x.reshape(len(x),-1)
                y=data[1][:,:,[1]]
                #print(y.shape)#(8953, 4, 1)
                scalerx = StandardScaler()

                scalerx.fit(x)

                x=scalerx.transform(x)
                x=x.reshape(xshape)
                print(x)
        #处理RT_data_48_1.pk
        #处理RT_data_48_12.pk
        #处理ST_data_30_1.pk
        #处理ST_data_30_7.pk
        #处理MT_data_3_1.pk
        #处理MT_data_12_3.pk
        #处理MT_data_3_1.pk