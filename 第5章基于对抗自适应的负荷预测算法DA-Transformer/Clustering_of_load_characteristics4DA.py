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
from sklearn.cluster import KMeans

for root, dirs, filelist in os.walk("D:\\数据采样完成new\\"):
    for i in filelist:
        if i.endswith("pk"):
            #print(1)
            with open(root+i, 'rb') as f:
                data = pickle.load(f)
            f.close()
            #print(data[1].shape)
            #print(data[0].shape)#(38562, 96, 15)
            #print(data[0][:,:,3].shape)#电力负荷
            load_series=data[0][:, :, 3]
            load_series_y = data[1][:, :, 3]
            #print(load_series.shape)#(38562, 96)
            load_series_mean=np.mean(load_series, axis=1)
            load_series_mean_y = np.mean(load_series_y, axis=1)
            load_series_std = np.std(load_series.astype(np.float32),axis=1)
            load_series_std_y = np.std(load_series_y.astype(np.float32), axis=1)
            # print(load_series_mean.shape)#(38562,)
            # print(load_series_std.shape)#(38562,)
            x=np.array([load_series_mean,load_series_std]).T#(38562,2)
            x2 = np.array([load_series_mean_y, load_series_std_y]).T  # (38562,2)
            model=KMeans(n_clusters=2).fit(x)
            y_pred = model.predict(x)
            y_pred_y = model.predict(x2)
            #print(y_pred[:10])
            y_pred=np.expand_dims(y_pred, 1)
            y_pred=np.repeat(y_pred, data[0].shape[1], 1)
            y_pred = np.expand_dims(y_pred, 2)
            #print(y_pred.shape)#(38562, 96, 1)
            data[0]=np.append(data[0], y_pred,axis=2)
            y_pred_y=np.expand_dims(y_pred_y, 1)
            y_pred_y=np.repeat(y_pred_y, data[1].shape[1], 1)
            y_pred_y = np.expand_dims(y_pred_y, 2)
            #print(y_pred.shape)#(38562, 96, 1)
            data[1]=np.append(data[1], y_pred_y,axis=2)
            #print(data[2].shape)#(38562, 96, 16)
            with open('D:\\数据采样完成(负荷特性聚类)\\'+i, 'wb+') as f:
                pickle.dump(data, f)
            f.close()
