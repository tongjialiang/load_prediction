#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
from sklearn.preprocessing import StandardScaler
import math
import os
import shutil
import datetime
import torch
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
import MMD
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
res=[]
for root, dirs, filelist in os.walk("D:\\数据采样完成new\\"):
    for i in filelist:
        if i.endswith("pk"):
            with open(root+i, 'rb') as f:
                data = pickle.load(f)
            f.close()
            print(i)
            #print(data[0][:5000].shape)#(5000, 96, 15)
            data=data[0][:5000]
            domain1=data[data[:,0,-1]==1][:,:,3].astype(np.float32)
            domain2=data[data[:,0,-1]==0][:,:,3].astype(np.float32)
            domain1=torch.from_numpy(domain1)
            domain2=torch.from_numpy(domain2)

            mmd_score = 0
            if len(domain1) == 0 or len(domain2) == 0:
                pass
            elif len(domain2) <= len(domain1):
                domain1 = domain1[:len(domain2)]
                print(domain1.shape)
                print(domain2.shape)
                mmd_score=MMD.mmd_rbf(domain1, domain2)
            else:
                domain2 = domain2[:len(domain1)]
                print(domain1.shape)
                print(domain2.shape)
                mmd_score=MMD.mmd_rbf(domain1, domain2)
            print("mmd_score",mmd_score)

            res.append(mmd_score)
print(res)