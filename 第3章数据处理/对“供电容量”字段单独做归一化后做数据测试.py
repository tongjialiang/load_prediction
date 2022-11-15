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

a="按行业划分数据集_方案2-对每个企业标准化_文化体育和娱乐业_MT_data_3_1.pk"
print(a)
with open('d://归一化以前//'+a, 'rb') as f:
    data1 = pickle.load(f)
    print(data1[0].shape)
    print(data1[1].shape)
    print(data1[0][0])

with open('d://归一化以后//'+a, 'rb') as f:
    data1 = pickle.load(f)
    print(data1[0].shape)
    print(data1[1].shape)
    print(data1[0][0])

a="按行业划分数据集_方案2-对每个企业标准化_文化体育和娱乐业_MT_data_12_3.pk"
print(a)
with open('d://归一化以前//'+a, 'rb') as f:
    data1 = pickle.load(f)
    print(data1[0].shape)
    print(data1[1].shape)
    print(data1[0][0])

with open('d://归一化以后//'+a, 'rb') as f:
    data1 = pickle.load(f)
    print(data1[0].shape)
    print(data1[1].shape)
    print(data1[0][0])

a="按行业划分数据集_方案2-对每个企业标准化_文化体育和娱乐业_ST_data_30_1.pk"
print(a)
with open('d://归一化以前//'+a, 'rb') as f:
    data1 = pickle.load(f)
    print(data1[0].shape)
    print(data1[1].shape)
    print(data1[0][0])

with open('d://归一化以后//'+a, 'rb') as f:
    data1 = pickle.load(f)
    print(data1[0].shape)
    print(data1[1].shape)
    print(data1[0][0])

a="按行业划分数据集_方案2-对每个企业标准化_文化体育和娱乐业_ST_data_30_7.pk"
print(a)
with open('d://归一化以前//'+a, 'rb') as f:
    data1 = pickle.load(f)
    print(data1[0].shape)
    print(data1[1].shape)
    print(data1[0][0])

with open('d://归一化以后//'+a, 'rb') as f:
    data1 = pickle.load(f)
    print(data1[0].shape)
    print(data1[1].shape)
    print(data1[0][0])
