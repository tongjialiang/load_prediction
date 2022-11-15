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
disk="D:\\用电数据集\\Informer对比实验用的数据集\\"
files=os.listdir(disk)
for i in files:
    if str(i).startswith("mt"):
        for root, dirs, filelist in os.walk(disk+i):
            for j in filelist:
                if j == 'ST_data.csv':
                    #print(root)
                    print("remove1")
                    os.remove(root+"\\"+j)

for i in files:
    if str(i).startswith("st"):
        for root, dirs, filelist in os.walk(disk+i):
            for j in filelist:
                if j == 'MT_data.csv':
                    #print(root)
                    #print(j)
                    os.remove(root+"\\"+j)
                    print("remove2")