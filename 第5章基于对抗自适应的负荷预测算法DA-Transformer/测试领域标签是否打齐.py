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
disk="D:\\用电数据集\\归一化之后的数据集-待采样\\按频域分解聚类划分数据集V2_领域自适应-汇总标准化\\"

for root, dirs, filelist in os.walk(disk):
    for i in filelist:
        if i in ['ST_data.csv','RT_data.csv','MT_data.csv']:
            try:
                data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            if "聚类领域类别" not in list(data.columns):
                print("bad")
            if "行业领域类别" not in list(data.columns):
                print("bad")
            if "地区领域类别" not in list(data.columns):
                print("bad")
