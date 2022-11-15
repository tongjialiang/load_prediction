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
tree_parameter={}
try:
    tree_data = pd.read_csv("D:\\实验记录\\实验结果\\DecisionTreeRegressor.csv", encoding='utf-8', sep=',')
except:
    tree_data = pd.read_csv("D:\\实验记录\\实验结果\\DecisionTreeRegressor.csv", encoding='gbk', sep=',')

for i in range(len(tree_data["文件名"])):
    tree_parameter.update({tree_data["文件名"][i]:tree_data["最佳参数"][i]})
with open('D:\\实验记录\pk\\tree_parameter_dict.pk', 'wb+') as f:
    pickle.dump(tree_parameter, f)
f.close()