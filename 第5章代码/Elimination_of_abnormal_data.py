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
with open("D:\\实验记录\\聚类dbscan做异常检测\\exception_company.pk", 'rb') as f:
    data = pickle.load(f)
    print(data)
for root, dirs, filelist in os.walk("D:\\按地域划分数据集\\"):
        for i in filelist:
            if i in ["RT_data.csv","MT_data.csv","ST_data.csv"]:

                a = root.split("\\")[-2]
                b = root.split("\\")[-1]
                busname=a+"_"+b
                if busname in data[0] or busname in data[1]:
                    continue
                c = root.split("\\")[-3]
                link_new = "D:\\按地域划分数据集_去异常"+"\\"+c + "\\" + a + "\\" + b

                #D:\按地域划分数据集_去异常\嘉善县\嘉善县供电分公司2\188
                link_old=root    #D:\按地域划分数据集\余杭区\余杭区供电分公司3\18
                if  os.path.exists("D:\\按地域划分数据集_去异常" + "\\" + c + "\\" + a + "\\" + b):
                    continue
                # print(link_old)
                print(link_new)
                shutil.copytree(link_old, link_new)#递归复制文件夹和文件，如果文件夹存在报错

for root, dirs, filelist in os.walk("D:\\按行业划分数据集2\\"):
        for i in filelist:
            if i in ["RT_data.csv","MT_data.csv","ST_data.csv"]:

                a = root.split("\\")[-2]
                b = root.split("\\")[-1]
                busname=a+"_"+b
                if busname in data[0] or busname in data[1]:
                    continue
                c = root.split("\\")[-3]
                link_new = "D:\\按行业划分数据集_去异常"+"\\"+c + "\\" + a + "\\" + b

                #D:\按地域划分数据集_去异常\嘉善县\嘉善县供电分公司2\188
                link_old=root    #D:\按地域划分数据集\余杭区\余杭区供电分公司3\18
                if  os.path.exists("D:\\按行业划分数据集_去异常" + "\\" + c + "\\" + a + "\\" + b):
                    continue
                # print(link_old)
                print(link_new)
                shutil.copytree(link_old, link_new)#递归复制文件夹和文件，如果文件夹存在报错

# for root, dirs, filelist in os.walk("D:\\按地域划分数据集(方案2-对每个企业标准化)\\"):
#         for i in filelist:
#             if i in ["RT_data.csv","MT_data.csv","ST_data.csv"]:
#
#                 a = root.split("\\")[-2]
#                 b = root.split("\\")[-1]
#                 busname=a+"_"+b
#                 if busname in data[0] or busname in data[1]:
#                     continue
#                 c = root.split("\\")[-3]
#                 link_new = "D:\\按地域划分数据集(方案2-对每个企业标准化)_去异常"+"\\"+c + "\\" + a + "\\" + b
#
#                 #D:\按地域划分数据集_去异常\嘉善县\嘉善县供电分公司2\188
#                 link_old=root    #D:\按地域划分数据集\余杭区\余杭区供电分公司3\18
#                 if  os.path.exists("D:\\按地域划分数据集(方案2-对每个企业标准化)_去异常" + "\\" + c + "\\" + a + "\\" + b):
#                     continue
#                 # print(link_old)
#                 print(link_new)
#                 shutil.copytree(link_old, link_new)#递归复制文件夹和文件，如果文件夹存在报错