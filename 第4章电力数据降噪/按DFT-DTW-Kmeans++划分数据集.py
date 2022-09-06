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

#按聚类划分数据集，根据聚类结果把RT文件存放到新的目录
with open("D:\\实验记录\\重要结果文件\\pk\\dtw聚类结果_rt_在行业划分的基础上_频域分解后分2类.pk", 'rb') as f:
    data = pickle.load(f)
    #print(data[0])#{'临安区供电分公司_1': 'rtclass_1', '临安区供电分公司_10': 'rtclass_1', '临安区供电分公司_100'
    f.close()


for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集_特征增强\\"):
        for i in filelist:
            link_new=0
            if i in ["RT_data.csv"]:
                a = root.split("\\")[-2]#167
                b = root.split("\\")[-1]#淳安县供电分公司2
                # c = root.split("\\")[-3]
                busname=a+"_"+b
                for j in data:
                    if j==busname:#j class1
                        link_new = "D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集(行业划分后聚类)" + "\\"+data[j]+"\\"  + a + "\\" + b
                        break
                print(link_new)
                if link_new==0:
                    print("error")
                    continue
                file_old=root+"\\"+i    #D:\按地域划分数据集\余杭区\余杭区供电分公司3\18
                if os.path.exists(link_new) == False:
                    os.makedirs(link_new)
                shutil.copy(file_old, link_new)


#按聚类划分数据集，根据聚类结果把ST,MT文件存放到新的目录
with open("D:\\实验记录\\重要结果文件\\pk\\dtw聚类结果_st_在行业划分的基础上_频域分解后分2类.pk", 'rb') as f:
    data = pickle.load(f)
    #print(data[0])#{'临安区供电分公司_1': 'rtclass_1', '临安区供电分公司_10': 'rtclass_1', '临安区供电分公司_100'
    f.close()

for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集_特征增强\\"):
        for i in filelist:
            link_new=0
            if i in ["ST_data.csv","MT_data.csv"]:
                a = root.split("\\")[-2]#167
                b = root.split("\\")[-1]#淳安县供电分公司2
                # c = root.split("\\")[-3]
                busname=a+"_"+b
                for j in data:
                    if j==busname:#j class1
                        link_new = "D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集(行业划分后聚类)" + "\\"+data[j]+"\\"  + a + "\\" + b
                        break
                print(link_new)
                if link_new==0:
                    print("error")
                    continue
                file_old=root+"\\"+i    #D:\按地域划分数据集\余杭区\余杭区供电分公司3\18
                if os.path.exists(link_new) == False:
                    os.makedirs(link_new)
                shutil.copy(file_old, link_new)