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
with open("c:\\实验记录\\聚类kmeans对数据集做分类-标准化\\25分类数据集\\kmeans_for_classification.pk", 'rb') as f:
    data = pickle.load(f)
    print(data)#'class6': array(['富阳区供电分公司2_15'], dtype='<U13')
#存放每个类的公司个数
# count_class={}
# for k in data.keys():
#     count_class.update({k:len(data[k])})
# print(count_class)
for root, dirs, filelist in os.walk("c:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集_特征增强\\"):
        for i in filelist:
            if i in ["RT_data.csv","MT_data.csv","ST_data.csv"]:
                a = root.split("\\")[-2]#167
                b = root.split("\\")[-1]#淳安县供电分公司2
                # c = root.split("\\")[-3]
                busname=a+"_"+b
                for j in data:
                    if busname in data[j]:#j class1
                        link_new = "c:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集(负荷特性)" + "\\"+j+"\\"  + a + "\\" + b
                        break
                print(link_new)
                link_old=root    #D:\按地域划分数据集\余杭区\余杭区供电分公司3\18
                # if  os.path.exists(link_new):
                #     #or count_class[j]<=10:#剔除公司数量小于10的类
                #     shutil.copyfile(link_old+"\\"+i, link_new+"\\"+i)
                # # print(link_old)
                # print(link_new)
                # shutil.copytree(link_old, link_new)#递归复制文件夹和文件，如果文件夹存在报错

                if os.path.exists(link_new) == False:
                    os.makedirs(link_new)
                shutil.copy(link_old+"\\"+i, link_new)


# for root, dirs, filelist in os.walk("D:\\按行业划分数据集(方案2-对每个企业标准化)\\"):
#         for i in filelist:
#             if i in ["RT_data.csv","MT_data.csv","ST_data.csv"]:
#
#                 a = root.split("\\")[-2]
#                 b = root.split("\\")[-1]
#                 busname=a+"_"+b
#                 if busname in data[0] or busname in data[1]:
#                     continue
#                 c = root.split("\\")[-3]
#                 link_new = "D:\\按行业划分数据集(方案2-对每个企业标准化)_去异常"+"\\"+c + "\\" + a + "\\" + b
#
#                 #D:\按地域划分数据集_去异常\嘉善县\嘉善县供电分公司2\188
#                 link_old=root    #D:\按地域划分数据集\余杭区\余杭区供电分公司3\18
#                 if  os.path.exists("D:\\按行业划分数据集(方案2-对每个企业标准化)_去异常" + "\\" + c + "\\" + a + "\\" + b):
#                     continue
#                 # print(link_old)
#                 print(link_new)
#                 shutil.copytree(link_old, link_new)#递归复制文件夹和文件，如果文件夹存在报错

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