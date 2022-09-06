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
for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_去异常"):
        for i in filelist:
            if i in ["MT_data.csv","ST_data.csv","RT_data.csv"]:
                #print(root)#D:\用电数据集\按地域划分数据集(方案1-汇总标准化)_去异常\萧山区\萧山区供电分公司5\9
                a = root.split("\\")[-2]
                #print(a)#临安区供电分公司
                b = root.split("\\")[-1]
                #print(b)#1
                c = root.split("\\")[-3]
                #print(c)#临安区
                d = root.split("\\")[-4]
                #print(d)#按地域划分数据集_方案1-汇总标准化
                business_id=a+"_"+b
                print(business_id)

                try:
                    data_open = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    data_open = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                data_open["id"]=business_id

                link_new = "D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_去异常_ok\\"+c + "\\" + a + "\\" + b+ "\\" +i
                #print(link_new)#D:\用电数据集\归一化之后的数据集(除供电容量)-待采样2\按地域划分数据集_方案1-汇总标准化\临安区\临安区供电分公司\1\MT_data.csv
                dir_new="D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_去异常_ok\\"+c + "\\" + a + "\\"+ b
                #print(dir_new)#D:\用电数据集\归一化之后的数据集(除供电容量)-待采样2\按地域划分数据集_方案1-汇总标准化\临安区\临安区供电分公司\1

                link_old=root+ "\\" +i

                if  not os.path.exists(dir_new):
                    os.makedirs(dir_new)
                data_open.to_csv(path_or_buf=link_new, encoding="utf_8_sig", index=False)