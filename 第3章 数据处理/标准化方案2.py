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
#读取均值和方差
with open('d:\\实验记录\\pk\\dict_BusTypeOrArea_MeanAndStd_method2.pk', 'rb') as f:
    res = pickle.load(f)
#对按地域分类的数据做全量标准化
for root, dirs, filelist in os.walk("D:\\按地域划分数据集_去异常\\"):
        for i in filelist:
            #filename=root+"\\"+i
            #print(filename)
            business_name = root.split("\\")[-2] + "_" + root.split("\\")[-1]

            if i == 'RT_data.csv':
                #print(i)#RT_data.csv
                #print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                #获取当前地区的均值和方差值
                area_type=root.split("\\")[-3]
                rt_mean=res[business_name]["rt_mean"]
                rt_std = res[business_name]["rt_std"]
                # print(rt_mean)
                # print(rt_std)
                RT_data.iloc[:,1]=(RT_data.iloc[:,1]-rt_mean)/rt_std
                RT_data["standard_key"] = business_name
                #存入新的目录
                link="D:\\按地域划分数据集(方案2-对每个企业标准化)_去异常\\"+area_type+"\\"+root.split("\\")[-2]+"\\"+root.split("\\")[-1]+"\\"
                print(link)
                if os.path.exists(link) == False:
                    os.makedirs(link)
                linknew=link+i
                RT_data.to_csv(path_or_buf=linknew, encoding="utf_8_sig", index=False)
            if i == 'MT_data.csv':

                # print(i)#RT_data.csv
                # print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    MT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    MT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                # 获取当前地区的均值和方差值
                area_type = root.split("\\")[-3]
                print(business_name)
                print(res[business_name])
                mt_mean = res[business_name]["mt_mean"]
                mt_std = res[business_name]["mt_std"]
                mt_mean_max = res[business_name]["mt_mean_max"]
                mt_std_max = res[business_name]["mt_std_max"]
                mt_mean_min = res[business_name]["mt_mean_min"]
                mt_std_min = res[business_name]["mt_std_min"]
                # print(mt_mean)
                # print(mt_std)
                # print(mt_mean_max)
                # print(mt_std_max)
                # print(mt_mean_min)
                # print(mt_std_min)

                MT_data.iloc[:, 1] = (MT_data.iloc[:, 1] - mt_mean) / mt_std
                MT_data.iloc[:, 8] = (MT_data.iloc[:, 8] - mt_mean_max) / mt_std_max
                MT_data.iloc[:, 9] = (MT_data.iloc[:, 9] - mt_mean_min) / mt_std_min
                MT_data["standard_key"] = business_name
                # 存入新的目录
                link = "D:\\按地域划分数据集(方案2-对每个企业标准化)_去异常\\" + area_type + "\\" + root.split("\\")[-2] + "\\" + root.split("\\")[
                    -1] + "\\"
                print(link)
                if os.path.exists(link) == False:
                    os.makedirs(link)
                linknew = link + i
                MT_data.to_csv(path_or_buf=linknew, encoding="utf_8_sig", index=False)

            if i == 'ST_data.csv':
                # print(i)#RT_data.csv
                # print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                # 获取当前地区的均值和方差值
                area_type = root.split("\\")[-3]
                st_mean = res[business_name]["st_mean"]
                st_std = res[business_name]["st_std"]
                st_mean_max = res[business_name]["st_mean_max"]
                st_std_max = res[business_name]["st_std_max"]
                st_mean_min = res[business_name]["st_mean_min"]
                st_std_min = res[business_name]["st_std_min"]
                print(st_mean)
                print(st_std)
                print(st_mean_max)
                print(st_std_max)
                print(st_mean_min)
                print(st_std_min)

                ST_data.iloc[:, 1] = (ST_data.iloc[:, 1] - st_mean) / st_std
                ST_data.iloc[:, 2] = (ST_data.iloc[:, 2] - st_mean_max) / st_std_max
                ST_data.iloc[:, 3] = (ST_data.iloc[:, 3] - st_mean_min) / st_std_min
                ST_data["standard_key"] = business_name
                # 存入新的目录
                link = "D:\\按地域划分数据集(方案2-对每个企业标准化)_去异常\\" + area_type + "\\" + root.split("\\")[-2] + "\\" + root.split("\\")[
                    -1] + "\\"
                print(link)
                if os.path.exists(link) == False:
                    os.makedirs(link)
                linknew = link + i
                ST_data.to_csv(path_or_buf=linknew, encoding="utf_8_sig", index=False)
                # res = {"rt_mean": rt_mean, "rt_std": rt_std, "mt_mean": mt_mean, "mt_std": mt_std,
                #        "mt_mean_max": mt_mean_max,
                #        "mt_std_max": mt_std_max, "mt_mean_min": mt_mean_min, "mt_std_min": mt_std_min,
                #        "st_mean": st_mean,
                #        "st_std": st_std, "st_mean_max": st_mean_max, "st_std_max": st_std_max,
                #        "st_mean_min": st_mean_min
                #     , "st_std_min": st_std_min}








#对按企业类型分类的数据做全量标准化
for root, dirs, filelist in os.walk("D:\\按行业划分数据集_去异常\\"):
        for i in filelist:
            #filename=root+"\\"+i
            #print(filename)
            business_name = root.split("\\")[-2] + "_" + root.split("\\")[-1]
            if i == 'RT_data.csv':
                #print(i)#RT_data.csv
                #print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                #获取当前地区的均值和方差值
                area_type=root.split("\\")[-3]
                rt_mean=res[business_name]["rt_mean"]
                rt_std = res[business_name]["rt_std"]
                # print(rt_mean)
                # print(rt_std)
                RT_data.iloc[:,1]=(RT_data.iloc[:,1]-rt_mean)/rt_std
                RT_data["standard_key"] = business_name
                #存入新的目录
                link="D:\\按行业划分数据集(方案2-对每个企业标准化)_去异常\\"+area_type+"\\"+root.split("\\")[-2]+"\\"+root.split("\\")[-1]+"\\"
                print(link)
                if os.path.exists(link) == False:
                    os.makedirs(link)
                linknew=link+i
                RT_data.to_csv(path_or_buf=linknew, encoding="utf_8_sig", index=False)
            if i == 'MT_data.csv':

                # print(i)#RT_data.csv
                # print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    MT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    MT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                # 获取当前地区的均值和方差值
                area_type = root.split("\\")[-3]
                mt_mean = res[business_name]["mt_mean"]
                mt_std = res[business_name]["mt_std"]
                mt_mean_max = res[business_name]["mt_mean_max"]
                mt_std_max = res[business_name]["mt_std_max"]
                mt_mean_min = res[business_name]["mt_mean_min"]
                mt_std_min = res[business_name]["mt_std_min"]
                # print(mt_mean)
                # print(mt_std)
                # print(mt_mean_max)
                # print(mt_std_max)
                # print(mt_mean_min)
                # print(mt_std_min)

                MT_data.iloc[:, 1] = (MT_data.iloc[:, 1] - mt_mean) / mt_std
                MT_data.iloc[:, 8] = (MT_data.iloc[:, 8] - mt_mean_max) / mt_std_max
                MT_data.iloc[:, 9] = (MT_data.iloc[:, 9] - mt_mean_min) / mt_std_min
                MT_data["standard_key"] = business_name
                # 存入新的目录
                link = "D:\\按行业划分数据集(方案2-对每个企业标准化)_去异常\\" + area_type + "\\" + root.split("\\")[-2] + "\\" + root.split("\\")[
                    -1] + "\\"
                print(link)
                if os.path.exists(link) == False:
                    os.makedirs(link)
                linknew = link + i
                MT_data.to_csv(path_or_buf=linknew, encoding="utf_8_sig", index=False)

            if i == 'ST_data.csv':
                # print(i)#RT_data.csv
                # print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                # 获取当前地区的均值和方差值
                area_type = root.split("\\")[-3]
                st_mean = res[business_name]["st_mean"]
                st_std = res[business_name]["st_std"]
                st_mean_max = res[business_name]["st_mean_max"]
                st_std_max = res[business_name]["st_std_max"]
                st_mean_min = res[business_name]["st_mean_min"]
                st_std_min = res[business_name]["st_std_min"]
                print(st_mean)
                print(st_std)
                print(st_mean_max)
                print(st_std_max)
                print(st_mean_min)
                print(st_std_min)

                ST_data.iloc[:, 1] = (ST_data.iloc[:, 1] - st_mean) / st_std
                ST_data.iloc[:, 2] = (ST_data.iloc[:, 2] - st_mean_max) / st_std_max
                ST_data.iloc[:, 3] = (ST_data.iloc[:, 3] - st_mean_min) / st_std_min
                ST_data["standard_key"] = business_name
                # 存入新的目录
                link = "D:\\按行业划分数据集(方案2-对每个企业标准化)_去异常\\" + area_type + "\\" + root.split("\\")[-2] + "\\" + root.split("\\")[
                    -1] + "\\"
                print(link)
                if os.path.exists(link) == False:
                    os.makedirs(link)
                linknew = link + i
                ST_data.to_csv(path_or_buf=linknew, encoding="utf_8_sig", index=False)