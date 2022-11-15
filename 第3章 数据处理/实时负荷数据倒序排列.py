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


for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按地域划分数据集_去异常"):
        for i in filelist:
            if i in ["MT_data.csv","ST_data.csv"]:
                print(root)#D:\用电数据集\按地域划分数据集(方案1-汇总标准化)_去异常\萧山区\萧山区供电分公司5\9
                a = root.split("\\")[-2]
                b = root.split("\\")[-1]
                c = root.split("\\")[-3]
                link_new = "D:\\用电数据集\\归一化之前的数据集\\按地域划分数据集_去异常_ok"+"\\"+c + "\\" + a + "\\" + b+ "\\" +i
                dir_new="D:\\用电数据集\\归一化之前的数据集\\按地域划分数据集_去异常_ok"+"\\"+c + "\\" + a + "\\"+ b
                link_old=root+ "\\" +i
                if  not os.path.exists(dir_new):
                    os.makedirs(dir_new)
                # # print(link_old)
                # print(link_new)
                shutil.copyfile(link_old,link_new)
            if i in ["RT_data.csv"]:
                try:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                RT_data=RT_data.iloc[::-1]
                a = root.split("\\")[-2]
                b = root.split("\\")[-1]
                c = root.split("\\")[-3]
                link_new = "D:\\用电数据集\\归一化之前的数据集\\按地域划分数据集_去异常_ok" + "\\" + c + "\\" + a + "\\" + b + "\\" + i
                dir_new = "D:\\用电数据集\\归一化之前的数据集\\按地域划分数据集_去异常_ok" + "\\" + c + "\\" + a + "\\" + b
                if  not os.path.exists(dir_new):
                    os.makedirs(dir_new)
                RT_data.to_csv(path_or_buf=link_new, encoding="utf_8_sig", index=False)

# for root, dirs, filelist in os.walk("D:\\用电数据集\\按地域划分数据集(方案2-对每个企业标准化)_去异常\\"):
#         for i in filelist:
#             if i in ["MT_data.csv","ST_data.csv"]:
#                 print(root)#D:\用电数据集\按地域划分数据集(方案1-汇总标准化)_去异常\萧山区\萧山区供电分公司5\9
#                 a = root.split("\\")[-2]
#                 b = root.split("\\")[-1]
#                 c = root.split("\\")[-3]
#                 link_new = "D:\\用电数据集\\按地域划分数据集(方案2-对每个企业标准化)_ok"+"\\"+c + "\\" + a + "\\" + b+ "\\" +i
#                 dir_new="D:\\用电数据集\\按地域划分数据集(方案2-对每个企业标准化)_ok"+"\\"+c + "\\" + a + "\\"+ b
#                 link_old=root+ "\\" +i
#                 if  not os.path.exists(dir_new):
#                     os.makedirs(dir_new)
#                 # # print(link_old)
#                 # print(link_new)
#                 shutil.copyfile(link_old,link_new)
#             if i in ["RT_data.csv"]:
#                 try:
#                     RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
#                 except:
#                     RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
#                 RT_data=RT_data.iloc[::-1]
#                 a = root.split("\\")[-2]
#                 b = root.split("\\")[-1]
#                 c = root.split("\\")[-3]
#                 link_new = "D:\\用电数据集\\按地域划分数据集(方案2-对每个企业标准化)_ok" + "\\" + c + "\\" + a + "\\" + b + "\\" + i
#                 dir_new = "D:\\用电数据集\\按地域划分数据集(方案2-对每个企业标准化)_ok" + "\\" + c + "\\" + a + "\\" + b
#                 if  not os.path.exists(dir_new):
#                     os.makedirs(dir_new)
#                 RT_data.to_csv(path_or_buf=link_new, encoding="utf_8_sig", index=False)

for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_去异常"):
        for i in filelist:
            if i in ["MT_data.csv","ST_data.csv"]:
                print(root)#D:\用电数据集\按地域划分数据集(方案1-汇总标准化)_去异常\萧山区\萧山区供电分公司5\9
                a = root.split("\\")[-2]
                b = root.split("\\")[-1]
                c = root.split("\\")[-3]
                link_new = "D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_去异常_ok"+"\\"+c + "\\" + a + "\\" + b+ "\\" +i
                dir_new="D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_去异常_ok"+"\\"+c + "\\" + a + "\\"+ b
                link_old=root+ "\\" +i
                if  not os.path.exists(dir_new):
                    os.makedirs(dir_new)
                # # print(link_old)
                # print(link_new)
                shutil.copyfile(link_old,link_new)
            if i in ["RT_data.csv"]:
                try:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                RT_data=RT_data.iloc[::-1]
                a = root.split("\\")[-2]
                b = root.split("\\")[-1]
                c = root.split("\\")[-3]
                link_new = "D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_去异常_ok" + "\\" + c + "\\" + a + "\\" + b + "\\" + i
                dir_new = "D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_去异常_ok" + "\\" + c + "\\" + a + "\\" + b
                if  not os.path.exists(dir_new):
                    os.makedirs(dir_new)
                RT_data.to_csv(path_or_buf=link_new, encoding="utf_8_sig", index=False)
# for root, dirs, filelist in os.walk("D:\\用电数据集\\按聚类划分数据集(方案1-汇总标准化)_c25\\"):
#         for i in filelist:
#             if i in ["MT_data.csv","ST_data.csv"]:
#                 print(root)#D:\用电数据集\按地域划分数据集(方案1-汇总标准化)_去异常\萧山区\萧山区供电分公司5\9
#                 a = root.split("\\")[-2]
#                 b = root.split("\\")[-1]
#                 c = root.split("\\")[-3]
#                 link_new = "D:\\用电数据集\\按聚类划分数据集(方案1-汇总标准化)_c25_ok"+"\\"+c + "\\" + a + "\\" + b+ "\\" +i
#                 dir_new="D:\\用电数据集\\按聚类划分数据集(方案1-汇总标准化)_c25_ok"+"\\"+c + "\\" + a + "\\"+ b
#                 link_old=root+ "\\" +i
#                 if  not os.path.exists(dir_new):
#                     os.makedirs(dir_new)
#                 # # print(link_old)
#                 # print(link_new)
#                 shutil.copyfile(link_old,link_new)
#             if i in ["RT_data.csv"]:
#                 try:
#                     RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
#                 except:
#                     RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
#                 RT_data=RT_data.iloc[::-1]
#                 a = root.split("\\")[-2]
#                 b = root.split("\\")[-1]
#                 c = root.split("\\")[-3]
#                 link_new = "D:\\用电数据集\\按聚类划分数据集(方案1-汇总标准化)_c25_ok" + "\\" + c + "\\" + a + "\\" + b + "\\" + i
#                 dir_new = "D:\\用电数据集\\按聚类划分数据集(方案1-汇总标准化)_c25_ok" + "\\" + c + "\\" + a + "\\" + b
#                 if  not os.path.exists(dir_new):
#                     os.makedirs(dir_new)
#                 RT_data.to_csv(path_or_buf=link_new, encoding="utf_8_sig", index=False)
# for root, dirs, filelist in os.walk("D:\\用电数据集\\按聚类划分数据集(方案2-对每个企业标准化)_c3\\"):
#         for i in filelist:
#             if i in ["MT_data.csv","ST_data.csv"]:
#                 print(root)#D:\用电数据集\按地域划分数据集(方案1-汇总标准化)_去异常\萧山区\萧山区供电分公司5\9
#                 a = root.split("\\")[-2]
#                 b = root.split("\\")[-1]
#                 c = root.split("\\")[-3]
#                 link_new = "D:\\用电数据集\\按聚类划分数据集(方案2-对每个企业标准化)_c3_ok"+"\\"+c + "\\" + a + "\\" + b+ "\\" +i
#                 dir_new="D:\\用电数据集\\按聚类划分数据集(方案2-对每个企业标准化)_c3_ok"+"\\"+c + "\\" + a + "\\"+ b
#                 link_old=root+ "\\" +i
#                 if  not os.path.exists(dir_new):
#                     os.makedirs(dir_new)
#                 # # print(link_old)
#                 # print(link_new)
#                 shutil.copyfile(link_old,link_new)
#             if i in ["RT_data.csv"]:
#                 try:
#                     RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
#                 except:
#                     RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
#                 RT_data=RT_data.iloc[::-1]
#                 a = root.split("\\")[-2]
#                 b = root.split("\\")[-1]
#                 c = root.split("\\")[-3]
#                 link_new = "D:\\用电数据集\\按聚类划分数据集(方案2-对每个企业标准化)_c3_ok" + "\\" + c + "\\" + a + "\\" + b + "\\" + i
#                 dir_new = "D:\\用电数据集\\按聚类划分数据集(方案2-对每个企业标准化)_c3_ok" + "\\" + c + "\\" + a + "\\" + b
#                 if  not os.path.exists(dir_new):
#                     os.makedirs(dir_new)
#                 RT_data.to_csv(path_or_buf=link_new, encoding="utf_8_sig", index=False)
for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集_去异常"):
        for i in filelist:
            if i in ["MT_data.csv","ST_data.csv"]:
                print(root)#D:\用电数据集\按地域划分数据集(方案1-汇总标准化)_去异常\萧山区\萧山区供电分公司5\9
                a = root.split("\\")[-2]
                b = root.split("\\")[-1]
                c = root.split("\\")[-3]
                link_new = "D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集_去异常_ok"+"\\"+c + "\\" + a + "\\" + b+ "\\" +i
                dir_new="D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集_去异常_ok"+"\\"+c + "\\" + a + "\\"+ b
                link_old=root+ "\\" +i
                if  not os.path.exists(dir_new):
                    os.makedirs(dir_new)
                # # print(link_old)
                # print(link_new)
                shutil.copyfile(link_old,link_new)
            if i in ["RT_data.csv"]:
                try:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                RT_data=RT_data.iloc[::-1]
                a = root.split("\\")[-2]
                b = root.split("\\")[-1]
                c = root.split("\\")[-3]
                link_new = "D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集_去异常_ok" + "\\" + c + "\\" + a + "\\" + b + "\\" + i
                dir_new = "D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集_去异常_ok" + "\\" + c + "\\" + a + "\\" + b
                if  not os.path.exists(dir_new):
                    os.makedirs(dir_new)
                RT_data.to_csv(path_or_buf=link_new, encoding="utf_8_sig", index=False)

#
# for root, dirs, filelist in os.walk("D:\\用电数据集\\按行业划分数据集(方案2-对每个企业标准化)_去异常\\"):
#         for i in filelist:
#             if i in ["MT_data.csv","ST_data.csv"]:
#                 print(root)#D:\用电数据集\按地域划分数据集(方案1-汇总标准化)_去异常\萧山区\萧山区供电分公司5\9
#                 a = root.split("\\")[-2]
#                 b = root.split("\\")[-1]
#                 c = root.split("\\")[-3]
#                 link_new = "D:\\用电数据集\\按行业划分数据集(方案2-对每个企业标准化)_ok"+"\\"+c + "\\" + a + "\\" + b+ "\\" +i
#                 dir_new="D:\\用电数据集\\按行业划分数据集(方案2-对每个企业标准化)_ok"+"\\"+c + "\\" + a + "\\"+ b
#                 link_old=root+ "\\" +i
#                 if  not os.path.exists(dir_new):
#                     os.makedirs(dir_new)
#                 # # print(link_old)
#                 # print(link_new)
#                 shutil.copyfile(link_old,link_new)
#             if i in ["RT_data.csv"]:
#                 try:
#                     RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
#                 except:
#                     RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
#                 RT_data=RT_data.iloc[::-1]
#                 a = root.split("\\")[-2]
#                 b = root.split("\\")[-1]
#                 c = root.split("\\")[-3]
#                 link_new = "D:\\用电数据集\\按行业划分数据集(方案2-对每个企业标准化)_ok" + "\\" + c + "\\" + a + "\\" + b + "\\" + i
#                 dir_new = "D:\\用电数据集\\按行业划分数据集(方案2-对每个企业标准化)_ok" + "\\" + c + "\\" + a + "\\" + b
#                 if  not os.path.exists(dir_new):
#                     os.makedirs(dir_new)
#                 RT_data.to_csv(path_or_buf=link_new, encoding="utf_8_sig", index=False)
# for root, dirs, filelist in os.walk("D:\\用电数据集\\按聚类划分数据集(方案2-对每个企业标准化)_c25\\"):
#         for i in filelist:
#             if i in ["MT_data.csv","ST_data.csv"]:
#                 print(root)#D:\用电数据集\按地域划分数据集(方案1-汇总标准化)_去异常\萧山区\萧山区供电分公司5\9
#                 a = root.split("\\")[-2]
#                 b = root.split("\\")[-1]
#                 c = root.split("\\")[-3]
#                 link_new = "D:\\用电数据集\\test_ok"+"\\"+c + "\\" + a + "\\" + b+ "\\" +i
#                 dir_new="D:\\用电数据集\\test_ok"+"\\"+c + "\\" + a + "\\"+ b
#                 link_old=root+ "\\" +i
#                 if  not os.path.exists(dir_new):
#                     os.makedirs(dir_new)
#                 # # print(link_old)
#                 # print(link_new)
#                 shutil.copyfile(link_old,link_new)
#             if i in ["RT_data.csv"]:
#                 try:
#                     RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
#                 except:
#                     RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
#                 RT_data=RT_data.iloc[::-1]
#                 a = root.split("\\")[-2]
#                 b = root.split("\\")[-1]
#                 c = root.split("\\")[-3]
#                 link_new = "D:\\用电数据集\\按聚类划分数据集(方案2-对每个企业标准化)_c25_ok" + "\\" + c + "\\" + a + "\\" + b + "\\" + i
#                 dir_new = "D:\\用电数据集\\按聚类划分数据集(方案2-对每个企业标准化)_c25_ok" + "\\" + c + "\\" + a + "\\" + b
#                 if  not os.path.exists(dir_new):
#                     os.makedirs(dir_new)
#                 RT_data.to_csv(path_or_buf=link_new, encoding="utf_8_sig", index=False)