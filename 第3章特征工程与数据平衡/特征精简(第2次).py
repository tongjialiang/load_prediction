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
from ShowapiRequest import ShowapiRequest
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
warnings.filterwarnings("ignore")

for root, dirs, filelist in os.walk("aD:\\用电数据集\\归一化之后的数据集-待采样\\按地域划分数据集V2_方案1-汇总标准化\\"):
    for i in filelist:
        if i == 'ST_data.csv':
            try:
                ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(ST_data.columns)
            # 1/0
            if ST_data.columns.__contains__("月份_1"):####
                del ST_data['月份_1']
            if ST_data.columns.__contains__("月份_2"):####
                del ST_data['月份_2']
            if ST_data.columns.__contains__("year_2016"):#####
                del ST_data['year_2016']
            if ST_data.columns.__contains__("year_2017"):####
                del ST_data['year_2017']
            if ST_data.columns.__contains__("year_2018"):####
                del ST_data['year_2018']
            if ST_data.columns.__contains__("year_2019"):#####
                del ST_data['year_2019']
            if ST_data.columns.__contains__("year_2020"):####
                del ST_data['year_2020']
            if ST_data.columns.__contains__("year_2021"):####
                del ST_data['year_2021']
            if ST_data.columns.__contains__("能源生产比上年增长(%)"):
                del ST_data['能源生产比上年增长(%)']
            if ST_data.columns.__contains__("能源生产弹性系数"):
                del ST_data['能源生产弹性系数']
            if ST_data.columns.__contains__("全省能源消费量(百万吨标准煤)"):
                del ST_data['全省能源消费量(百万吨标准煤)']
            if ST_data.columns.__contains__("全省电力消费量(百亿千瓦小时)"):
                del ST_data['全省电力消费量(百亿千瓦小时)']
            if MT_data.columns.__contains__("按地区分组的法人单位数(万人)_p35"):
                del MT_data['按地区分组的法人单位数(万人)_p35']
            if ST_data.columns.__contains__("能源消费弹性系数"):
                del ST_data['能源消费弹性系数']
            if ST_data.columns.__contains__("电力消费弹性系数"):
                del ST_data['电力消费弹性系数']
            if ST_data.columns.__contains__("各市规模以上企业年末单位就业人员(万人)_p72"):
                del ST_data['各市规模以上企业年末单位就业人员(万人)_p72']
            if ST_data.columns.__contains__("生产总值(百亿元)_p518_p539"):
                del ST_data['生产总值(百亿元)_p518_p539']
            if ST_data.columns.__contains__("第一产业(百亿元)_p537_p539"):
                del ST_data['第一产业(百亿元)_p537_p539']
            if ST_data.columns.__contains__("第二产业(百亿元)_p537_p539"):
                del ST_data['第二产业(百亿元)_p537_p539']
            if ST_data.columns.__contains__("第三产业(百亿元)_p537_p539"):
                del ST_data['第三产业(百亿元)_p537_p539']
            if ST_data.columns.__contains__("分行业全社会单位就业人员年平均工资_p163"):
                del ST_data['分行业全社会单位就业人员年平均工资_p163']
            ST_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)
        if i == 'MT_data.csv':
            try:
                MT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                MT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(MT_data.columns)
            # 1/0
            if MT_data.columns.__contains__("月份_1"):
                del MT_data['月份_1']
            if MT_data.columns.__contains__("月份_2"):
                del MT_data['月份_2']
            if MT_data.columns.__contains__("year_2016"):
                del MT_data['year_2016']
            if MT_data.columns.__contains__("year_2017"):
                del MT_data['year_2017']
            if MT_data.columns.__contains__("year_2018"):
                del MT_data['year_2018']
            if MT_data.columns.__contains__("year_2019"):
                del MT_data['year_2019']
            if MT_data.columns.__contains__("year_2020"):
                del MT_data['year_2020']
            if MT_data.columns.__contains__("year_2021"):
                del MT_data['year_2021']
            if MT_data.columns.__contains__("能源生产比上年增长(%)"):
                del MT_data['能源生产比上年增长(%)']
            if MT_data.columns.__contains__("能源生产弹性系数"):
                del MT_data['能源生产弹性系数']
            if MT_data.columns.__contains__("全省能源消费量(百万吨标准煤)"):
                del MT_data['全省能源消费量(百万吨标准煤)']
            if MT_data.columns.__contains__("全省电力消费量(百亿千瓦小时)"):
                del MT_data['全省电力消费量(百亿千瓦小时)']
            if MT_data.columns.__contains__("能源消费弹性系数"):
                del MT_data['能源消费弹性系数']
            if MT_data.columns.__contains__("电力消费弹性系数"):
                del MT_data['电力消费弹性系数']
            if MT_data.columns.__contains__("按地区分组的法人单位数(万人)_p35"):
                del MT_data['按地区分组的法人单位数(万人)_p35']
            if MT_data.columns.__contains__("各市规模以上企业年末单位就业人员(万人)_p72"):
                del MT_data['各市规模以上企业年末单位就业人员(万人)_p72']
            if MT_data.columns.__contains__("生产总值(百亿元)_p518_p539"):
                del MT_data['生产总值(百亿元)_p518_p539']
            if MT_data.columns.__contains__("第一产业(百亿元)_p537_p539"):
                del MT_data['第一产业(百亿元)_p537_p539']
            if MT_data.columns.__contains__("第二产业(百亿元)_p537_p539"):
                del MT_data['第二产业(百亿元)_p537_p539']
            if MT_data.columns.__contains__("第三产业(百亿元)_p537_p539"):
                del MT_data['第三产业(百亿元)_p537_p539']
            if MT_data.columns.__contains__("分行业全社会单位就业人员年平均工资_p163"):
                del MT_data['分行业全社会单位就业人员年平均工资_p163']
            MT_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)

        if i == 'RT_data.csv':
            try:
                RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(RT_data.columns)
            # 1/0
            if RT_data.columns.__contains__("月份_1"):
                del RT_data['月份_1']
            if RT_data.columns.__contains__("月份_2"):
                del RT_data['月份_2']
            if RT_data.columns.__contains__("year_2016"):
                del RT_data['year_2016']
            if RT_data.columns.__contains__("year_2017"):
                del RT_data['year_2017']
            if RT_data.columns.__contains__("year_2018"):
                del RT_data['year_2018']
            if RT_data.columns.__contains__("year_2019"):
                del RT_data['year_2019']
            if RT_data.columns.__contains__("year_2020"):
                del RT_data['year_2020']
            if RT_data.columns.__contains__("year_2021"):
                del RT_data['year_2021']
            if RT_data.columns.__contains__("能源生产比上年增长(%)"):
                del RT_data['能源生产比上年增长(%)']
            if RT_data.columns.__contains__("能源生产弹性系数"):
                del RT_data['能源生产弹性系数']
            if RT_data.columns.__contains__("全省能源消费量(百万吨标准煤)"):
                del RT_data['全省能源消费量(百万吨标准煤)']
            if RT_data.columns.__contains__("全省电力消费量(百亿千瓦小时)"):
                del RT_data['全省电力消费量(百亿千瓦小时)']
            if RT_data.columns.__contains__("能源消费弹性系数"):
                del RT_data['能源消费弹性系数']
            if RT_data.columns.__contains__("电力消费弹性系数"):
                del RT_data['电力消费弹性系数']
            if RT_data.columns.__contains__("按地区分组的法人单位数(万人)_p35"):
                del RT_data['按地区分组的法人单位数(万人)_p35']
            if RT_data.columns.__contains__("各市规模以上企业年末单位就业人员(万人)_p72"):
                del RT_data['各市规模以上企业年末单位就业人员(万人)_p72']
            if RT_data.columns.__contains__("生产总值(百亿元)_p518_p539"):
                del RT_data['生产总值(百亿元)_p518_p539']
            if RT_data.columns.__contains__("第一产业(百亿元)_p537_p539"):
                del RT_data['第一产业(百亿元)_p537_p539']
            if RT_data.columns.__contains__("第二产业(百亿元)_p537_p539"):
                del RT_data['第二产业(百亿元)_p537_p539']
            if RT_data.columns.__contains__("第三产业(百亿元)_p537_p539"):
                del RT_data['第三产业(百亿元)_p537_p539']
            if RT_data.columns.__contains__("分行业全社会单位就业人员年平均工资_p163"):
                del RT_data['分行业全社会单位就业人员年平均工资_p163']
            RT_data.to_csv(path_or_buf=root+'\\'+i, encoding="utf_8_sig", index=False)
#
#
for root, dirs, filelist in os.walk("aD:\\用电数据集\\归一化之后的数据集-待采样\\按行业划分数据集V2_方案1-汇总标准化\\"):
    for i in filelist:
        if i == 'ST_data.csv':
            try:
                ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(ST_data.columns)
            # 1/0
            if ST_data.columns.__contains__("月份_1"):
                del ST_data['月份_1']
            if ST_data.columns.__contains__("月份_2"):
                del ST_data['月份_2']
            if ST_data.columns.__contains__("year_2016"):
                del ST_data['year_2016']
            if ST_data.columns.__contains__("year_2017"):
                del ST_data['year_2017']
            if ST_data.columns.__contains__("year_2018"):
                del ST_data['year_2018']
            if ST_data.columns.__contains__("year_2019"):
                del ST_data['year_2019']
            if ST_data.columns.__contains__("year_2020"):
                del ST_data['year_2020']
            if ST_data.columns.__contains__("year_2021"):
                del ST_data['year_2021']
            if ST_data.columns.__contains__("能源生产比上年增长(%)"):
                del ST_data['能源生产比上年增长(%)']
            if ST_data.columns.__contains__("能源生产弹性系数"):
                del ST_data['能源生产弹性系数']
            if ST_data.columns.__contains__("全省能源消费量(百万吨标准煤)"):
                del ST_data['全省能源消费量(百万吨标准煤)']
            if ST_data.columns.__contains__("全省电力消费量(百亿千瓦小时)"):
                del ST_data['全省电力消费量(百亿千瓦小时)']
            if ST_data.columns.__contains__("能源消费弹性系数"):
                del ST_data['能源消费弹性系数']
            if ST_data.columns.__contains__("电力消费弹性系数"):
                del ST_data['电力消费弹性系数']
            if ST_data.columns.__contains__("按地区分组的法人单位数(万人)_p35"):
                del ST_data['按地区分组的法人单位数(万人)_p35']
            if ST_data.columns.__contains__("各市规模以上企业年末单位就业人员(万人)_p72"):
                del ST_data['各市规模以上企业年末单位就业人员(万人)_p72']
            if ST_data.columns.__contains__("生产总值(百亿元)_p518_p539"):
                del ST_data['生产总值(百亿元)_p518_p539']
            if ST_data.columns.__contains__("第一产业(百亿元)_p537_p539"):
                del ST_data['第一产业(百亿元)_p537_p539']
            if ST_data.columns.__contains__("第二产业(百亿元)_p537_p539"):
                del ST_data['第二产业(百亿元)_p537_p539']
            if ST_data.columns.__contains__("第三产业(百亿元)_p537_p539"):
                del ST_data['第三产业(百亿元)_p537_p539']
            if ST_data.columns.__contains__("分行业全社会单位就业人员年平均工资_p163"):
                del ST_data['分行业全社会单位就业人员年平均工资_p163']
            ST_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)
        if i == 'MT_data.csv':
            try:
                MT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                MT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(MT_data.columns)
            # 1/0
            if MT_data.columns.__contains__("月份_1"):
                del MT_data['月份_1']
            if MT_data.columns.__contains__("月份_2"):
                del MT_data['月份_2']
            if MT_data.columns.__contains__("year_2016"):
                del MT_data['year_2016']
            if MT_data.columns.__contains__("year_2017"):
                del MT_data['year_2017']
            if MT_data.columns.__contains__("year_2018"):
                del MT_data['year_2018']
            if MT_data.columns.__contains__("year_2019"):
                del MT_data['year_2019']
            if MT_data.columns.__contains__("year_2020"):
                del MT_data['year_2020']
            if MT_data.columns.__contains__("year_2021"):
                del MT_data['year_2021']
            if MT_data.columns.__contains__("能源生产比上年增长(%)"):
                del MT_data['能源生产比上年增长(%)']
            if MT_data.columns.__contains__("能源生产弹性系数"):
                del MT_data['能源生产弹性系数']
            if MT_data.columns.__contains__("全省能源消费量(百万吨标准煤)"):
                del MT_data['全省能源消费量(百万吨标准煤)']
            if MT_data.columns.__contains__("全省电力消费量(百亿千瓦小时)"):
                del MT_data['全省电力消费量(百亿千瓦小时)']
            if MT_data.columns.__contains__("能源消费弹性系数"):
                del MT_data['能源消费弹性系数']
            if MT_data.columns.__contains__("电力消费弹性系数"):
                del MT_data['电力消费弹性系数']
            if MT_data.columns.__contains__("按地区分组的法人单位数(万人)_p35"):
                del MT_data['按地区分组的法人单位数(万人)_p35']
            if MT_data.columns.__contains__("各市规模以上企业年末单位就业人员(万人)_p72"):
                del MT_data['各市规模以上企业年末单位就业人员(万人)_p72']
            if MT_data.columns.__contains__("生产总值(百亿元)_p518_p539"):
                del MT_data['生产总值(百亿元)_p518_p539']
            if MT_data.columns.__contains__("第一产业(百亿元)_p537_p539"):
                del MT_data['第一产业(百亿元)_p537_p539']
            if MT_data.columns.__contains__("第二产业(百亿元)_p537_p539"):
                del MT_data['第二产业(百亿元)_p537_p539']
            if MT_data.columns.__contains__("第三产业(百亿元)_p537_p539"):
                del MT_data['第三产业(百亿元)_p537_p539']
            if MT_data.columns.__contains__("分行业全社会单位就业人员年平均工资_p163"):
                del MT_data['分行业全社会单位就业人员年平均工资_p163']
            MT_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)

        if i == 'RT_data.csv':
            try:
                RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(RT_data.columns)
            # 1/0
            if RT_data.columns.__contains__("月份_1"):
                del RT_data['月份_1']
            if RT_data.columns.__contains__("月份_2"):
                del RT_data['月份_2']
            if RT_data.columns.__contains__("year_2016"):
                del RT_data['year_2016']
            if RT_data.columns.__contains__("year_2017"):
                del RT_data['year_2017']
            if RT_data.columns.__contains__("year_2018"):
                del RT_data['year_2018']
            if RT_data.columns.__contains__("year_2019"):
                del RT_data['year_2019']
            if RT_data.columns.__contains__("year_2020"):
                del RT_data['year_2020']
            if RT_data.columns.__contains__("year_2021"):
                del RT_data['year_2021']
            if RT_data.columns.__contains__("能源生产比上年增长(%)"):
                del RT_data['能源生产比上年增长(%)']
            if RT_data.columns.__contains__("能源生产弹性系数"):
                del RT_data['能源生产弹性系数']
            if RT_data.columns.__contains__("全省能源消费量(百万吨标准煤)"):
                del RT_data['全省能源消费量(百万吨标准煤)']
            if RT_data.columns.__contains__("全省电力消费量(百亿千瓦小时)"):
                del RT_data['全省电力消费量(百亿千瓦小时)']
            if RT_data.columns.__contains__("能源消费弹性系数"):
                del RT_data['能源消费弹性系数']
            if RT_data.columns.__contains__("电力消费弹性系数"):
                del RT_data['电力消费弹性系数']
            if RT_data.columns.__contains__("按地区分组的法人单位数(万人)_p35"):
                del RT_data['按地区分组的法人单位数(万人)_p35']
            if RT_data.columns.__contains__("各市规模以上企业年末单位就业人员(万人)_p72"):
                del RT_data['各市规模以上企业年末单位就业人员(万人)_p72']
            if RT_data.columns.__contains__("生产总值(百亿元)_p518_p539"):
                del RT_data['生产总值(百亿元)_p518_p539']
            if RT_data.columns.__contains__("第一产业(百亿元)_p537_p539"):
                del RT_data['第一产业(百亿元)_p537_p539']
            if RT_data.columns.__contains__("第二产业(百亿元)_p537_p539"):
                del RT_data['第二产业(百亿元)_p537_p539']
            if RT_data.columns.__contains__("第三产业(百亿元)_p537_p539"):
                del RT_data['第三产业(百亿元)_p537_p539']
            if RT_data.columns.__contains__("分行业全社会单位就业人员年平均工资_p163"):
                del RT_data['分行业全社会单位就业人员年平均工资_p163']
            RT_data.to_csv(path_or_buf=root+'\\'+i, encoding="utf_8_sig", index=False)
#
for root, dirs, filelist in os.walk("D:\\a用电数据集\\归一化之后的数据集-待采样\\按聚类划分数据集V2_方案2-行业划分后聚类\\"):
    for i in filelist:
        if i == 'ST_data.csv':
            try:
                ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            if ST_data.columns.__contains__("月份_1"):
                del ST_data['月份_1']
            if ST_data.columns.__contains__("月份_2"):
                del ST_data['月份_2']
            if ST_data.columns.__contains__("year_2016"):
                del ST_data['year_2016']
            if ST_data.columns.__contains__("year_2017"):
                del ST_data['year_2017']
            if ST_data.columns.__contains__("year_2018"):
                del ST_data['year_2018']
            if ST_data.columns.__contains__("year_2019"):
                del ST_data['year_2019']
            if ST_data.columns.__contains__("year_2020"):
                del ST_data['year_2020']
            if ST_data.columns.__contains__("year_2021"):
                del ST_data['year_2021']
            if ST_data.columns.__contains__("能源生产比上年增长(%)"):
                del ST_data['能源生产比上年增长(%)']
            if ST_data.columns.__contains__("能源生产弹性系数"):
                del ST_data['能源生产弹性系数']
            if ST_data.columns.__contains__("全省能源消费量(百万吨标准煤)"):
                del ST_data['全省能源消费量(百万吨标准煤)']
            if ST_data.columns.__contains__("全省电力消费量(百亿千瓦小时)"):
                del ST_data['全省电力消费量(百亿千瓦小时)']
            if ST_data.columns.__contains__("能源消费弹性系数"):
                del ST_data['能源消费弹性系数']
            if ST_data.columns.__contains__("电力消费弹性系数"):
                del ST_data['电力消费弹性系数']
            if ST_data.columns.__contains__("按地区分组的法人单位数(万人)_p35"):
                del ST_data['按地区分组的法人单位数(万人)_p35']
            if ST_data.columns.__contains__("各市规模以上企业年末单位就业人员(万人)_p72"):
                del ST_data['各市规模以上企业年末单位就业人员(万人)_p72']
            if ST_data.columns.__contains__("生产总值(百亿元)_p518_p539"):
                del ST_data['生产总值(百亿元)_p518_p539']
            if ST_data.columns.__contains__("第一产业(百亿元)_p537_p539"):
                del ST_data['第一产业(百亿元)_p537_p539']
            if ST_data.columns.__contains__("第二产业(百亿元)_p537_p539"):
                del ST_data['第二产业(百亿元)_p537_p539']
            if ST_data.columns.__contains__("第三产业(百亿元)_p537_p539"):
                del ST_data['第三产业(百亿元)_p537_p539']
            if ST_data.columns.__contains__("分行业全社会单位就业人员年平均工资_p163"):
                del ST_data['分行业全社会单位就业人员年平均工资_p163']
            ST_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)
        if i == 'MT_data.csv':
            try:
                MT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                MT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(MT_data.columns)
            # 1/0
            if MT_data.columns.__contains__("月份_1"):
                del MT_data['月份_1']
            if MT_data.columns.__contains__("月份_2"):
                del MT_data['月份_2']
            if MT_data.columns.__contains__("year_2016"):
                del MT_data['year_2016']
            if MT_data.columns.__contains__("year_2017"):
                del MT_data['year_2017']
            if MT_data.columns.__contains__("year_2018"):
                del MT_data['year_2018']
            if MT_data.columns.__contains__("year_2019"):
                del MT_data['year_2019']
            if MT_data.columns.__contains__("year_2020"):
                del MT_data['year_2020']
            if MT_data.columns.__contains__("year_2021"):
                del MT_data['year_2021']
            if MT_data.columns.__contains__("能源生产比上年增长(%)"):
                del MT_data['能源生产比上年增长(%)']
            if MT_data.columns.__contains__("能源生产弹性系数"):
                del MT_data['能源生产弹性系数']
            if MT_data.columns.__contains__("全省能源消费量(百万吨标准煤)"):
                del MT_data['全省能源消费量(百万吨标准煤)']
            if MT_data.columns.__contains__("全省电力消费量(百亿千瓦小时)"):
                del MT_data['全省电力消费量(百亿千瓦小时)']
            if MT_data.columns.__contains__("能源消费弹性系数"):
                del MT_data['能源消费弹性系数']
            if MT_data.columns.__contains__("电力消费弹性系数"):
                del MT_data['电力消费弹性系数']
            if MT_data.columns.__contains__("按地区分组的法人单位数(万人)_p35"):
                del MT_data['按地区分组的法人单位数(万人)_p35']
            if MT_data.columns.__contains__("各市规模以上企业年末单位就业人员(万人)_p72"):
                del MT_data['各市规模以上企业年末单位就业人员(万人)_p72']
            if MT_data.columns.__contains__("生产总值(百亿元)_p518_p539"):
                del MT_data['生产总值(百亿元)_p518_p539']
            if MT_data.columns.__contains__("第一产业(百亿元)_p537_p539"):
                del MT_data['第一产业(百亿元)_p537_p539']
            if MT_data.columns.__contains__("第二产业(百亿元)_p537_p539"):
                del MT_data['第二产业(百亿元)_p537_p539']
            if MT_data.columns.__contains__("第三产业(百亿元)_p537_p539"):
                del MT_data['第三产业(百亿元)_p537_p539']
            if MT_data.columns.__contains__("分行业全社会单位就业人员年平均工资_p163"):
                del MT_data['分行业全社会单位就业人员年平均工资_p163']
            MT_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)

        if i == 'RT_data.csv':
            try:
                RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(RT_data.columns)
            # 1/0
            if RT_data.columns.__contains__("月份_1"):
                del RT_data['月份_1']
            if RT_data.columns.__contains__("月份_2"):
                del RT_data['月份_2']
            if RT_data.columns.__contains__("year_2016"):
                del RT_data['year_2016']
            if RT_data.columns.__contains__("year_2017"):
                del RT_data['year_2017']
            if RT_data.columns.__contains__("year_2018"):
                del RT_data['year_2018']
            if RT_data.columns.__contains__("year_2019"):
                del RT_data['year_2019']
            if RT_data.columns.__contains__("year_2020"):
                del RT_data['year_2020']
            if RT_data.columns.__contains__("year_2021"):
                del RT_data['year_2021']
            if RT_data.columns.__contains__("能源生产比上年增长(%)"):
                del RT_data['能源生产比上年增长(%)']
            if RT_data.columns.__contains__("能源生产弹性系数"):
                del RT_data['能源生产弹性系数']
            if RT_data.columns.__contains__("全省能源消费量(百万吨标准煤)"):
                del RT_data['全省能源消费量(百万吨标准煤)']
            if RT_data.columns.__contains__("全省电力消费量(百亿千瓦小时)"):
                del RT_data['全省电力消费量(百亿千瓦小时)']
            if RT_data.columns.__contains__("能源消费弹性系数"):
                del RT_data['能源消费弹性系数']
            if RT_data.columns.__contains__("电力消费弹性系数"):
                del RT_data['电力消费弹性系数']
            if RT_data.columns.__contains__("按地区分组的法人单位数(万人)_p35"):
                del RT_data['按地区分组的法人单位数(万人)_p35']
            if RT_data.columns.__contains__("各市规模以上企业年末单位就业人员(万人)_p72"):
                del RT_data['各市规模以上企业年末单位就业人员(万人)_p72']
            if RT_data.columns.__contains__("生产总值(百亿元)_p518_p539"):
                del RT_data['生产总值(百亿元)_p518_p539']
            if RT_data.columns.__contains__("第一产业(百亿元)_p537_p539"):
                del RT_data['第一产业(百亿元)_p537_p539']
            if RT_data.columns.__contains__("第二产业(百亿元)_p537_p539"):
                del RT_data['第二产业(百亿元)_p537_p539']
            if RT_data.columns.__contains__("第三产业(百亿元)_p537_p539"):
                del RT_data['第三产业(百亿元)_p537_p539']
            if RT_data.columns.__contains__("分行业全社会单位就业人员年平均工资_p163"):
                del RT_data['分行业全社会单位就业人员年平均工资_p163']
            RT_data.to_csv(path_or_buf=root+'\\'+i, encoding="utf_8_sig", index=False)
for root, dirs, filelist in os.walk("d:\\用电数据集\\归一化之后的数据集-待采样\\按负荷特性聚类划分数据集V2_方案1-汇总标准化\\"):
    for i in filelist:
        if i == 'ST_data.csv':
            try:
                ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            if ST_data.columns.__contains__("月份_1"):
                del ST_data['月份_1']
            if ST_data.columns.__contains__("月份_2"):
                del ST_data['月份_2']
            if ST_data.columns.__contains__("year_2016"):
                del ST_data['year_2016']
            if ST_data.columns.__contains__("year_2017"):
                del ST_data['year_2017']
            if ST_data.columns.__contains__("year_2018"):
                del ST_data['year_2018']
            if ST_data.columns.__contains__("year_2019"):
                del ST_data['year_2019']
            if ST_data.columns.__contains__("year_2020"):
                del ST_data['year_2020']
            if ST_data.columns.__contains__("year_2021"):
                del ST_data['year_2021']
            if ST_data.columns.__contains__("能源生产比上年增长(%)"):
                del ST_data['能源生产比上年增长(%)']
            if ST_data.columns.__contains__("能源生产弹性系数"):
                del ST_data['能源生产弹性系数']
            if ST_data.columns.__contains__("全省能源消费量(百万吨标准煤)"):
                del ST_data['全省能源消费量(百万吨标准煤)']
            if ST_data.columns.__contains__("全省电力消费量(百亿千瓦小时)"):
                del ST_data['全省电力消费量(百亿千瓦小时)']
            if ST_data.columns.__contains__("能源消费弹性系数"):
                del ST_data['能源消费弹性系数']
            if ST_data.columns.__contains__("电力消费弹性系数"):
                del ST_data['电力消费弹性系数']
            if ST_data.columns.__contains__("按地区分组的法人单位数(万人)_p35"):
                del ST_data['按地区分组的法人单位数(万人)_p35']
            if ST_data.columns.__contains__("各市规模以上企业年末单位就业人员(万人)_p72"):
                del ST_data['各市规模以上企业年末单位就业人员(万人)_p72']
            if ST_data.columns.__contains__("生产总值(百亿元)_p518_p539"):
                del ST_data['生产总值(百亿元)_p518_p539']
            if ST_data.columns.__contains__("第一产业(百亿元)_p537_p539"):
                del ST_data['第一产业(百亿元)_p537_p539']
            if ST_data.columns.__contains__("第二产业(百亿元)_p537_p539"):
                del ST_data['第二产业(百亿元)_p537_p539']
            if ST_data.columns.__contains__("第三产业(百亿元)_p537_p539"):
                del ST_data['第三产业(百亿元)_p537_p539']
            if ST_data.columns.__contains__("分行业全社会单位就业人员年平均工资_p163"):
                del ST_data['分行业全社会单位就业人员年平均工资_p163']
            ST_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)
        if i == 'MT_data.csv':
            try:
                MT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                MT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(MT_data.columns)
            # 1/0
            if MT_data.columns.__contains__("月份_1"):
                del MT_data['月份_1']
            if MT_data.columns.__contains__("月份_2"):
                del MT_data['月份_2']
            if MT_data.columns.__contains__("year_2016"):
                del MT_data['year_2016']
            if MT_data.columns.__contains__("year_2017"):
                del MT_data['year_2017']
            if MT_data.columns.__contains__("year_2018"):
                del MT_data['year_2018']
            if MT_data.columns.__contains__("year_2019"):
                del MT_data['year_2019']
            if MT_data.columns.__contains__("year_2020"):
                del MT_data['year_2020']
            if MT_data.columns.__contains__("year_2021"):
                del MT_data['year_2021']
            if MT_data.columns.__contains__("能源生产比上年增长(%)"):
                del MT_data['能源生产比上年增长(%)']
            if MT_data.columns.__contains__("能源生产弹性系数"):
                del MT_data['能源生产弹性系数']
            if MT_data.columns.__contains__("全省能源消费量(百万吨标准煤)"):
                del MT_data['全省能源消费量(百万吨标准煤)']
            if MT_data.columns.__contains__("全省电力消费量(百亿千瓦小时)"):
                del MT_data['全省电力消费量(百亿千瓦小时)']
            if MT_data.columns.__contains__("能源消费弹性系数"):
                del MT_data['能源消费弹性系数']
            if MT_data.columns.__contains__("电力消费弹性系数"):
                del MT_data['电力消费弹性系数']
            if MT_data.columns.__contains__("按地区分组的法人单位数(万人)_p35"):
                del MT_data['按地区分组的法人单位数(万人)_p35']
            if MT_data.columns.__contains__("各市规模以上企业年末单位就业人员(万人)_p72"):
                del MT_data['各市规模以上企业年末单位就业人员(万人)_p72']
            if MT_data.columns.__contains__("生产总值(百亿元)_p518_p539"):
                del MT_data['生产总值(百亿元)_p518_p539']
            if MT_data.columns.__contains__("第一产业(百亿元)_p537_p539"):
                del MT_data['第一产业(百亿元)_p537_p539']
            if MT_data.columns.__contains__("第二产业(百亿元)_p537_p539"):
                del MT_data['第二产业(百亿元)_p537_p539']
            if MT_data.columns.__contains__("第三产业(百亿元)_p537_p539"):
                del MT_data['第三产业(百亿元)_p537_p539']
            if MT_data.columns.__contains__("分行业全社会单位就业人员年平均工资_p163"):
                del MT_data['分行业全社会单位就业人员年平均工资_p163']
            MT_data.to_csv(path_or_buf=root + '\\' + i, encoding="utf_8_sig", index=False)

        if i == 'RT_data.csv':
            try:
                RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
            except:
                RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
            # print(RT_data.columns)
            # 1/0
            if RT_data.columns.__contains__("月份_1"):
                del RT_data['月份_1']
            if RT_data.columns.__contains__("月份_2"):
                del RT_data['月份_2']
            if RT_data.columns.__contains__("year_2016"):
                del RT_data['year_2016']
            if RT_data.columns.__contains__("year_2017"):
                del RT_data['year_2017']
            if RT_data.columns.__contains__("year_2018"):
                del RT_data['year_2018']
            if RT_data.columns.__contains__("year_2019"):
                del RT_data['year_2019']
            if RT_data.columns.__contains__("year_2020"):
                del RT_data['year_2020']
            if RT_data.columns.__contains__("year_2021"):
                del RT_data['year_2021']
            if RT_data.columns.__contains__("能源生产比上年增长(%)"):
                del RT_data['能源生产比上年增长(%)']
            if RT_data.columns.__contains__("能源生产弹性系数"):
                del RT_data['能源生产弹性系数']
            if RT_data.columns.__contains__("全省能源消费量(百万吨标准煤)"):
                del RT_data['全省能源消费量(百万吨标准煤)']
            if RT_data.columns.__contains__("全省电力消费量(百亿千瓦小时)"):
                del RT_data['全省电力消费量(百亿千瓦小时)']
            if RT_data.columns.__contains__("能源消费弹性系数"):
                del RT_data['能源消费弹性系数']
            if RT_data.columns.__contains__("电力消费弹性系数"):
                del RT_data['电力消费弹性系数']
            if RT_data.columns.__contains__("按地区分组的法人单位数(万人)_p35"):
                del RT_data['按地区分组的法人单位数(万人)_p35']
            if RT_data.columns.__contains__("各市规模以上企业年末单位就业人员(万人)_p72"):
                del RT_data['各市规模以上企业年末单位就业人员(万人)_p72']
            if RT_data.columns.__contains__("生产总值(百亿元)_p518_p539"):
                del RT_data['生产总值(百亿元)_p518_p539']
            if RT_data.columns.__contains__("第一产业(百亿元)_p537_p539"):
                del RT_data['第一产业(百亿元)_p537_p539']
            if RT_data.columns.__contains__("第二产业(百亿元)_p537_p539"):
                del RT_data['第二产业(百亿元)_p537_p539']
            if RT_data.columns.__contains__("第三产业(百亿元)_p537_p539"):
                del RT_data['第三产业(百亿元)_p537_p539']
            if RT_data.columns.__contains__("分行业全社会单位就业人员年平均工资_p163"):
                del RT_data['分行业全社会单位就业人员年平均工资_p163']
            RT_data.to_csv(path_or_buf=root+'\\'+i, encoding="utf_8_sig", index=False)