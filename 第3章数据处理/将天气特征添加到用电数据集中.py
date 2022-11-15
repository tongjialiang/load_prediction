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
#杭州天气和日期数据
try:
    hz_data = pd.read_csv('D:\\用电数据集\\杭州地区天气和节假日数据_增强.csv', encoding='utf-8', sep=',')
except:
    hz_data = pd.read_csv('D:\\用电数据集\\杭州地区天气和节假日数据_增强.csv', encoding='gbk', sep=',')
#把地区天气和节假日数据_增强的日期格式20200202改成2020-02-02，取字段名为'日期'， 并将其转换为datetime
for i,row in hz_data.iterrows():
    #print(row['time'])#20160101
    rowtime=str(int(row['time']))
    #print(rowtime)#20160101
    rowtime=list(rowtime)
    #print(rowtime)#['2', '0', '1', '6', '0', '1', '0', '1']
    rowtime.insert(4,'-')
    #print(rowtime)#['2', '0', '1', '6', '-', '0', '1', '0', '1']
    rowtime.insert(-2, '-')
    #print(rowtime)#['2', '0', '1', '6', '-', '0', '1', '-', '0', '1']
    rowtime=''.join(rowtime)
    #print(rowtime)#2016-01-01
    #date_now = datetime.datetime.strptime(rowtime, "%Y-%m-%d")
    hz_data.loc[i,'日期']=rowtime
    #hz_data.loc[i, '日期'] = date_now

#print(hz_data[:2])

#嘉兴天气和日期数据
try:
    jx_data = pd.read_csv('D:\\用电数据集\\嘉兴地区天气和节假日数据_增强.csv', encoding='utf-8', sep=',')
except:
    jx_data = pd.read_csv('D:\\用电数据集\\嘉兴地区天气和节假日数据_增强.csv', encoding='gbk', sep=',')
for i,row in jx_data.iterrows():
    #print(row['time'])#20160101
    rowtime=str(int(row['time']))
    #print(rowtime)#20160101
    rowtime=list(rowtime)
    #print(rowtime)#['2', '0', '1', '6', '0', '1', '0', '1']
    rowtime.insert(4,'-')
    #print(rowtime)#['2', '0', '1', '6', '-', '0', '1', '0', '1']
    rowtime.insert(-2, '-')
    #print(rowtime)#['2', '0', '1', '6', '-', '0', '1', '-', '0', '1']
    rowtime=''.join(rowtime)
    #print(rowtime)#2016-01-01
    #date_now = datetime.datetime.strptime(rowtime, "%Y-%m-%d")
    jx_data.loc[i,'日期']=rowtime
    #jx_data.loc[i, '日期'] = date_now
# print(jx_data[:2])
# 1/0
#打开企业信息查询表
try:
    business_data = pd.read_csv('D:\\用电数据集\\企业信息查询表.csv', encoding='utf-8', sep=',')
except:
    business_data = pd.read_csv('D:\\用电数据集\\企业信息查询表.csv', encoding='gbk', sep=',')
#print(business_data[:2])
#打开st文件
for root, dirs, filelist in os.walk("D:\\用电数据集\\特征工程加强\\用于特征选择的行业数据集\\"):
        for i in filelist:
            if i == 'ST_data.csv':
                #print(root) #D:\用电数据集\特征工程加强\用于特征选择的行业数据集\农林牧渔业\临安区供电分公司\121
                new_dir = 'D:\\用电数据集\\特征工程加强\\用于特征选择的行业数据集_字段已添加\\' + root.split('\\')[-3] + '\\' + root.split('\\')[
                    -2] + '\\' + root.split('\\')[-1] + '\\'
                new_file='D:\\用电数据集\\特征工程加强\\用于特征选择的行业数据集_字段已添加\\'+root.split('\\')[-3]+'\\'+root.split('\\')[-2]+'\\'+root.split('\\')[-1]+'\\'+i
                print(new_file)
                try:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                print(ST_data[:5])
                for i, STrow in ST_data.iterrows(): #遍历st的每一行数据
                    STtime=STrow['数据时间'].strip('\t')  # 删除末尾换行符
                    #STtime_dt=pd.to_datetime(STtime, format='%Y-%m-%d') #转datetime
                    print(STtime)#2016-08-04
                    #print(STrow['id'])#临安区供电分公司_121
                    #把id输入至企业信息查询表查到地区
                    area=business_data[business_data['id'] == STrow['id']]['所在地区'].array[0]
                    #print(area)#临安区

                    #如果地区属于嘉兴，查嘉兴地区天气和节假日数据_增强.csv
                    #如果地区属于杭州，查杭州地区天气和节假日数据_增强.csv
                    if area not in ['海宁市','海盐县','嘉善县','嘉兴市桐乡市','平湖市']:
                        #根据ST_data的日期从嘉兴地区天气和节假日数据_增强表中查到对应日期的天气等数据
                        #print(hz_data[hz_data.loc[:,'日期']==STtime])
                        temp=hz_data[hz_data.loc[:, '日期'] == STtime]
                        #print(temp.columns)
                        #把每个特征加入到ST文件中
                        for new_feature in temp.columns:
                            if new_feature != 'time' and new_feature != '日期':
                                ST_data.loc[i,new_feature]=temp[new_feature].array[0]
                        #print(ST_data[:3])
                    if area in ['海宁市','海盐县','嘉善县','嘉兴市桐乡市','平湖市']:
                        #根据ST_data的日期从嘉兴地区天气和节假日数据_增强表中查到对应日期的天气等数据
                        #print(hz_data[hz_data.loc[:,'日期']==STtime])
                        temp=jx_data[jx_data.loc[:, '日期'] == STtime]
                        #print(temp.columns)
                        #把每个特征加入到ST文件中
                        for new_feature in temp.columns:
                            if new_feature != 'time' and new_feature != '日期':
                                ST_data.loc[i,new_feature]=temp[new_feature].array[0]
                        #print(ST_data[:3])
                if os.path.exists(new_dir) == False:
                    os.makedirs(new_dir)
                ST_data.to_csv(path_or_buf=new_file, encoding="utf_8_sig", index=False)