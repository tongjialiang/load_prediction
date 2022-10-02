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
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
#杭州天气和日期数据
try:
    hz_data = pd.read_csv('D:\\用电数据集\\需要保留的特征\\杭州地区天气和节假日数据_增强.csv', encoding='utf-8', sep=',')
except:
    hz_data = pd.read_csv('D:\\用电数据集\\需要保留的特征\\杭州地区天气和节假日数据_增强.csv', encoding='gbk', sep=',')

    # print(hz_data[:2])
    # print(hz_data.columns)
    # 1/0
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
    jx_data = pd.read_csv('D:\\用电数据集\\需要保留的特征\\嘉兴地区天气和节假日数据_增强.csv', encoding='utf-8', sep=',')
except:
    jx_data = pd.read_csv('D:\\用电数据集\\需要保留的特征\\嘉兴地区天气和节假日数据_增强.csv', encoding='gbk', sep=',')

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
    business_data = pd.read_csv('D:\\用电数据集\\需要保留的特征\\企业信息查询表.csv', encoding='utf-8', sep=',')
except:
    business_data = pd.read_csv('D:\\用电数据集\\需要保留的特征\\企业信息查询表.csv', encoding='gbk', sep=',')
# print(business_data.columns)
# 1/0


#打开企业信息查询表
try:
    hangye_data = pd.read_csv('D:\\用电数据集\\需要保留的特征\\宏观经济数据-行业.csv', encoding='utf-8', sep=',')
except:
    hangye_data = pd.read_csv('D:\\用电数据集\\需要保留的特征\\宏观经济数据-行业.csv', encoding='gbk', sep=',')

#打开地区信息查询表
try:
    diqu_data = pd.read_csv('D:\\用电数据集\\需要保留的特征\\宏观经济数据-地区.csv', encoding='utf-8', sep=',')
except:
    diqu_data = pd.read_csv('D:\\用电数据集\\需要保留的特征\\宏观经济数据-地区.csv', encoding='gbk', sep=',')

#打开电力弹性系数表
try:
    tanxin_data = pd.read_csv('D:\\用电数据集\\需要保留的特征\\宏观经济数据-电力弹性系数.csv', encoding='utf-8', sep=',')
except:
    tanxin_data = pd.read_csv('D:\\用电数据集\\需要保留的特征\\宏观经济数据-电力弹性系数.csv', encoding='gbk', sep=',')

#print(business_data[:2])
#打开st文件
for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_去异常\\"):
        for i in filelist:
            if i == 'ST_data.csv':
                #print(root) #D:\用电数据集\特征工程加强\用于特征选择的行业数据集\农林牧渔业\临安区供电分公司\121
                new_dir = 'D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_特征增强\\' + root.split('\\')[-3] + '\\' + root.split('\\')[
                    -2] + '\\' + root.split('\\')[-1] + '\\'
                new_file='D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_特征增强\\'+root.split('\\')[-3]+'\\'+root.split('\\')[-2]+'\\'+root.split('\\')[-1]+'\\'+i
                print(new_file)
                try:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    ST_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')

                #print(ST_data[:5])
                for i, STrow in ST_data.iterrows(): #遍历st的每一行数据
                    STtime=STrow['数据时间'].strip('\t')  # 删除末尾换行符
                    #STtime_dt=pd.to_datetime(STtime, format='%Y-%m-%d') #转datetime
                    #print(STtime)#2016-08-04
                    #print(STrow['id'])#临安区供电分公司_121
                    #把id输入至企业信息查询表查到地区
                    area=business_data[business_data['id'] == STrow['id']]['所在地区'].array[0]
                    #print(area)#临安区

                    #添加天气特征

                    #如果地区属于嘉兴，查嘉兴地区天气和节假日数据_增强.csv
                    #如果地区属于杭州，查杭州地区天气和节假日数据_增强.csv
                    if area not in ['海宁市','海盐县','嘉善县','嘉兴市桐乡市','平湖市']:
                        #根据ST_data的日期从嘉兴地区天气和节假日数据_增强表中查到对应日期的天气等数据
                        #print(hz_data[hz_data.loc[:,'日期']==STtime])
                        temp=hz_data[hz_data.loc[:, '日期'] == STtime]

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

                    #添加电力弹性数据
                    # print("))))")
                    # print(STtime.split('-')[0])
                    #print(tanxin_data['年份']==int(2016))
                    temp=tanxin_data[tanxin_data['年份']+1==int(STtime.split('-')[0])]
                    #print(temp)
                    for new_feature in temp.columns:
                        if new_feature != '年份':
                            #print(new_feature)
                            ST_data.loc[i, new_feature] = temp[new_feature].array[0]
                    #添加行业特征
                    #print(root.split('\\')[-3])#D:\用电数据集\归一化之前的数据集\按行业划分数据集_去异常\交通运输仓储和邮政业\临安区供电分公司\122
                    temp=hangye_data[hangye_data['所属行业']==root.split('\\')[-3]]
                    for new_feature in temp.columns:
                        if new_feature != '所属行业':
                            #print(new_feature)
                            ST_data.loc[i, new_feature] = temp[new_feature].array[0]
                    # 添加地区特征
                    temp=diqu_data[diqu_data['所在地区']==area]
                    for new_feature in temp.columns:
                        if new_feature != '所在地区':
                            #print(new_feature)
                            ST_data.loc[i, new_feature] = temp[new_feature].array[0]
                    #print(temp)
                del ST_data['最大负荷(kW)']
                del ST_data['最小负荷(kW)']
                del ST_data['最小负荷发生时间']
                del ST_data['最大负荷发生时间']
                del ST_data['region']
                del ST_data['company']
                del ST_data['business_type']
                del ST_data['局号(终端/表计)']
                # print(ST_data.columns)
                # print(ST_data.columns[0])
                # 1/0
                if os.path.exists(new_dir) == False:
                    os.makedirs(new_dir)
                ST_data.to_csv(path_or_buf=new_file, encoding="utf_8_sig", index=False)

            if i == 'MT_data.csv':
                # print(root) #D:\用电数据集\特征工程加强\用于特征选择的行业数据集\农林牧渔业\临安区供电分公司\121
                new_dir = 'D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_特征增强\\' + root.split('\\')[-3] + '\\' + root.split('\\')[
                    -2] + '\\' + root.split('\\')[-1] + '\\'
                new_file = 'D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_特征增强\\' + root.split('\\')[-3] + '\\' + root.split('\\')[
                    -2] + '\\' + root.split('\\')[-1] + '\\' + i
                print(new_file)
                try:
                    MT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    MT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')

                # print(ST_data[:5])
                for i, MTrow in MT_data.iterrows():  # 遍历st的每一行数据
                    MTtime = MTrow['数据时间'].strip('\t')  # 删除末尾换行符
                    # STtime_dt=pd.to_datetime(STtime, format='%Y-%m-%d') #转datetime
                    #print(MTtime)  # 2016-08-04
                    # print(STrow['id'])#临安区供电分公司_121
                    # 把id输入至企业信息查询表查到地区
                    area = business_data[business_data['id'] == MTrow['id']]['所在地区'].array[0]
                    # print(area)#临安区

                    # 添加天气特征

                    # 如果地区属于嘉兴，查嘉兴地区天气和节假日数据_增强.csv
                    # 如果地区属于杭州，查杭州地区天气和节假日数据_增强.csv
                    if area not in ['海宁市', '海盐县', '嘉善县', '嘉兴市桐乡市', '平湖市']:
                        # 根据ST_data的日期从嘉兴地区天气和节假日数据_增强表中查到对应日期的天气等数据
                        # print(hz_data[hz_data.loc[:,'日期']==STtime])
                        temp = hz_data[hz_data.loc[:, '日期'] == MTtime]
                        # print(temp.columns)
                        # 把每个特征加入到ST文件中
                        for new_feature in temp.columns:
                            if new_feature in ['月份_1','月份_2']:
                                MT_data.loc[i, new_feature] = temp[new_feature].array[0]
                        # print(ST_data[:3])
                    if area in ['海宁市', '海盐县', '嘉善县', '嘉兴市桐乡市', '平湖市']:
                        # 根据ST_data的日期从嘉兴地区天气和节假日数据_增强表中查到对应日期的天气等数据
                        # print(hz_data[hz_data.loc[:,'日期']==STtime])
                        temp = jx_data[jx_data.loc[:, '日期'] == MTtime]
                        # print(temp.columns)
                        # 把每个特征加入到ST文件中
                        for new_feature in temp.columns:
                            if new_feature in ['月份_1','月份_2']:
                                MT_data.loc[i, new_feature] = temp[new_feature].array[0]
                        # print(ST_data[:3])

                    # 添加电力弹性数据
                    # print("))))")
                    # print(STtime.split('-')[0])
                    # print(tanxin_data['年份']==int(2016))
                    temp = tanxin_data[tanxin_data['年份'] + 1 == int(MTtime.split('-')[0])]
                    # print(temp)
                    for new_feature in temp.columns:
                        if new_feature != '年份':
                            # print(new_feature)
                            MT_data.loc[i, new_feature] = temp[new_feature].array[0]
                    # 添加行业特征
                    # print(root.split('\\')[-3])#D:\用电数据集\归一化之前的数据集\按行业划分数据集_去异常\交通运输仓储和邮政业\临安区供电分公司\122
                    temp = hangye_data[hangye_data['所属行业'] == root.split('\\')[-3]]
                    for new_feature in temp.columns:
                        if new_feature != '所属行业':
                            # print(new_feature)
                            MT_data.loc[i, new_feature] = temp[new_feature].array[0]
                    # 添加地区特征
                    temp = diqu_data[diqu_data['所在地区'] == area]
                    for new_feature in temp.columns:
                        if new_feature != '所在地区':
                            # print(new_feature)
                            MT_data.loc[i, new_feature] = temp[new_feature].array[0]
                    # print(temp)
                # print(MT_data.columns)
                # 1/0
                del MT_data['最大负荷(kW)']
                del MT_data['最小负荷(kW)']
                del MT_data['最大负荷发生日']
                del MT_data['最小负荷发生日']
                del MT_data['region']
                del MT_data['company']
                del MT_data['business_type']
                del MT_data['局号(终端/表计)']
                del MT_data['数据时间2']
                if os.path.exists(new_dir) == False:
                    os.makedirs(new_dir)
                MT_data.to_csv(path_or_buf=new_file, encoding="utf_8_sig", index=False)

            if i == 'RT_data.csv':
                # print(root) #D:\用电数据集\特征工程加强\用于特征选择的行业数据集\农林牧渔业\临安区供电分公司\121
                new_dir = 'D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_特征增强\\' + root.split('\\')[-3] + '\\' + root.split('\\')[
                    -2] + '\\' + root.split('\\')[-1] + '\\'
                new_file = 'D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_特征增强\\' + root.split('\\')[-3] + '\\' + root.split('\\')[
                    -2] + '\\' + root.split('\\')[-1] + '\\' + i
                print(new_file)
                try:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')

                # print(ST_data[:5])
                for i, RTrow in RT_data.iterrows():  # 遍历st的每一行数据
                    RTtime = RTrow['日期'].strip('\t')  # 删除末尾换行符
                    RTtime = RTtime.split(' ')[0]#2021-08-15

                    # STtime_dt=pd.to_datetime(STtime, format='%Y-%m-%d') #转datetime
                    #print(MTtime)  # 2016-08-04
                    # print(STrow['id'])#临安区供电分公司_121
                    # 把id输入至企业信息查询表查到地区
                    area = business_data[business_data['id'] == RTrow['id']]['所在地区'].array[0]
                    # print(area)#临安区

                    # 添加天气特征

                    # 如果地区属于嘉兴，查嘉兴地区天气和节假日数据_增强.csv
                    # 如果地区属于杭州，查杭州地区天气和节假日数据_增强.csv
                    if area not in ['海宁市', '海盐县', '嘉善县', '嘉兴市桐乡市', '平湖市']:
                        # 根据ST_data的日期从嘉兴地区天气和节假日数据_增强表中查到对应日期的天气等数据
                        # print(hz_data[hz_data.loc[:,'日期']==STtime])
                        temp = hz_data[hz_data.loc[:, '日期'] == RTtime]
                        # print(temp.columns)
                        # 把每个特征加入到ST文件中
                        for new_feature in temp.columns:
                            if new_feature in ['星期映射_遗传算法','天气映射','平均气温']:
                                RT_data.loc[i, new_feature] = temp[new_feature].array[0]
                        # print(ST_data[:3])
                    if area in ['海宁市', '海盐县', '嘉善县', '嘉兴市桐乡市', '平湖市']:
                        # 根据ST_data的日期从嘉兴地区天气和节假日数据_增强表中查到对应日期的天气等数据
                        # print(hz_data[hz_data.loc[:,'日期']==STtime])

                        temp = jx_data[jx_data.loc[:, '日期'] == RTtime]
                        # print(temp.columns)
                        # 把每个特征加入到ST文件中
                        for new_feature in temp.columns:
                            if new_feature in ['星期映射_遗传算法','天气映射','平均气温']:
                                RT_data.loc[i, new_feature] = temp[new_feature].array[0]
                        # print(ST_data[:3])

                    # 添加行业特征
                    # print(root.split('\\')[-3])#D:\用电数据集\归一化之前的数据集\按行业划分数据集_去异常\交通运输仓储和邮政业\临安区供电分公司\122
                    temp = hangye_data[hangye_data['所属行业'] == root.split('\\')[-3]]
                    for new_feature in temp.columns:
                        if new_feature != '所属行业':
                            # print(new_feature)
                            RT_data.loc[i, new_feature] = temp[new_feature].array[0]
                    # 添加地区特征
                    temp = diqu_data[diqu_data['所在地区'] == area]
                    for new_feature in temp.columns:
                        if new_feature != '所在地区':
                            # print(new_feature)
                            RT_data.loc[i, new_feature] = temp[new_feature].array[0]
                    # print(temp)
                #print(RT_data.columns)
                del RT_data['户名']
                del RT_data['供电单位']
                del RT_data['business_type']
                del RT_data['局号(终端/表计)']
                if os.path.exists(new_dir) == False:
                    os.makedirs(new_dir)
                RT_data.to_csv(path_or_buf=new_file, encoding="utf_8_sig", index=False)