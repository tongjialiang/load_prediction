from numpy import random
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
import scipy.stats as stats
import gc
import json
from ShowapiRequest import ShowapiRequest

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

#待处理特征
list=['平均负荷(kW)', '最大负荷(kW)', '最小负荷(kW)','受电容量(KVA)','max_temperature', 'min_temperature', '平均气温', '是否工作日', 'weather_中雨',
       'weather_多云', 'weather_大雨', 'weather_小雨', 'weather_晴', 'weather_暴雨',
       'weather_阴', 'weather_阵雨', 'weather_雪', '星期_0', '星期_1', '星期_2', '星期_3',
       '星期_4', '星期_5', '星期_6', '月份_1', '月份_2', '月份_3', '月份_4', '月份_5', '月份_6',
       '月份_7', '月份_8', '月份_9', '月份_10', '月份_11', '月份_12', 'year_2016',
       'year_2017', 'year_2018', 'year_2019', 'year_2020', 'year_2021',
       '小到中雨及下雪天', '大雨暴雨天', '不下雨', '天气映射', '周末', '星期映射', '春节期间', '春', '夏', '秋',
       '冬','星期特征_遗传算法','月份特征_遗传算法']
# list=['月份特征_遗传算法']
analysis_res=pd.DataFrame()#分析结果
for feature in list:
    count_pearsonr=0#计数：皮尔逊系数的结果中，中度显著相关的公司数
    correlation_sum_pearsonr=0#计数：皮尔逊系数的结果中，中度显著相关的相关系数之和
    pvalue_sum_pearsonr=0#计数：皮尔逊系数的结果中，中度显著相关的pvalue之和

    count_spearmanr=0#
    correlation_sum_spearmanr=0
    pvalue_sum_spearmanr=0

    count_pointbiserialr=0#
    correlation_sum_pointbiserialr=0
    pvalue_sum_pointbiserialr=0

    flag=0
    for root, dirs, filelist in os.walk("D:\\用电数据集\\特征工程加强\\用于特征选择的行业数据集_2020—2021每月取3周数据_星期增强_月份增强\\"):
            for i in filelist:
                #filename=root+"\\"+i
                #print(filename)
                #print(root)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_每月取一周\交通运输仓储和邮政业\余杭区供电分公司2\190
                if i == 'ST_data.csv':
                    #print(root)#D:\用电数据集\特征工程加强\用于特征选择的行业数据集_字段已添加\交通运输仓储和邮政业\临安区供电分公司\122
                    dir_new=''
                    file_new=''
                    try:
                        ST_data = pd.read_csv(root+'\\'+i, encoding='utf-8', sep=',')
                    except:
                        ST_data = pd.read_csv(root+'\\'+i, encoding='gbk', sep=',')
                    if flag==0:
                        print(ST_data.columns)
                        flag=1
                    #皮尔逊相关系数、斯皮尔曼相关系数、点二列相关
                    correlation_pearsonr, pvalue_pearsonr = stats.stats.pearsonr(ST_data[feature],ST_data['平均负荷(kW)'])#皮尔逊相关系数 周末0.82 星期映射 0.485059
                    correlation_spearmanr, pvalue_spearmanr = stats.stats.spearmanr(ST_data[feature],ST_data['平均负荷(kW)'])#### 斯皮尔曼等级相关
                    #print((ST_data['平均气温']<ST_data['平均气温'].mean())*1)
                    correlation_pointbiserialr, pvalue_pointbiserialr = stats.stats.pointbiserialr((ST_data[feature]<ST_data[feature].mean())*1,ST_data['平均负荷(kW)'])  #### 点二列相关0.75673839
                    #correlation, pvalue = stats.stats.pointbiserialr(ST_data['平均气温'],ST_data['平均负荷(kW)'])#### 点二列相关 0.7444693425815634
                    #### 斯皮尔曼等级相关
                    if abs(correlation_pearsonr)>=0.45 and pvalue_pearsonr<0.05:
                        count_pearsonr += 1
                        #print(root)
                        #print(correlation_pearsonr)
                        #print(pvalue_pearsonr)
                        pvalue_sum_pearsonr+=pvalue_pearsonr
                        correlation_sum_pearsonr+=abs(correlation_pearsonr)

                    if abs(correlation_spearmanr)>=0.45 and pvalue_spearmanr<0.05:
                        count_spearmanr += 1
                        #print(root)
                        #print(correlation_pearsonr)
                        #print(pvalue_pearsonr)
                        pvalue_sum_spearmanr+=pvalue_spearmanr
                        correlation_sum_spearmanr+=abs(correlation_spearmanr)

                    if abs(correlation_pointbiserialr) >= 0.45 and pvalue_pointbiserialr < 0.05:
                        count_pointbiserialr += 1
                        # print(root)
                        # print(correlation_pearsonr)
                        # print(pvalue_pearsonr)
                        pvalue_sum_pointbiserialr += pvalue_pointbiserialr
                        correlation_sum_pointbiserialr += abs(correlation_pointbiserialr)
                    #print(correlation_sum)
                        #print('pvalue', pvalue/count)
    #print(correlation_sum)
    # print('采样公司总数',3992)
    #
    # print('显著中度相关公司数_pearsonr',count_pearsonr)#总企业数1115
    # print('显著中度相关公司数占比_pearsonr',count_pearsonr/3992)#0.215
    # print('平均相关系数_pearsonr', correlation_sum_pearsonr/count_pearsonr)#0.48505927
    # print('平均pvalue_pearsonr', pvalue_sum_pearsonr/count_pearsonr)
    #
    # print('显著中度相关公司数_spearmanr',count_spearmanr)#总企业数1115
    # print('显著中度相关公司数占比_spearmanr',count_spearmanr/3992)#0.215
    # print('平均相关系数_spearmanr', correlation_sum_spearmanr/count_spearmanr)#0.48505927
    # print('平均pvalue_spearmanr', pvalue_sum_spearmanr/count_spearmanr)
    #
    # print('显著中度相关公司数_pointbiserialr',count_pointbiserialr)#总企业数1115
    # print('显著中度相关公司数占比_pointbiserialr',count_pointbiserialr/3992)#0.215
    # print('平均相关系数_pointbiserialr', correlation_sum_pointbiserialr/count_pointbiserialr)#0.48505927
    # print('平均pvalue_pointbiserialr', pvalue_sum_pointbiserialr/count_pointbiserialr)
    analysis_res=analysis_res.append([{'特征':feature,'采样公司总数':3992,'显著中度相关公司数_pearsonr':count_pearsonr,
    '显著中度相关公司数占比_pearsonr':count_pearsonr/3992,'平均相关系数_pearsonr':correlation_sum_pearsonr/(count_pearsonr+1),
    '平均pvalue_pearsonr': pvalue_sum_pearsonr/(count_pearsonr+1),
    '显著中度相关公司数_spearmanr':count_spearmanr,'显著中度相关公司数占比_spearmanr':count_spearmanr/3992,
    '平均相关系数_spearmanr':correlation_sum_spearmanr/(count_spearmanr+1),'平均pvalue_spearmanr': pvalue_sum_spearmanr/(count_spearmanr+1),
    '显著中度相关公司数_pointbiserialr':count_pointbiserialr,'显著中度相关公司数占比_pointbiserialr':count_pointbiserialr/3992,
    '平均相关系数_pointbiserialr':correlation_sum_pointbiserialr/(count_pointbiserialr+1),
    '平均pvalue_pointbiserialr':pvalue_sum_pointbiserialr/(count_pointbiserialr+1)}])
    print("特征",feature,"处理完毕")
    #+1是为了消除除数为0的问题
print(analysis_res)
analysis_res.to_csv(path_or_buf='D:\\用电数据集\\特征工程加强\\日期天气特征分析结果_特征学习之后999.csv', encoding="utf_8_sig",index=False)
