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

try:
    ST_data = pd.read_csv('D:\\用电数据集\\特征工程加强\\宏观经济数据-电力弹性系数.csv', encoding='utf-8', sep=',')
except:
    ST_data = pd.read_csv('D:\\用电数据集\\特征工程加强\\宏观经济数据-电力弹性系数.csv', encoding='gbk', sep=',')

analysis_res=pd.DataFrame()#分析结果

print(ST_data.columns)

list=['按地区分组的法人单位数(万人)_p35', '总人口数(万人)_p46',
       '各市规模以上企业年末单位就业人员(万人)_p72', '各市、县居民消费价格指数_p144', '土地面积(万平方公里)_p539',
       '生产总值(百亿元)_p518_p539', '全年用电量_百亿千瓦时_p529_p571',
       '城镇居民人均可支配收入(万元)_p537_p539', '第一产业(百亿元)_p537_p539',
       '第二产业(百亿元)_p537_p539', '第三产业(百亿元)_p537_p539']#地区
list2=['按行业分的法人单位数_p29', '按产业分的全省生产总值_亿元_p18', '按行业分的全省生产总值_亿元_p18',
       '总支出_亿元_p22', '按行业和经济类型分的就业人员总数_非私营与规上私营之和_年末数_万人_p61_p71',
       '项目建成投产率_p84', '分行业全社会单位就业人员年平均工资_p163', '按行业分全社会用电情况_亿千瓦时_p305',
       '按产业分用电合计_亿千瓦时_p305']#行业
list3=['年份', '全省能源生产量(百万吨标准煤)', '全省电力生产量(百亿千瓦小时)', '能源生产比上年增长(%)',
       '电力生产比上年增长(%)', '生产总值比上年增长\n', '能源生产弹性系数', '电力生产弹性系数',
       '全省能源消费量(百万吨标准煤)', '全省电力消费量(百亿千瓦小时)', '能源消费比上年增长(%)', '电力消费比上年增长(%)',
       '能源消费弹性系数', '电力消费弹性系数']
for feature in list3:
    #皮尔逊相关系数、斯皮尔曼相关系数、点二列相关
    correlation_pearsonr, pvalue_pearsonr = stats.stats.pearsonr(ST_data[feature],ST_data['全省电力消费量(百亿千瓦小时)'])#皮尔逊相关系数 周末0.82 星期映射 0.485059
    correlation_spearmanr, pvalue_spearmanr = stats.stats.spearmanr(ST_data[feature],ST_data['全省电力消费量(百亿千瓦小时)'])#### 斯皮尔曼等级相关
    correlation_pointbiserialr, pvalue_pointbiserialr = stats.stats.pointbiserialr((ST_data[feature]<ST_data[feature].mean())*1,ST_data['全省电力消费量(百亿千瓦小时)'])  #### 点二列相关0.75673839

    analysis_res=analysis_res.append([{'特征':feature,
    '相关系数_pearsonr':correlation_pearsonr,
    'pvalue_pearsonr': pvalue_pearsonr,
    '相关系数_spearmanr':correlation_spearmanr,
    'pvalue_spearmanr': pvalue_spearmanr,
    '相关系数_pointbiserialr':correlation_pointbiserialr,
    'pvalue_pointbiserialr':pvalue_pointbiserialr}])
    #+1是为了消除除数为0的问题
print(analysis_res)
analysis_res.to_csv(path_or_buf='D:\\用电数据集\\特征工程加强\\宏观经济数据-电力弹性系数分析结果.csv', encoding="utf_8_sig",index=False)
