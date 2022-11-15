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
import random
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

#遗传代数n
n=10000

#待处理特征
feature='月份映射'

#变异系数
variable_coefficient=0.04
variable=0
#扩展7*12*3次
#初始个体生成
month_map=[0.048,0.28,0.005,0.008,0.015,0.056,0.036,0.310,0.0097,0.003,0.004,0.023]
month_map_all=[]
type=''
#存储最佳的显著中相关公司数、占比、相关系数、pvalue
count_pearsonr_best,count_pearsonr_ratio_best,correlation_avg_best,pvalue_avg_best=0,0,0,0
month_map_best=month_map.copy()#初始种群生成 显著中度相关公司数占比  注意：把list赋值给变量，赋的是地址而不是值
def extend_map():
    for i in range(21):#7*12*3
        month_map_all.extend([month_map[6]])#extend() 向列表尾部追加一个列表，将列表中的每个元素都追加进来，在原有列表上增加
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[7]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[8]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[9]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[10]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[11]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[0]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[1]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[2]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[3]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[4]])
    for i in range(21):  # 7*12*3
        month_map_all.extend([month_map[5]])

extend_map()
#print(week_map_all)

analysis_res=pd.DataFrame()#分析结果


for num in range(n):
    print('迭代次数'+str(num))
    if num!=0: #第一次不发生进化
        #print("基因开始改变")
        hereditary_type=random.randint(1, 100)
        #print('生成随机数',hereditary_type)
        if hereditary_type>=0 and hereditary_type<= 8:#10%的可能性交叉
            type='基因交叉'
            print(type)
            first=random.randint(0,11)
            second=random.randint(0,11)
            temp=month_map[first]
            month_map[first]=month_map[second]
            month_map[second]=temp
            extend_map()
            print(type, '基因', first, '和基因', second,'交换')
            way='基因'+str(first)+'和基因'+str(second)+'交换'
            # print('交叉后基因',week_map)
            # print('交叉后最佳基因', week_map_best)
        if hereditary_type>8 and hereditary_type<= 16:#10%的可能性复制
            type='基因复制'
            print(type)
            first=random.randint(0,11)
            second=random.randint(0,11)
            month_map[first]=month_map[second]
            extend_map()
            print(type,'把基因', second,'复制到',first)
            way='把基因'+str(second)+'复制到'+str(first)
        if hereditary_type>16:#80%的可能性变异
            type='基因变异'
            print(type)
            first=random.randint(0,11)
            second = random.randint(0, 11)
            variable=random.uniform(-variable_coefficient, variable_coefficient)
            month_map[first]=month_map[first]+variable
            #month_map[second] = month_map[second] + variable
            print(type,'变异值',variable,'变异哪个基因',first)
            way='变异值'+str(variable)+'变异基因为'+str(first)
            extend_map()

    count_pearsonr=0#计数：皮尔逊系数的结果中，中度显著相关的公司数
    correlation_sum_pearsonr=0#计数：皮尔逊系数的结果中，中度显著相关的相关系数之和
    pvalue_sum_pearsonr=0#计数：皮尔逊系数的结果中，中度显著相关的pvalue之和

    for root, dirs, filelist in os.walk("D:\\用电数据集\\特征工程加强\\用于特征选择的行业数据集_2020—2021每月取3周数据\\"):
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

                    #皮尔逊相关系数、斯皮尔曼相关系数、点二列相关
                    correlation_pearsonr, pvalue_pearsonr = stats.stats.pearsonr(month_map_all,ST_data['平均负荷(kW)'])#皮尔逊相关系数 周末0.82 星期映射 0.485059

                    if abs(correlation_pearsonr)>=0.45 and pvalue_pearsonr<0.05:
                        count_pearsonr += 1
                        #print(root)
                        #print(correlation_pearsonr)
                        #print(pvalue_pearsonr)
                        pvalue_sum_pearsonr+=pvalue_pearsonr
                        correlation_sum_pearsonr+=abs(correlation_pearsonr)
    if num==0:
        print("父代基因")
        count_pearsonr_best=count_pearsonr
        count_pearsonr_ratio_best=count_pearsonr/3992
        correlation_avg_best=correlation_sum_pearsonr/(count_pearsonr+1)
        pvalue_avg_best=pvalue_sum_pearsonr/(count_pearsonr+1)
        #week_map_best=week_map
        analysis_res=analysis_res.append([{'特征':feature,'采样公司总数':3992,'显著中度相关公司数_pearsonr':count_pearsonr,
        '显著中度相关公司数占比_pearsonr':(count_pearsonr/3992),'平均相关系数_pearsonr':(correlation_sum_pearsonr/(count_pearsonr+1)),
        '平均pvalue_pearsonr': (pvalue_sum_pearsonr/(count_pearsonr+1)),
                                           '后代基因':str(month_map),
                                           '显著中度相关公司数(最佳值)':count_pearsonr_best,
                                           '平均相关系数(最佳值)':correlation_avg_best,
                                           '遗传算子':'父代基因无遗传类型','改变方式':'无',
                                           '该基因是否遗传':'无','基因(最佳值)':str(month_map_best)}],ignore_index=True)
        month_map_all=[]
        print(month_map_best)
    elif (count_pearsonr >= count_pearsonr_best) and (((correlation_sum_pearsonr / (count_pearsonr + 1)) > correlation_avg_best)):
        print('历史最优相关系数',correlation_avg_best)
        print('优质基因')
        count_pearsonr_best = count_pearsonr
        count_pearsonr_ratio_best = (count_pearsonr / 3992)
        correlation_avg_best = (correlation_sum_pearsonr / (count_pearsonr + 1))
        pvalue_avg_best = (pvalue_sum_pearsonr / (count_pearsonr + 1))
        res_xs=(correlation_sum_pearsonr / (count_pearsonr + 1))
        print('相关系数', res_xs)
        month_map_best = month_map.copy()#不加copy会使得week_map_best、week_map两个变量指向同一个地址
        analysis_res = analysis_res.append([{'特征': feature, '采样公司总数': 3992, '显著中度相关公司数_pearsonr': count_pearsonr,
                                             '显著中度相关公司数占比_pearsonr': (count_pearsonr / 3992),
                                             '平均相关系数_pearsonr':res_xs ,
                                             '平均pvalue_pearsonr': (pvalue_sum_pearsonr / (count_pearsonr + 1)),
                                             '后代基因': str(month_map),
                                             '显著中度相关公司数(最佳值)': count_pearsonr_best,
                                             '平均相关系数(最佳值)': correlation_avg_best, '遗传算子': type, '改变方式':way,'该基因是否遗传': '是',
                                             '基因(最佳值)': str(month_map_best)}],ignore_index=True)

        month_map_all = []
        print(month_map_best)
        analysis_res.to_csv(path_or_buf='D:\\用电数据集\\特征工程加强\\月份特征学习_遗传规划算法.csv', encoding="utf_8_sig", index=False)
    else:
        print('历史最优相关系数', correlation_avg_best)
        print('非优质基因')
        #print(week_map)
        res_xs=(correlation_sum_pearsonr / (count_pearsonr + 1))
        print('相关系数',res_xs)
        analysis_res = analysis_res.append([{'特征': feature, '采样公司总数': 3992, '显著中度相关公司数_pearsonr': count_pearsonr,
                                             '显著中度相关公司数占比_pearsonr': (count_pearsonr / 3992),
                                             '平均相关系数_pearsonr': res_xs,
                                             '平均pvalue_pearsonr': (pvalue_sum_pearsonr / (count_pearsonr + 1)),
                                             '后代基因': str(month_map),
                                             '显著中度相关公司数(最佳值)': count_pearsonr_best,
                                             '平均相关系数(最佳值)': correlation_avg_best, '遗传算子': type, '改变方式':way,'该基因是否遗传': '否',
                                             '基因(最佳值)': str(month_map_best)}],ignore_index=True)
        month_map = month_map_best.copy()  # 复原week_map
        month_map_all = []
        print(month_map_best)
        if num%500==0:
            analysis_res.to_csv(path_or_buf='D:\\用电数据集\\特征工程加强\\月份特征学习_遗传规划算法.csv', encoding="utf_8_sig", index=False)

    #+1是为了消除除数为0的问题
print(analysis_res)
analysis_res.to_csv(path_or_buf='D:\\用电数据集\\特征工程加强\\月份特征学习_遗传规划算法.csv', encoding="utf_8_sig",index=False)
