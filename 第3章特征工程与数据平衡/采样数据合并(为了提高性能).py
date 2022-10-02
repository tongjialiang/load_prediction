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
import gc
warnings.filterwarnings("ignore")
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

def do_sample_merge(input1_dir,input2_dir,output_dir):
    with open(input1_dir, 'rb') as f:
        data1 = pickle.load(f)
        print(data1[0].shape)
        print(data1[1].shape)
    f.close()
    gc.collect()

    with open(input2_dir, 'rb') as f:
        data2 = pickle.load(f)
        print(data2[0].shape)
        print(data2[1].shape)
    f.close()
    gc.collect()

    data3=[np.vstack((data1[0], data2[0])),np.vstack((data1[1], data2[1]))]
    print(data3[0].shape)
    print(data3[1].shape)
    del data1
    del data2
    gc.collect()
    with open(output_dir, 'wb+') as f:
        pickle.dump(data3, f)
    f.close()
    gc.collect()
    return 1



#测试
# do_sample_merge('D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_RT_data_48_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_3_RT_data_48_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_RT_data_48_1.pk')
#
# do_sample_merge('D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_RT_data_48_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_4_RT_data_48_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_RT_data_48_1.pk')
#
# do_sample_merge('D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_RT_data_48_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_5_RT_data_48_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_RT_data_48_1.pk')
# with open('d:/classres.pk', 'rb') as f:
#     data1 = pickle.load(f)
#     print(data1[0].shape)
#     print(data1[1].shape)
# do_sample_merge('D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_1_ST_data_30_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_2_ST_data_30_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_ST_data_30_1.pk')
#
# do_sample_merge('D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_ST_data_30_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_3_ST_data_30_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_ST_data_30_1.pk')
#
# do_sample_merge('D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_ST_data_30_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_4_ST_data_30_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_ST_data_30_1.pk')
#
# do_sample_merge('D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_ST_data_30_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_5_ST_data_30_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c3_class1_ST_data_30_1.pk')

# do_sample_merge('D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_stclass_27_1_MT_data_9_3.pk'
#                 ,'D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_stclass_27_2_MT_data_9_3.pk'
#                 ,'D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_stclass_27_MT_data_9_3.pk')
#
# do_sample_merge('D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_stclass_27_1_ST_data_100_10.pk'
#                 ,'D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_stclass_27_2_ST_data_100_10.pk'
#                 ,'D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_stclass_27_ST_data_100_10.pk')


do_sample_merge('D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_rtclass14_RT_data_96_20.pk'
                ,'D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_rtclass26_RT_data_96_20.pk'
                ,'D:\\数据采样完成new\\1.pk')
do_sample_merge('D:\\数据采样完成new\\1.pk'
                ,'D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_rtclass7_RT_data_96_20.pk'
                ,'D:\\数据采样完成new\\2.pk')
do_sample_merge('D:\\数据采样完成new\\2.pk'
                ,'D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_rtclass4_RT_data_96_20.pk'
                ,'D:\\数据采样完成new\\3.pk')
do_sample_merge('D:\\数据采样完成new\\3.pk'
                ,'D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_rtclass27_RT_data_96_20.pk'
                ,'D:\\数据采样完成new\\4.pk')
do_sample_merge('D:\\数据采样完成new\\4.pk'
                ,'D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_rtclass18_RT_data_96_20.pk'
                ,'D:\\数据采样完成new\\5.pk')
do_sample_merge('D:\\数据采样完成new\\5.pk'
                ,'D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_rtclass20_RT_data_96_20.pk'
                ,'D:\\数据采样完成new\\6.pk')
do_sample_merge('D:\\数据采样完成new\\6.pk'
                ,'D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_rtclass25_RT_data_96_20.pk'
                ,'D:\\数据采样完成new\\7.pk')
do_sample_merge('D:\\数据采样完成new\\7.pk'
                ,'D:\\数据采样完成new\\按聚类划分数据集_方案1-汇总标准化_rtclass33_RT_data_96_20.pk'
                ,'D:\\数据采样完成new\\未做聚类的数据集.pk')
# do_sample_merge('D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c25_class3_RT_data_48_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c25_class3_3_RT_data_48_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c25_class3_RT_data_48_1.pk')
#
#
# do_sample_merge('D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c25_class3_1_ST_data_30_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c25_class3_2_ST_data_30_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c25_class3_ST_data_30_1.pk')
#
# do_sample_merge('D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c25_class3_ST_data_30_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c25_class3_3_ST_data_30_1.pk'
#                 ,'D:\\数据采样完成\\按聚类划分数据集_方案1-汇总标准化_c25_class3_ST_data_30_1.pk')

