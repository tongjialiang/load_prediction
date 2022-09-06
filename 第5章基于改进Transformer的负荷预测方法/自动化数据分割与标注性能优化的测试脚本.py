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
import gc
print("start")
with open("d:\\数据采样完成\\test_class1_RT_data_48_1-old.pk", 'rb') as f:
    data1 = pickle.load(f)
    #print(data1)
with open("d:\\数据采样完成\\test_class1_RT_data_48_1-gcsf.pk", 'rb') as f:
    data2 = pickle.load(f)
    #print(data2)
print("验证数据量")
print(len(data1[0]))
print(len(data2[0]))
print(len(data1[1]))
print(len(data2[1]))
print("验证shape")
print(data1[0].shape)
print(data2[0].shape)
print(data1[1].shape)
print(data2[1].shape)
print("验证最后一个数据")
print(data1[0][-1])
print(data2[0][-1])
print(data1[1][-1])
print(data2[1][-1])
# a=10
# a=100
# print(gc.collect())

# with open("D:\\dict_myclass_MeanAndStd(normal).pk", 'rb') as f:
# #     data = pickle.load(f)
#     print(len(data))#'class6': array(['富阳区供电分公司2_15'], dtype='<U13')
#存放每个类的公司个数
# count_class={}
# for k in data.keys():
#     count_class.update({k:len(data[k])})
# print(count_class)

# with open("D:/数据预处理完成后的样本/长期用电数据/淳安县-MT_data_12_3.pk", 'rb') as f:
#     data = pickle.load(f)
#     print(data[0])
#     print("33333333333333333333333333")
#     print(data[1])
#     print(data[0].shape)
#     print(data[1].shape)
#     print(len(data[0]))
#     print(len(data[1]))

# with open("E:/数据预处理完成后的样本/实时用电数据/淳安县-RT_data_48_1.pk", 'rb') as f:
#     data = pickle.load(f)
#     print(data[0])
#     print("33333333333333333333333333")
#     print(data[1])
#     print(data[0].shape)
#     print(data[1].shape)
#     print(len(data[0]))
#     print(len(data[1]))
#
# a=np.array([10,10])
# print(a*[10,1])
# k=9
# def aaaa():
#     c=1
#     k=9
#
#
#
#
#
#
# def bbbb():
#     k=99999
#     c=0
#     aaaa()
#     print(k)
#     k=k+1
#     print(k)
# bbbb()


