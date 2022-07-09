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


def do_GetClusteringXandBusname_rt():
    clustering_samples = {}  # 存放聚类样本X的字典，键：公司名称 值：X=[均值、标准差]
    business_name = []
    with open('d:/dict_BusTypeOrArea_MeanAndStd_method2.pk', 'rb') as f:
        data = pickle.load(f)
    # print(len(data))#9661
    #print(data)
    x = []  # 聚类样本
    # 把每个企业的[均值、标准差] 作为聚类的样本；均值和标准差优先取月的，其次取日的，再取实时的。
    for area_name in data:
        res = [0,0]
        # if "mt_mean" in data[area_name]:
        #     res[0]=data[area_name]["mt_mean"]
        #     res[1]=data[area_name]["mt_std"]
        # elif "st_mean" in data[area_name]:
        #     res[0]=data[area_name]["st_mean"]
        #     res[1] = data[area_name]["st_std"]
        # elif "rt_mean" in data[area_name]:
        #     res[0]=data[area_name]["rt_mean"]
        #     res[1] = data[area_name]["rt_std"]

        if "rt_mean" in data[area_name]:
            res[0] = data[area_name]["rt_mean"]
            res[1] = data[area_name]["rt_std"]
        elif "st_mean" in data[area_name]:
            res[0] = data[area_name]["st_mean"]
            res[1] = data[area_name]["st_std"]
        elif "mt_mean" in data[area_name]:
            res[0] = data[area_name]["mt_mean"]
            res[1] = data[area_name]["mt_std"]
        clustering_samples[area_name] = res
    print(clustering_samples)
    print(len(clustering_samples))  # 9661
    # all_previous_data = np.concatenate([all_previous_data, previous_data], axis=0)
    # 生成聚类的样本
    flag = 0
    for i in clustering_samples:
        now = np.array([clustering_samples[i]])
        # print(now)
        if flag == 0:
            x = now
            flag = 1
            business_name.append(i)
            continue
        x = np.concatenate([x, now], axis=0)
        business_name.append(i)
    # print(x)
    #print(x.shape)
    #print(len(business_name))

    # #[87.4303931049253, 16.873998275300476]
    # for index,i in enumerate(data):
    #     if i=='萧山区供电分公司2_7':
    #         print(index)
    #         print(data[i])
    # for index,i in enumerate(clustering_samples):
    #     if i=='萧山区供电分公司2_7':
    #         print(index)
    #         print(clustering_samples[i])
    # print(x[9201])
    # print(clustering_samples['萧山区供电分公司2_7'])
    return x, np.array(business_name)
