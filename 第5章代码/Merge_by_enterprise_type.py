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
#构建目录
type=["采矿业","电力热力燃气及水生产和供应业","房地产业","公共管理社会保障和社会组织","建筑业","交通运输仓储和邮政业",
      "教育","金融业","居民服务修理和其他服务业","科学研究和技术服务业","农林牧渔业","批发和零售业"
    ,"水利环境和公共设施管理业","卫生和社会工作","文化体育和娱乐业","信息传输软件和信息技术服务业"
    ,"制造业_机械电子制造业","制造业_轻纺工业","制造业_资源加工工业","住宿和餐饮业","租赁和商务服务业"]

if os.path.exists('D:\\按行业划分数据集2\\金融业与房地产业与租赁和商务服务业') == False:
    os.makedirs('D:\\按行业划分数据集2\\金融业与房地产业与租赁和商务服务业')
    os.makedirs('D:\\按行业划分数据集2\\科学研究和技术服务业与信息传输软件与信息技术服务业')
    os.makedirs('D:\\按行业划分数据集2\\卫生和社会工作与公共管理社会保障和社会组织')



for root, dirs, filelist in os.walk("D:\\按行业划分数据集\\"):
    if root.split("\\")[-3] in ["金融业","租赁和商务服务业","房地产业"]:
        #print(root) #D:\按行业划分数据集\金融业\余杭区供电分公司5\139
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = 'D:\\按行业划分数据集2\\金融业与房地产业与租赁和商务服务业' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-3] in ["科学研究和技术服务业","信息传输软件和信息技术服务业"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = 'D:\\按行业划分数据集2\\科学研究和技术服务业与信息传输软件与信息技术服务业' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-3] in ["卫生和社会工作","公共管理社会保障和社会组织"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = 'D:\\按行业划分数据集2\\卫生和社会工作与公共管理社会保障和社会组织' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-3] not in ["金融业","租赁和商务服务业","房地产业"
        ,"科学研究和技术服务业","信息传输软件和信息技术服务业","卫生和社会工作","公共管理社会保障和社会组织"]:
            if root.split("\\")[-3] in type:
                a = root.split("\\")[-2]
                b = root.split("\\")[-1]
                c = root.split("\\")[-3]
                link_new = "D:\\按行业划分数据集2\\" + c + "\\" + a + "\\" + b
                shutil.copytree(root, link_new)





# #测试
# count_before=0
# count_after=0
# for root, dirs, filelist in os.walk("D:\\按地域划分数据集\\"):
#     for i in filelist:
#         if i =="RT_data.csv":
#             count_after+=1
# for root, dirs, filelist in os.walk("D:\\用电数据集_已完成数据分类\\"):
#     for i in filelist:
#         if i =="RT_data.csv":
#             count_before+=1
# print(count_before)
# print(count_after)