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
area=["杭州市区","淳安县","富阳区","海宁市","海盐县","嘉善县",
      "嘉兴市桐乡市","建德市","临安区","平湖市","萧山区","余杭区"]

new_file='D:\\用电数据集\\归一化之前的数据集\\按地域划分数据集_特征增强\\'
if os.path.exists(new_file) == False:
    os.makedirs(new_file+'杭州市区')
    os.makedirs(new_file+'淳安县')
    os.makedirs(new_file+'富阳区')
    os.makedirs(new_file+'海宁市')
    os.makedirs(new_file+'海盐县')
    os.makedirs(new_file+'嘉善县')
    os.makedirs(new_file+'嘉兴市桐乡市')
    os.makedirs(new_file+'建德市')
    os.makedirs(new_file+'临安区')
    os.makedirs(new_file+'平湖市')
    os.makedirs(new_file+'萧山区')
    os.makedirs(new_file+'余杭区')

for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_特征增强"):
    if root.split("\\")[-2] in ["本市级高压","滨江供电分公司","城北供电分公司","城北供电分公司2",
                                "城南供电分公司","钱塘新区供电公司","西湖供电分公司"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = new_file+'杭州市区' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-2] in ["淳安县供电分公司1","淳安县供电分公司2","淳安县供电分公司3","淳安县供电分公司4",
                                "淳安县供电分公司5"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = new_file+'淳安县' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-2] in ["富阳区供电分公司","富阳区供电分公司2","富阳区供电分公司3","富阳区供电分公司4",
                                "富阳区供电分公司5"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = new_file+'富阳区' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-2] in ["海宁市供电分公司","海宁市供电分公司1","海宁市供电分公司2","海宁市供电分公司3",
                                "海宁市供电分公司4","海宁市供电分公司5"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = new_file+'海宁市' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-2] in ["海盐县供电分公司1","海盐县供电分公司2","海盐县供电分公司3","海盐县供电分公司4",
                                "海盐县供电分公司5"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = new_file+'海盐县' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-2] in ["嘉善县供电分公司1", "嘉善县供电分公司2", "嘉善县供电分公司3", "嘉善县供电分公司4",
                                "嘉善县供电分公司5", "嘉善县供电分公司6"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = new_file+'嘉善县' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-2] in ["嘉兴供电分公司1", "桐乡市供电分公司1"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = new_file+'嘉兴市桐乡市' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-2] in ["建德市供电分公司", "建德市供电分公司2", "建德市供电分公司3", "建德市供电分公司4",
                                "建德市供电分公司5", "建德市供电分公司6"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = new_file+'建德市' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-2] in ["临安区供电分公司", "临安区供电分公司2", "临安区供电分公司3", "临安区供电分公司4",
                                "临安区供电分公司5"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = new_file+'临安区' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-2] in ["平湖市供电分公司1", "平湖市供电分公司2", "平湖市供电分公司3", "平湖市供电分公司4",
                                "平湖市供电分公司5"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = new_file+'平湖市' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-2] in ["萧山区供电分公司", "萧山区供电分公司2", "萧山区供电分公司3", "萧山区供电分公司4",
                                "萧山区供电分公司5"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = new_file+'萧山区' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
    if root.split("\\")[-2] in ["余杭区供电分公司", "余杭区供电分公司2", "余杭区供电分公司3", "余杭区供电分公司4",
                                "余杭区供电分公司5", "余杭区供电分公司6", "余杭区供电分公司7"]:
        a = root.split("\\")[-2]
        b = root.split("\\")[-1]
        link_new = new_file+'余杭区' + "\\" + a + "\\" + b
        shutil.copytree(root, link_new)
#测试
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