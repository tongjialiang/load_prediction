import os
import shutil
import pandas as pd

#1.遍历所有文件夹，找到问题公司 存在MT,ST文件，而不存在RT
for root, dirs, filelist in os.walk("D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集_去异常"):
    # for i in filelist:
        #当前所遍历到的root D:\用电数据集\归一化之前的数据集\按地域划分数据集_去异常\临安区\临安区供电分公司2\175
        #遍历到该root下的目录列表 空
        #遍历到该root下的文件列表 ['MT_data.csv', 'ST_data.csv']
    if "ST_data.csv" in filelist and 'RT_data.csv' not in filelist:
        print(root)
        print(filelist)
        shutil.rmtree(root)#2.删除问题公司数据
#Remove_RTincomplete_data.py
    # 淳安县供电分公司4\98
    # 富阳区供电分公司2\102
    # 建德市供电分公司\7
    # 余杭区供电分公司4\5
    # 余杭区供电分公司7\166
    # 嘉善县供电分公司1\100
    # 建德市供电分公司5\147
    # 临安区供电分公司2\175
    # 嘉兴供电分公司1\6
    # 淳安县供电分公司2\61
    # 淳安县供电分公司4\140

    #         print(root)
        # if i=='MT_data.csv':
        #     MT_data_old=MT_data_old+1
        # if i == 'ST_data.csv':
        #     ST_data_old=ST_data_old+1
        # if i == 'RT_data.csv':
        #     RT_data_old=RT_data_old+1