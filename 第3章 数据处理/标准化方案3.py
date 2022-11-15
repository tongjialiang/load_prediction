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
#读取均值和方差
# with open('D:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按地域划分数据集V2.pk', 'rb') as f:
#     res = pickle.load(f)
# print(res)
# file_name="D:\\用电数据集\\归一化之前的数据集\\按地域划分数据集V2\\"
# file_name_new="D:\\用电数据集\\归一化之后的数据集-待采样\\按地域划分数据集V2\\"

# with open('D:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按行业划分数据集V2.pk', 'rb') as f:
#     res = pickle.load(f)
# print(res)
# file_name="D:\\用电数据集\\归一化之前的数据集\\按行业划分数据集V2\\"
# file_name_new="D:\\用电数据集\\归一化之后的数据集-待采样\\按行业划分数据集V2\\"

# with open('D:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按聚类划分数据集2V2.pk', 'rb') as f:
#     res = pickle.load(f)
# print(res)
# file_name="D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集(行业划分后聚类)V2\\"
# file_name_new="D:\\用电数据集\\归一化之后的数据集-待采样\\按聚类划分数据集(行业划分后聚类)V2\\"


# 各特征的均值和方差_按负荷特性聚类划分数据集V2.pk
# 各特征的均值和方差_按聚类划分数据集(频域分解动态规整)V2.pk
# with open('D:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按负荷特性聚类划分数据集V2.pk', 'rb') as f:
#     res = pickle.load(f)
# print(res)
# file_name="D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集(负荷特性)V2\\"
# file_name_new="D:\\用电数据集\\归一化之后的数据集-待采样\\按聚类划分数据集(负荷特性)V2\\"

with open('D:\\实验记录\\重要结果文件\\pk\\各特征的均值和方差_按聚类划分数据集(频域分解动态规整)V2.pk', 'rb') as f:
    res = pickle.load(f)
print(res)
file_name="D:\\用电数据集\\归一化之前的数据集\\按聚类划分数据集(频域分解动态规整)V2\\"
file_name_new="D:\\用电数据集\\归一化之后的数据集-待采样\\按聚类划分数据集(频域分解动态规整)V2\\"
#对按地域分类的数据做全量标准化
for root, dirs, filelist in os.walk(file_name):
        for i in filelist:
            #filename=root+"\\"+i
            #print(filename)
            if i == 'RT_data.csv':
                #print(i)#RT_data.csv
                #print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    rtdata = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    rtdata = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                #获取当前地区的均值和方差值
                area_type=root.split("\\")[-3]

                if rtdata.columns.__contains__('瞬时有功(kW)'):
                    a='rt_mean_ssyg'
                    b='rt_std_ssyg'
                    c='瞬时有功(kW)'
                    if res[area_type][b]<0.01:
                        rtdata.loc[:,c] =0.01
                    else:
                        rtdata.loc[:,c]=(rtdata.loc[:,c]-res[area_type][a])/res[area_type][b]
                if rtdata.columns.__contains__('平均气温'):
                    a='rt_mean_pjqw'
                    b='rt_std_pjqw'
                    c='平均气温'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if rtdata.columns.__contains__('按行业分的法人单位数_p29'):
                    a='rt_mean_29'
                    b='rt_std_29'
                    c = '按行业分的法人单位数_p29'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if rtdata.columns.__contains__('按行业分的全省生产总值_亿元_p18'):
                    a='rt_mean_18'
                    b='rt_std_18'
                    c = '按行业分的全省生产总值_亿元_p18'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if rtdata.columns.__contains__('分行业全社会单位就业人员年平均工资_p163'):
                    a='rt_mean_163'
                    b='rt_std_163'
                    c = '分行业全社会单位就业人员年平均工资_p163'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if rtdata.columns.__contains__('按行业分全社会用电情况_亿千瓦时_p305'):
                    a='rt_mean_305'
                    b='rt_std_305'
                    c = '按行业分全社会用电情况_亿千瓦时_p305'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if rtdata.columns.__contains__('按地区分组的法人单位数(万人)_p35'):
                    a='rt_mean_35'
                    b='rt_std_35'
                    c = '按地区分组的法人单位数(万人)_p35'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if rtdata.columns.__contains__('总人口数(万人)_p46'):
                    a='rt_mean_46'
                    b='rt_std_46'
                    c = '总人口数(万人)_p46'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if rtdata.columns.__contains__('各市规模以上企业年末单位就业人员(万人)_p72'):
                    a='rt_mean_72'
                    b='rt_std_72'
                    c = '各市规模以上企业年末单位就业人员(万人)_p72'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if rtdata.columns.__contains__('生产总值(百亿元)_p518_p539'):
                    a='rt_mean_518'
                    b='rt_std_518'
                    c = '生产总值(百亿元)_p518_p539'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if rtdata.columns.__contains__('全年用电量_百亿千瓦时_p529_p571'):
                    a='rt_mean_529'
                    b='rt_std_529'
                    c = '全年用电量_百亿千瓦时_p529_p571'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if rtdata.columns.__contains__('第一产业(百亿元)_p537_p539'):
                    a='rt_mean_537_1'
                    b='rt_std_537_1'
                    c = '第一产业(百亿元)_p537_p539'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if rtdata.columns.__contains__('第二产业(百亿元)_p537_p539'):
                    a='rt_mean_537_2'
                    b='rt_std_537_2'
                    c = '第二产业(百亿元)_p537_p539'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if rtdata.columns.__contains__('第三产业(百亿元)_p537_p539'):
                    a='rt_mean_537_3'
                    b='rt_std_537_3'
                    c = '第三产业(百亿元)_p537_p539'
                    if res[area_type][b] < 0.01:
                        rtdata.loc[:, c] = 0.01
                    else:
                        rtdata.loc[:, c] = (rtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                # rt_mean=res[area_type]["rt_mean"]
                # rt_std = res[area_type]["rt_std"]


                rtdata["standard_key"] = area_type#归一化查询字段
                order = rtdata.columns#把该字段放到数据集的第二个字段中
                order = order.insert(1, order[-1])
                order = order.delete(-1)
                rtdata=rtdata[order]
                #存入新的目录
                link=file_name_new+area_type+"\\"+root.split("\\")[-2]+"\\"+root.split("\\")[-1]+"\\"
                print(link)
                if os.path.exists(link) == False:
                    os.makedirs(link)
                linknew=link+i
                rtdata.to_csv(path_or_buf=linknew, encoding="utf_8_sig", index=False)
            if i == 'MT_data.csv':
                # print(i)#RT_data.csv
                # print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    mtdata = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    mtdata = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                # 获取当前地区的均值和方差值
                area_type = root.split("\\")[-3]

                if mtdata.columns.__contains__('平均负荷(kW)'):
                    a='mt_mean_pjfh'
                    b='mt_std_pjfh'
                    c='平均负荷(kW)'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]

                if mtdata.columns.__contains__('受电容量(KVA)'):
                    a='mt_mean_sdrl'
                    b='mt_std_sdrl'
                    c='受电容量(KVA)'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('全省能源生产量(百万吨标准煤)'):
                    a='mt_mean_qsnyscl'
                    b='mt_std_qsnyscl'
                    c='全省能源生产量(百万吨标准煤)'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('全省电力生产量(百亿千瓦小时)'):
                    a='mt_mean_qsdlscl'
                    b='mt_std_qsdlscl'
                    c='全省电力生产量(百亿千瓦小时)'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('能源生产比上年增长(%)'):
                    a='mt_mean_nyscbsnzz'
                    b='mt_std_nyscbsnzz'
                    c='能源生产比上年增长(%)'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('全省能源消费量(百万吨标准煤)'):
                    a='mt_mean_qsnyxfl'
                    b='mt_std_qsnyxfl'
                    c='全省能源消费量(百万吨标准煤)'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('全省电力消费量(百亿千瓦小时)'):
                    a='mt_mean_qsdlxfl'
                    b='mt_std_qsdlxfl'
                    c = '全省电力消费量(百亿千瓦小时)'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('按行业分的法人单位数_p29'):
                    a='mt_mean_29'
                    b='mt_std_29'
                    c='按行业分的法人单位数_p29'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('按行业分的全省生产总值_亿元_p18'):
                    a='mt_mean_18'
                    b='mt_std_18'
                    c='按行业分的全省生产总值_亿元_p18'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('分行业全社会单位就业人员年平均工资_p163'):
                    a='mt_mean_163'
                    b='mt_std_163'
                    c='分行业全社会单位就业人员年平均工资_p163'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('按行业分全社会用电情况_亿千瓦时_p305'):
                    a='mt_mean_305'
                    b='mt_std_305'
                    c='按行业分全社会用电情况_亿千瓦时_p305'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('按地区分组的法人单位数(万人)_p35'):
                    a='mt_mean_35'
                    b='mt_std_35'
                    c='按地区分组的法人单位数(万人)_p35'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('总人口数(万人)_p46'):
                    a='mt_mean_46'
                    b='mt_std_46'
                    c='总人口数(万人)_p46'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('各市规模以上企业年末单位就业人员(万人)_p72'):
                    a='mt_mean_72'
                    b='mt_std_72'
                    c='各市规模以上企业年末单位就业人员(万人)_p72'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('生产总值(百亿元)_p518_p539'):
                    a='mt_mean_518'
                    b='mt_std_518'
                    c='生产总值(百亿元)_p518_p539'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('全年用电量_百亿千瓦时_p529_p571'):
                    a='mt_mean_529'
                    b='mt_std_529'
                    c='全年用电量_百亿千瓦时_p529_p571'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('第一产业(百亿元)_p537_p539'):
                    a='mt_mean_537_1'
                    b='mt_std_537_1'
                    c='第一产业(百亿元)_p537_p539'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('第二产业(百亿元)_p537_p539'):
                    a='mt_mean_537_2'
                    b='mt_std_537_2'
                    c='第二产业(百亿元)_p537_p539'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if mtdata.columns.__contains__('第三产业(百亿元)_p537_p539'):
                    a='mt_mean_537_3'
                    b='mt_std_537_3'
                    c='第三产业(百亿元)_p537_p539'
                    if res[area_type][b] < 0.01:
                        mtdata.loc[:, c] = 0.01
                    else:
                        mtdata.loc[:, c] = (mtdata.loc[:, c] - res[area_type][a]) / res[area_type][b]


                mtdata["standard_key"] = area_type
                order = mtdata.columns#把该字段放到数据集的第二个字段中
                order = order.insert(1, order[-1])
                order = order.delete(-1)
                mtdata=mtdata[order]
                # 存入新的目录
                link = file_name_new + area_type + "\\" + root.split("\\")[-2] + "\\" + root.split("\\")[
                    -1] + "\\"
                print(link)
                if os.path.exists(link) == False:
                    os.makedirs(link)
                linknew = link + i
                mtdata.to_csv(path_or_buf=linknew, encoding="utf_8_sig", index=False)

            if i == 'ST_data.csv':
                # print(i)#RT_data.csv
                # print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                try:
                    stdata = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    stdata = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                # 获取当前地区的均值和方差值
                area_type = root.split("\\")[-3]

                if stdata.columns.__contains__('平均负荷(kW)'):
                    a='st_mean_pjfh'
                    b='st_std_pjfh'
                    c='平均负荷(kW)'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('受电容量(KVA)'):
                    a='st_mean_sdrl'
                    b='st_std_sdrl'
                    c='受电容量(KVA)'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('平均气温'):
                    a='st_mean_pjqw'
                    b='st_std_pjqw'
                    c='平均气温'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('全省能源生产量(百万吨标准煤)'):
                    a='st_mean_qsnyscl'
                    b='st_std_qsnyscl'
                    c='全省能源生产量(百万吨标准煤)'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('全省电力生产量(百亿千瓦小时)'):
                    a='st_mean_qsdlscl'
                    b='st_std_qsdlscl'
                    c='全省电力生产量(百亿千瓦小时)'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('能源生产比上年增长(%)'):
                    a='st_mean_nyscbsnzz'
                    b='st_std_nyscbsnzz'
                    c='能源生产比上年增长(%)'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('全省能源消费量(百万吨标准煤)'):
                    a='st_mean_qsnyxfl'
                    b='st_std_qsnyxfl'
                    c='全省能源消费量(百万吨标准煤)'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('全省电力消费量(百亿千瓦小时)'):
                    a='st_mean_qsdlxfl'
                    b='st_std_qsdlxfl'
                    c='全省电力消费量(百亿千瓦小时)'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('按行业分的法人单位数_p29'):
                    a='st_mean_29'
                    b='st_std_29'
                    c='按行业分的法人单位数_p29'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('按行业分的全省生产总值_亿元_p18'):
                    a='st_mean_18'
                    b='st_std_18'
                    c='按行业分的全省生产总值_亿元_p18'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('分行业全社会单位就业人员年平均工资_p163'):
                    a='st_mean_163'
                    b='st_std_163'
                    c='分行业全社会单位就业人员年平均工资_p163'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('按行业分全社会用电情况_亿千瓦时_p305'):
                    a='st_mean_305'
                    b='st_std_305'
                    c = '按行业分全社会用电情况_亿千瓦时_p305'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('按地区分组的法人单位数(万人)_p35'):
                    a='st_mean_35'
                    b='st_std_35'
                    c='按地区分组的法人单位数(万人)_p35'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('总人口数(万人)_p46'):
                    a='st_mean_46'
                    b='st_std_46'
                    c='总人口数(万人)_p46'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('各市规模以上企业年末单位就业人员(万人)_p72'):
                    a='st_mean_72'
                    b='st_std_72'
                    c='各市规模以上企业年末单位就业人员(万人)_p72'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('生产总值(百亿元)_p518_p539'):
                    a='st_mean_518'
                    b='st_std_518'
                    c='生产总值(百亿元)_p518_p539'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('全年用电量_百亿千瓦时_p529_p571'):
                    a='st_mean_529'
                    b='st_std_529'
                    c='全年用电量_百亿千瓦时_p529_p571'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('第一产业(百亿元)_p537_p539'):
                    a='st_mean_537_1'
                    b='st_std_537_1'
                    c='第一产业(百亿元)_p537_p539'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('第二产业(百亿元)_p537_p539'):
                    a='st_mean_537_2'
                    b='st_std_537_2'
                    c='第二产业(百亿元)_p537_p539'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]
                if stdata.columns.__contains__('第三产业(百亿元)_p537_p539'):
                    a='st_mean_537_3'
                    b='st_std_537_3'
                    c='第三产业(百亿元)_p537_p539'
                    if res[area_type][b] < 0.01:
                        stdata.loc[:, c] = 0.01
                    else:
                        stdata.loc[:, c] = (stdata.loc[:, c] - res[area_type][a]) / res[area_type][b]


                stdata["standard_key"] = area_type
                order = stdata.columns#把该字段放到数据集的第二个字段中
                order = order.insert(1, order[-1])
                order = order.delete(-1)
                stdata=stdata[order]
                # 存入新的目录
                link = file_name_new + area_type + "\\" + root.split("\\")[-2] + "\\" + root.split("\\")[
                    -1] + "\\"
                print(link)
                if os.path.exists(link) == False:
                    os.makedirs(link)
                linknew = link + i
                stdata.to_csv(path_or_buf=linknew, encoding="utf_8_sig", index=False)


