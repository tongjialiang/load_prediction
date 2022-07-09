import os
import shutil
import datetime
import time
import numpy as np
import pandas as pd
import scipy.stats
import pickle
import pprint
import gc
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 500)
pd.set_option('max_colwidth',100)
np.set_printoptions(threshold = np.inf)
# import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#盘符
res_all=[]
disk="E:\\用电数据集备份\\归一化之后的数据集备份\\"
#拼接待处理的数据
# dir1=["按地域划分数据集V2_方案1-汇总标准化"]
# h1=os.listdir(disk+"按地域划分数据集V2_方案1-汇总标准化\\")
# print(h1)
# dir2=["按行业划分数据集V2_方案1-汇总标准化"]
# h2=os.listdir(disk+"按行业划分数据集V2_方案1-汇总标准化\\")
# print(h2)
# dir3= ["按聚类划分数据集V2_方案2-行业划分后聚类"]
# h3= os.listdir(disk+"按聚类划分数据集V2_方案2-行业划分后聚类\\")
# print(h3)

# dir4=["组合建模与区间估计地域_方案1_汇总标准化"]
# h4=os.listdir(disk+"组合建模与区间估计地域_方案1_汇总标准化\\")
# dir5=["组合建模与区间估计聚类_方案1_汇总标准化"]
# h5=os.listdir(disk+"组合建模与区间估计聚类_方案1_汇总标准化\\")
# dir6=["组合建模与区间估计行业_方案1_汇总标准化"]
# h6=os.listdir(disk+"组合建模与区间估计行业_方案1_汇总标准化\\")
dir7=["区间估计_方案1_汇总标准化"]
h7=os.listdir(disk+"区间估计_方案1_汇总标准化\\")
# dir8=["不划分数据集V2_方案1-汇总标准化"]
# h8=os.listdir(disk+"不划分数据集V2_方案1-汇总标准化\\")
# dir9=["按负荷特性聚类划分数据集V2_方案1-汇总标准化"]
# h9=os.listdir(disk+"按负荷特性聚类划分数据集V2_方案1-汇总标准化\\")
# dir10=["按频域分解聚类划分数据集V2_方案1-汇总标准化"]
# h10=os.listdir(disk+"按频域分解聚类划分数据集V2_方案1-汇总标准化\\")
# 1/0
# dir5=["test_ok"]
# h5=["class1"]
# h1=[]
# h2=[]
# h3=[]
# h4=[]

linknew="c:\\数据采样完成new\\"

# if len(h3)>0:
#     for i in dir3:
#         for j in h3:
#             res_all.append([i,j])
# if len(h1)>0:
#     for i in dir1:
#         for j in h1:
#             res_all.append([i,j])
# if len(h2)>0:
#     for i in dir2:
#         for j in h2:
#             res_all.append([i,j])

# if len(h4)>0:
#     for i in dir4:
#         for j in h4:
#             res_all.append([i,j])
# if len(h5)>0:
#     for i in dir5:
#         for j in h5:
#             res_all.append([i,j])
# if len(h6)>0:
#     for i in dir6:
#         for j in h6:
#             res_all.append([i,j])
if len(h7)>0:
    for i in dir7:
        for j in h7:
            res_all.append([i,j])
# if len(h8)>0:
#     for i in dir8:
#         for j in h8:
#             res_all.append([i,j])
# if len(h9)>0:
#     for i in dir9:
#         for j in h9:
#             res_all.append([i,j])
# if len(h10)>0:
#     for i in dir10:
#         for j in h10:
#             res_all.append([i,j])
print(res_all)#['按行业划分数据集_方案1-汇总标准化', '交通运输仓储和邮政业']
# 1/0
def read_data(filename, look_back, pred_len,whichcsv):
    '''
    filename
    columns: list
    look_back:使用多少天前的数据
    pred_len:预测多少天后的数据
    '''
    try:
        df = pd.read_csv(filename, encoding='utf-8', sep=',')
    except:
        df = pd.read_csv(filename, encoding='gbk', sep=',')
    #print(df.head())
    value = df.values#返回不带索引的numpy数组
    # print(value)
    # 1/0
    all_previous_data = []
    all_future_data = []

    for i in range(len(value)-look_back-pred_len+1):#如果range(n),n<=0,则不执行，程序不会报错，函数返回2个空列表
        previous_data = value[i:i+look_back]#0-3
        future_data = value[i + look_back:i + look_back + pred_len]#4 5
        # print(previous_data)
        # 1/0

    #去除时间上不连续的样本
        if whichcsv=="RT_data.csv":
            # 样本输入的第一个时间
            firsttime = previous_data[0][0]
            #print(type(firsttime))  # string
            firsttime = firsttime.strip('\t')  # 删除末尾换行符
            firsttime_py = pd.to_datetime(firsttime,format="%Y-%m-%d %H:%M:%S")
            #firsttime_py = datetime.datetime.strptime(firsttime, "%Y-%m-%d %H:%M:%S")
            #print("输入X的第一个时间 "+str(firsttime_py))
            # 样本输出的最后一个时间
            #print(future_data)
            future_data_last=future_data[pred_len-1][0]
            future_data_last = future_data_last.strip('\t')  # 删除末尾换行符
            #future_data_last_py = datetime.datetime.strptime(future_data_last, "%Y-%m-%d %H:%M:%S")
            future_data_last_py = pd.to_datetime(future_data_last, format="%Y-%m-%d %H:%M:%S")
            #print(future_data_last_py)
            #样本输出Y的最后一个时间-样本输入X的第一个时间
            # print(future_data_last_py)
            # print(firsttime_py)
            diff=future_data_last_py-firsttime_py
            # print(diff)
            # print(type(diff))
            #print(diff.total_seconds()==900*(look_back+pred_len-1))
            # print(900 * (look_back + pred_len - 1))
            # print(diff.seconds)
            res=(diff.total_seconds()-900*(look_back+pred_len-1))<3600*3
            #print(all_future_data)
            #previous_data=np.delete(previous_data, [0], axis=1).squeeze()
            #future_data=np.delete(future_data, [0], axis=1).squeeze()
            previous_data = previous_data.squeeze()
            future_data = future_data.squeeze()
            #print(previous_data)
            #print(future_data)

        if whichcsv=="MT_data.csv":
            #print(previous_data)
            # 样本输入的第一个时间
            firsttime = previous_data[0][0]
            # print(firsttime)  # string
            # 1/0
            firsttime = firsttime.strip('\t')  # 删除末尾换行符
            #firsttime_py = pd.to_datetime(firsttime,format='%b-%y')
            firsttime_py = pd.to_datetime(firsttime, format='%Y-%m-%d')
            #print("输入X的第一个时间 "+str(firsttime_py))
            # 样本输出的最后一个时间
            #print(future_data)
            future_data_last=future_data[pred_len-1][0]
            future_data_last = future_data_last.strip('\t')  # 删除末尾换行符
            #future_data_last_py = pd.to_datetime(future_data_last, format='%b-%y')
            future_data_last_py = pd.to_datetime(future_data_last, format='%Y-%m-%d')
            #print("样本输出Y的最后一个时间")
            #datetime.timedelta()
            #print(future_data_last_py)
            #样本输出Y的最后一个时间-样本输入X的第一个时间

            diff=(firsttime_py.year - future_data_last_py.year) * 12 + (firsttime_py.month - future_data_last_py.month)
            diff=-diff
            #print(diff)
            #print(diff==1*(look_back+pred_len-1))
            res=(diff-1*(look_back+pred_len-1))<=1
            # print(res)
            # 1/0
            #print(all_future_data)
            #previous_data=np.delete(previous_data, [0], axis=1).squeeze()
            #future_data=np.delete(future_data, [0], axis=1).squeeze()
            previous_data = previous_data.squeeze()
            future_data = future_data.squeeze()
            #print(previous_data)
            #print(future_data)

        if whichcsv == "ST_data.csv":
            #print(previous_data)
            # 样本输入的第一个时间
            firsttime = previous_data[0][0]
            # print(type(firsttime))  # string
            firsttime = firsttime.strip('\t')  # 删除末尾换行符
            firsttime_py = pd.to_datetime(firsttime, format='%Y-%m-%d')
            #print("输入X的第一个时间 " + str(firsttime_py))
            # 样本输出的最后一个时间
            #print(future_data)
            future_data_last = future_data[pred_len - 1][0]
            future_data_last = future_data_last.strip('\t')  # 删除末尾换行符
            future_data_last_py = pd.to_datetime(future_data_last, format='%Y-%m-%d')
            #print("样本输出Y的最后一个时间")
            #datetime.timedelta()
            #print(future_data_last_py)
            # 样本输出Y的最后一个时间-样本输入X的第一个时间

            diff =  future_data_last_py-firsttime_py
            #print(diff.total_seconds() == 3600*24* (look_back + pred_len - 1))
            res = ((diff.total_seconds()) - (24*3600 * (look_back + pred_len - 1)))<7*3600*24
            # print(all_future_data)
            previous_data=previous_data.squeeze()
            future_data = future_data.squeeze()
            #previous_data = np.delete(previous_data, [0], axis=1).squeeze()
            #future_data = np.delete(future_data, [0], axis=1).squeeze()
            # print(previous_data)
            # print(future_data)

        if res:
            all_previous_data.append(previous_data)
            all_future_data.append(future_data)
            res = ''

    return np.array(all_previous_data), np.array(all_future_data)


#C:\按地域划分数据集
#read_from_dir_RT_data(disk+p[0]+"\\"+p[1], [0,1,11,12],100,10, "ST_data.csv")
def read_from_dir_RT_data(dir, look_back, pred_len,whichcsv):
    all_previous_data2=''
    all_future_data2=''
    previous_list=[]
    future_list=[]
    flag = 0
    for root, dirs, filelist in os.walk(dir):
        for i in filelist:

            if i == whichcsv :  # 'MT_data.csv','ST_data.csv',
                print(root +"\\"+ i)
                #get_area = root.split("\\")[-3]
                file_name=root+'\\'+ whichcsv
                #print(file_name)
                previous_data, future_data = read_data(file_name, look_back, pred_len,whichcsv)#读具体文件，返回np数组

                #print(future_data)
                if flag == 0:
                    all_previous_data2 = previous_data
                    all_future_data2 = future_data
                    if len(all_previous_data2)!=0:#防止第一次拼到的仍然是空
                        flag = 1
                    continue
                if not len(previous_data):
                    continue
                if not len(future_data):
                    continue
                all_previous_data2 = np.concatenate([all_previous_data2, previous_data], axis=0)
                all_future_data2 = np.concatenate([all_future_data2, future_data],axis=0)
                del previous_data
                del future_data
                gc.collect()
                ####性能优化######
                if len(all_previous_data2)>400000:#############################################批次大小
                    previous_list.append(all_previous_data2)
                    future_list.append(all_future_data2)
                    all_previous_data2=''
                    all_future_data2=''
                    flag=0
                    gc.collect()
                ####性能优化######

    ####性能优化######
    #所有文件遍历完毕后，把最后一部分数据一起倾倒进List
    if len(all_previous_data2) > 0:
        previous_list.append(all_previous_data2)
        future_list.append(all_future_data2)
        all_previous_data2 = ''
        all_future_data2 = ''
        flag = 0
        gc.collect()
    if len(previous_list)==0 or len(future_list)==0:
        return -1
    else:
        all_previous_data2=previous_list[0]
        all_future_data2=future_list[0]
        previous_list[0]=0
        future_list[0]=0
        gc.collect()
    if len(previous_list)==1 or len(future_list)==1:
        return [all_previous_data2, all_future_data2]
    if len(previous_list)>1 or len(future_list)>1:
        for i,x in enumerate(previous_list):
            if i ==0:
                continue
            else:
                all_previous_data2=np.concatenate([all_previous_data2, x], axis=0)
                all_previous_data2[i]=0
                gc.collect()
                print(str(i)+"正在拼接x")
        for i,y in enumerate(future_list):
            if i ==0:
                continue
            else:
                all_future_data2=np.concatenate([all_future_data2, y], axis=0)
                all_future_data2[i]=0
                gc.collect()
                print(str(i) + "正在拼接y")

    ####性能优化######


    if len(all_previous_data2)==0 or len(all_future_data2)==0:
        return -1
    return [all_previous_data2, all_future_data2]

#RT [0,1]
#MT []
#data=read_from_dir_RT_data("C:\按地域划分数据集test\淳安县",[0,1],4,2,"RT_data.csv")
#data=read_from_dir_RT_data("C:\按地域划分数据集test\淳安县",[7,1,2,8,9,10,11],4,2,"MT_data.csv")
#data=read_from_dir_RT_data("C:\按地域划分数据集test\淳安县",[0,1,2,3,4,5,6],4,2,"ST_data.csv")
#存储目录



try:
    start =time.clock()
except:
    start =time.perf_counter()

#批量执行
#预测一个点

print("-------------------------1开始处理-- rt100-10 ----------------------------")
for p in res_all:
    print(disk+p[0]+"\\"+p[1])
    data = read_from_dir_RT_data(disk+p[0]+"\\"+p[1], 96, 20, "RT_data.csv")#数据时间、瞬时有功(kW)、standard_key、id
    if data==-1:
        continue
    if os.path.exists(linknew) == False:
        os.makedirs(linknew)
    with open(linknew+p[0]+"_"+p[1]+"_RT_data_96_20.pk", 'wb+') as f:
        pickle.dump(data, f)
        f.close()
    data=-1
    gc.collect()
print("-------------------------1结束处理-- rt100-10--------------------------")

print("-------------------------2开始处理-- rt48-4----------------------------")
# for p in res_all:
#     print(disk+p[0]+"\\"+p[1])
#     data = read_from_dir_RT_data(disk+p[0]+"\\"+p[1], [0,1,3,4,5,6], 48, 4, "RT_data.csv")
#     if data==-1:
#         continue
#
#     if os.path.exists(linknew) == False:
#         os.makedirs(linknew)
#     with open(linknew+p[0]+"_"+p[1]+"_RT_data_48_4.pk", 'wb+') as f:
#         pickle.dump(data, f)
#         f.close()
#     data=-1
#     gc.collect()
print("-------------------------2结束处理-- rt48-4 ----------------------------")

print("-------------------------3开始处理-- mt9-3 ----------------------------")
for p in res_all:
    print(disk+p[0]+"\\"+p[1])
    data = read_from_dir_RT_data(disk+p[0]+"\\"+p[1],6,2, "MT_data.csv")#数据时间2,平均负荷(kW),standard_key,id
    if data==-1:
        continue
    if os.path.exists(linknew) == False:
        os.makedirs(linknew)
    with open(linknew+p[0]+"_"+p[1]+"_MT_data_6_2.pk", 'wb+') as f:
        pickle.dump(data, f)
        f.close()
    data=-1
    gc.collect()
print("-------------------------3结束处理-- mt6-1 ----------------------------")
print("-------------------------4开始处理-- mt12-3 ----------------------------")


# for p in [["按聚类划分数据集(方案1-汇总标准化)_去异常","class15"]]:
# for p in res_all:
#     print(disk+p[0]+"\\"+p[1])
#     data = read_from_dir_RT_data(disk+p[0]+"\\"+p[1], [7,1,2,3,4,5,8,9,10,11,12],12,3, "MT_data.csv")
#     if data==-1:
#         continue
#     if os.path.exists(linknew) == False:
#         os.makedirs(linknew)
#     with open(linknew+p[0]+"_"+p[1]+"_MT_data_12_3.pk", 'wb+') as f:
#         pickle.dump(data, f)
#         f.close()
#     data=-1
#     gc.collect()
print("-------------------------4结束处理-- mt12-3 ----------------------------")

print("-------------------------5开始处理-- St100-10 ----------------------------")
for p in res_all:
    print(disk+p[0]+"\\"+p[1])
    data = read_from_dir_RT_data(disk+p[0]+"\\"+p[1],70,20, "ST_data.csv")#数据时间、平均负荷,standard_key,id
    if data==-1:
        continue
    if os.path.exists(linknew) == False:
        os.makedirs(linknew)
    with open(linknew+p[0]+"_"+p[1]+"_ST_data_70_20.pk", 'wb+') as f:
        pickle.dump(data, f)
        f.close()
    data=-1
    gc.collect()
print("-------------------------5结束处理-- St30-10 ----------------------------")
print("-------------------------6开始处理-- St30-7 ----------------------------")
# for p in res_all:
#     print(disk+p[0]+"\\"+p[1])
#     data = read_from_dir_RT_data(disk+p[0]+"\\"+p[1], [0, 1, 2, 3, 4, 5, 6, 7, 8,9,11],30,7, "ST_data.csv")
#     if data==-1:
#         continue
#     if os.path.exists(linknew) == False:
#         os.makedirs(linknew)
#     with open(linknew+p[0]+"_"+p[1]+"_ST_data_30_7.pk", 'wb+') as f:
#         pickle.dump(data, f)
#         f.close()
#     data=-1
#     gc.collect()
print("-------------------------6结束处理-- St30-7 ----------------------------")
#
#
#     # with open("d:/淳安县_RT_data.pk", 'rb') as f:
#     #     data=pickle.load(f)
#     #     print(data[0])
# #预测多个点
# try:
#     end =time.clock()
# except:
#     end =time.perf_counter()
# print('Running time: %s Minutes'%((end-start)/60))