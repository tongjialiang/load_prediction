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

dir="d:\\实验记录\\数据分析\\"

df_res = pd.DataFrame(columns = ['模型名称','划分数据集的方式'
    ,'标准化的方式','期望MSE','期望RMSE','期望MAE'
   ])

for root, dirs, filelist in os.walk(dir):
        for i in filelist:
            #filename=root+"\\"+i
            #print(filename)
            if i.endswith("csv") and ~(i.endswith('数据分析_机器学习模型.csv')):
                #print(i)#RT_data.csv
                #print(root)#D:\按地域划分数据集\淳安县\淳安县供电分公司1\99
                print(i)#DecisionTreeRegressor.csv
                model_name=i

                try:
                    my_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    my_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                ##############################################################地域划分，方案1
                way_Partition_DS ="按地域划分数据集"
                way_normal='方案1-汇总标准化'
                this_data=my_data[(my_data['划分数据集的方式']==way_Partition_DS)&(my_data['标准化的方式']==way_normal) ]
                num_data=sum(this_data["样本数据量"])
                print(num_data)#总数据量
                p=this_data["样本数据量"]/num_data
                print(p)#取值的概率
                MSE_expectation=sum(this_data["MSE_score"]*p)
                RMSE_expectation = sum(this_data["RMSE_score"] * p)
                MAE_expectation = sum(this_data["MAE"] * p)
                R2_score_expectation = sum(this_data["R2_score"] * p)

                print(MSE_expectation)
                df_res = df_res.append([{'模型名称': model_name, '划分数据集的方式': way_Partition_DS, '标准化的方式': way_normal,
                                          '期望MSE': MSE_expectation, '期望RMSE':RMSE_expectation,
                                         '期望MAE': MAE_expectation,
                                         '期望R2_score': R2_score_expectation}])
                ##############################################################地域划分，方案2
                way_Partition_DS ="按地域划分数据集"
                way_normal='方案2-对每个企业标准化'
                this_data=my_data[(my_data['划分数据集的方式']==way_Partition_DS)&(my_data['标准化的方式']==way_normal)]
                num_data=sum(this_data["样本数据量"])
                print(num_data)#总数据量
                p=this_data["样本数据量"]/num_data
                print(p)#取值的概率
                MSE_expectation=sum(this_data["MSE_score"]*p)
                RMSE_expectation = sum(this_data["RMSE_score"] * p)
                MAE_expectation = sum(this_data["MAE"] * p)
                R2_score_expectation = sum(this_data["R2_score"] * p)

                print(MSE_expectation)
                df_res = df_res.append([{'模型名称': model_name, '划分数据集的方式': way_Partition_DS, '标准化的方式': way_normal,
                                          '期望MSE': MSE_expectation, '期望RMSE':RMSE_expectation,
                                         '期望MAE': MAE_expectation,
                                         '期望R2_score': R2_score_expectation}])
                ##############################################################行业划分，方案1
                way_Partition_DS = "按行业划分数据集"
                way_normal = '方案1-汇总标准化'
                this_data = my_data[(my_data['划分数据集的方式'] == way_Partition_DS) & (my_data['标准化的方式'] == way_normal)]
                num_data = sum(this_data["样本数据量"])
                print(num_data)  # 总数据量
                p = this_data["样本数据量"] / num_data
                print(p)  # 取值的概率
                MSE_expectation = sum(this_data["MSE_score"] * p)
                RMSE_expectation = sum(this_data["RMSE_score"] * p)
                MAE_expectation = sum(this_data["MAE"] * p)
                R2_score_expectation = sum(this_data["R2_score"] * p)

                print(MSE_expectation)
                df_res = df_res.append([{'模型名称': model_name, '划分数据集的方式': way_Partition_DS, '标准化的方式': way_normal,
                                         '期望MSE': MSE_expectation, '期望RMSE': RMSE_expectation,
                                         '期望MAE': MAE_expectation,
                                         '期望R2_score': R2_score_expectation}])
                ##############################################################行业划分，方案2
                way_Partition_DS = "按行业划分数据集"
                way_normal = '方案2-对每个企业标准化'
                this_data = my_data[(my_data['划分数据集的方式'] == way_Partition_DS) & (my_data['标准化的方式'] == way_normal)]
                num_data = sum(this_data["样本数据量"])
                print(num_data)  # 总数据量
                p = this_data["样本数据量"] / num_data
                print(p)  # 取值的概率
                MSE_expectation = sum(this_data["MSE_score"] * p)
                RMSE_expectation = sum(this_data["RMSE_score"] * p)
                MAE_expectation = sum(this_data["MAE"] * p)
                R2_score_expectation = sum(this_data["R2_score"] * p)

                print(MSE_expectation)
                df_res = df_res.append([{'模型名称': model_name, '划分数据集的方式': way_Partition_DS, '标准化的方式': way_normal,
                                         '期望MSE': MSE_expectation, '期望RMSE': RMSE_expectation,
                                         '期望MAE': MAE_expectation,
                                         '期望R2_score': R2_score_expectation}])
                ##############################################################行业划分，方案1
                way_Partition_DS = "按聚类划分数据集"
                way_normal = '方案1-汇总标准化'
                this_data = my_data[(my_data['划分数据集的方式'] == way_Partition_DS) & (my_data['标准化的方式'] == way_normal)]
                num_data = sum(this_data["样本数据量"])
                print(num_data)  # 总数据量
                p = this_data["样本数据量"] / num_data
                print(p)  # 取值的概率
                MSE_expectation = sum(this_data["MSE_score"] * p)
                RMSE_expectation = sum(this_data["RMSE_score"] * p)
                MAE_expectation = sum(this_data["MAE"] * p)
                R2_score_expectation = sum(this_data["R2_score"] * p)

                print(MSE_expectation)
                df_res = df_res.append([{'模型名称': model_name, '划分数据集的方式': way_Partition_DS, '标准化的方式': way_normal,
                                         '期望MSE': MSE_expectation, '期望RMSE': RMSE_expectation,
                                         '期望MAE': MAE_expectation,
                                         '期望R2_score': R2_score_expectation}])
        df_res.to_csv(path_or_buf=dir+'数据分析_机器学习模型.csv', encoding="utf_8_sig", index=False)