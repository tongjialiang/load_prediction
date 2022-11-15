#!/usr/bin/python
# -*- coding: utf-8 -*-
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
import GetClusteringXandBusname_long
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn import mixture
from scipy.cluster.hierarchy import linkage
from sklearn import preprocessing
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.tree import export_graphviz
from IPython.display import Image
class save_csv:
    def __init__(self,num_data, test_per, how_long, predict_type, x_type, y_type, model, dir, file_name, way4divided,
                 data_class, way4standard, way4adjustParameter, ajustParameter, way4ajustParameter, way4cv,
                 way4adjustParameter_score, best_estimator, best_params, MSE_score, RMSE_score, MAE, R2_score, res_ajustParameter,
                 csv_file,use_time,opt,total_epochs):
        self.model = model  # csv 使用的模型
        self.way4divided = way4divided ## 划分数据集的方式
        self.data_class = data_class  ## 数据类型
        self.way4standard = way4standard  ## 标准化的方式
        self.num_data = num_data  # csv 样本数据量
        self.test_per = test_per #测试集占比


        self.predict_type = predict_type # 预测类型
        self.x_type = x_type  # csv 采样维度
        self.y_type = y_type  # csv 预测维度
        self.way4adjustParameter = way4adjustParameter  # 调参方式
        self.ajustParameter = ajustParameter # 待调整的参数
        self.way4ajustParameter =way4ajustParameter  # 调参算法
        self.way4cv = way4cv  # 交叉验证的折数
        self.way4adjustParameter_score = way4adjustParameter_score  # 调参评价指标
        self.best_params = best_params # 最佳参数
        self.MSE_score = MSE_score
        self.RMSE_score = RMSE_score
        self.MAE = MAE
        self.R2_score = R2_score

        self.dir = dir
        self.file_name = file_name  # csv 样本名称 按地域划分数据集_方案1-汇总标准化_海宁市_MT_data_4_1.pk
        self.best_estimator = best_estimator  # 最佳模型-保存
        self.res_ajustParameter = res_ajustParameter  # 调参结果
        self.how_long = how_long
        self.csv_file=csv_file
        self.use_time=use_time
        self.opt=opt
        self.total_epochs=total_epochs


    def do_savecsv(self):
        #存放分析结果
        df_savecsv = pd.DataFrame(columns = ['使用的模型','划分数据集的方式','数据类型','标准化的方式'
            ,'样本数据量','测试集占比','类别','采样维度','预测维度','调整的参数','调参算法','交叉验证的折数','调参评价指标'
            ,'最佳参数','MSE_score','RMSE_score','MAE','R2_score','文件名','执行时间_分钟','优化器','total_epochs'])



        df_savecsv = df_savecsv.append([{'使用的模型':self.model,'划分数据集的方式':self.way4divided,'数据类型':self.data_class,'标准化的方式':self.way4standard
            ,'样本数据量':self.num_data,'测试集占比':self.test_per,'类别':self.predict_type,'采样维度':self.x_type,'预测维度':self.y_type,
                                         '调整的参数':self.ajustParameter,'调参算法':self.way4ajustParameter,
                                         '交叉验证的折数':self.way4cv,'调参评价指标':self.way4adjustParameter_score
            ,'最佳参数':self.best_params,'MSE_score':self.MSE_score,'RMSE_score':self.RMSE_score,'MAE':self.MAE,'R2_score':self.R2_score
                                         ,'文件名':self.file_name,'执行时间_分钟':self.use_time,'优化器':self.opt,'total_epochs':self.total_epochs}])
        model_res = [self.best_estimator, self.res_ajustParameter]  # 最优模型，调参过程
        #model_res = [self.res_ajustParameter,self.state]  # 最优模型，调参过程
        return df_savecsv,model_res

    def do_drow_decision_tree(self,dir):#RandomForestRegressor
        print(dir)
        if os.path.exists("D:\\实验记录\\实验结果\\DecisionTreeRegressor\\") == False:
            os.makedirs("D:\\实验记录\\实验结果\\DecisionTreeRegressor\\")
        export_graphviz(
            self.best_estimator,

            out_file="D:\\实验记录\\实验结果\\DecisionTreeRegressor\\"+dir+".dot",
            #feature_names=iris.feature_names[2:],
            #class_names=iris.target_names,
            rounded=True,
            filled=True
        )


        #Image(filename='d:\\draw\\a.png', width=400, height=400)

