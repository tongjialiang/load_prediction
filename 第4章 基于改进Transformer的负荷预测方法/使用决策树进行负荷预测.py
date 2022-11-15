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
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square
from skopt import BayesSearchCV  # pip install scikit-optimize
from sklearn.linear_model import Ridge
import use_save_csv
from sklearn.tree import DecisionTreeRegressor
import gc
import auto_search

warnings.filterwarnings("ignore")

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# 设置参数搜索范围
ajustParameter = "max_depth：5-100,min_samples_split:8-20,min_samples_leaf:5-20"
# 设置模型
model = DecisionTreeRegressor(criterion='mse')
# 设置model-name，记录日志用
model_name = 'DecisionTreeRegressor'
# 执行小批量数据集 百分比
use_data = False
use_data_ratio = 1
min_use_data = 2000
# 调参使用的数据量
use_adjust_parameter = True
use_adjust_parameter_ratio = 0.01
min_use_adjust_parameter = 10000
#松弛变量
slack_variable=8

# dir='D:\\数据采样完成\\按行业划分数据集_方案1-汇总标准化_住宿和餐饮业_MT_data_4_1.pk'
class model_DecisionTreeRegressor:
    def __init__(self, dir, how_long, file_name, way4adjustParameter, csv_file):
        self.num_data = ''  # csv 样本数据量
        self.test_per = '5%'  # 测试集占比
        self.how_long = how_long
        self.predict_type = ''  # 预测类型
        self.x_type = ''  # csv 采样维度
        self.y_type = ''  # csv 预测维度
        self.model = model_name  # csv 使用的模型
        self.dir = dir
        self.file_name = file_name  # csv 样本名称  按地域划分数据集_方案1-汇总标准化_海宁市_MT_data_4_1.pk
        # print(self.file_name)  # 按地域划分数据集_方案1-汇总标准化_海宁市_MT_data_4_1.pk
        self.way4divided = file_name.split("_")[0]  # 划分数据集的方式
        self.data_class = file_name.split("_")[-5]  # 数据分类
        self.way4standard = file_name.split("_")[1]  # 标准化的方式
        self.way4adjustParameter = way4adjustParameter  # 调参方式
        self.ajustParameter = ''  # 调整的参数
        self.way4ajustParameter = self.way4adjustParameter  # 调参算法
        self.way4cv = ''  # 交叉验证的折数
        self.way4adjustParameter_score = ''  # 调参评价指标
        self.best_estimator = ''  # 最佳模型-保存
        self.best_params = ''  # 最佳参数
        self.MSE_score = 0
        self.RMSE_score = 0
        self.MAE = 0
        self.R2_score = 0
        self.res_ajustParameter = ''  # 调参结果
        self.csv_file = csv_file

    def do_model(self):

        print(self.how_long)
        with open(self.dir, 'rb') as f:
            data = pickle.load(f)
        #

        num_data = len(data[0])
        self.num_data = num_data

        # 随机打散
        print(data[0].shape)  # (2778, 48, 6)
        print(data[1].shape)  # (2778, 6)
        index = np.arange(num_data)
        np.random.shuffle(index)
        data[0] = data[0][index, :, :]  # X_train是训练集，y_train是训练标签
        data[1] = data[1][index]
        gc.collect()
        # 95%是训练数据
        if self.how_long == 'RT':
            X_train = data[0][:int(num_data * 0.95), :, 1]
            X_train = X_train.reshape(len(X_train), -1)
            Y_train = data[1][:int(num_data * 0.95), 1]
            X_test = data[0][int(num_data * 0.95):, :, 1]
            X_test = X_test.reshape(len(X_test), -1)
            Y_test = data[1][int(num_data * 0.95):, 1]
            self.predict_type = '实时预测'
            self.x_type = '输入:过去48*15分钟数据'
            self.y_type = '输出：未来15-150分钟数据'
        if self.how_long == 'MT' and self.way4standard == '方案1-汇总标准化':
            X_train = data[0][:int(num_data * 0.95), :, [1, 2, 6, 7]]  # 2供电容量
            X_train = X_train.reshape(len(X_train), -1)
            Y_train = data[1][:int(num_data * 0.95), 1]
            X_test = data[0][int(num_data * 0.95):, :, [1, 2, 6, 7]]
            X_test = X_test.reshape(len(X_test), -1)
            Y_test = data[1][int(num_data * 0.95):, 1]
            self.predict_type = '长期预测'
            self.x_type = '输入:过去4个月数据'
            self.y_type = '输出：未来1-3个月数据'

        if self.how_long == 'ST' and self.way4standard == '方案1-汇总标准化':
            X_train = data[0][:int(num_data * 0.95), :, [1, 2, 3, 6]]  # 6供电容量
            X_train = X_train.reshape(len(X_train), -1)
            Y_train = data[1][:int(num_data * 0.95), 1]
            X_test = data[0][int(num_data * 0.95):, :, [1, 2, 3, 6]]
            X_test = X_test.reshape(len(X_test), -1)
            Y_test = data[1][int(num_data * 0.95):, 1]
            self.predict_type = '短期预测'
            self.x_type = '输入:过去30天数据'
            self.y_type = '输出：未来1-4天数据'

        if self.how_long == 'MT' and self.way4standard == '方案2-对每个企业标准化':
            X_train = data[0][:int(num_data * 0.95), :, [1, 6]]  # 2供电容量 7最小负荷 舍弃
            X_train = X_train.reshape(len(X_train), -1)
            Y_train = data[1][:int(num_data * 0.95), 1]
            X_test = data[0][int(num_data * 0.95):, :, [1, 6]]
            X_test = X_test.reshape(len(X_test), -1)
            Y_test = data[1][int(num_data * 0.95):, 1]
            self.predict_type = '长期预测'
            self.x_type = '输入:过去4个月数据'
            self.y_type = '输出：未来1-3个月数据'

        if self.how_long == 'ST' and self.way4standard == '方案2-对每个企业标准化':
            X_train = data[0][:int(num_data * 0.95), :, [1, 2]]  # 6供电容量 3最小负荷
            X_train = X_train.reshape(len(X_train), -1)
            Y_train = data[1][:int(num_data * 0.95), 1]
            X_test = data[0][int(num_data * 0.95):, :, [1, 2]]
            X_test = X_test.reshape(len(X_test), -1)
            Y_test = data[1][int(num_data * 0.95):, 1]
            self.predict_type = '短期预测'
            self.x_type = '输入:过去30天数据'
            self.y_type = '输出：未来1-4天数据'
        gc.collect()
        # 执行小批量数据集训练+预测
        len_x = len(X_train)
        if use_data and len_x > min_use_data:
            use_data_num = int(len_x * use_data_ratio)
            use_data_num = max(use_data_num, min_use_data)
            X_train = X_train[:use_data_num, :]
            Y_train = Y_train[:use_data_num]
            # X_test = X_test[:use_data_num, :]
            # Y_test = Y_test[:use_data_num]
        gc.collect()

        # 搜索最佳参数
        param_dist = {
            'criterion' : ['mse'],
            'max_depth': range(5, 100),  # 10-100
            'min_samples_split': range(8, 20, 1),  # 如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分。 10
            'min_samples_leaf': range(5, 20, 1)  # 如果某叶子节点必须具有的最小样本数。 5
        }
        self.ajustParameter = ajustParameter

        # 使用小批量数据进行调参
        def small_batch_adjust_parameter(X_train, Y_train, use_adjust_parameter, min_use_adjust_parameter,
                                         use_adjust_parameter_ratio):
            num_sample = len(X_train)
            if use_adjust_parameter and num_sample > min_use_adjust_parameter:
                num_use_sample = int(num_sample * use_adjust_parameter_ratio)
                num_use_sample = max(num_use_sample, min_use_adjust_parameter)
                grid.fit(X_train[:num_use_sample, :],
                         Y_train[:num_use_sample])
            else:
                grid.fit(X_train, Y_train)

        if self.way4adjustParameter == "网格搜索":
            grid = GridSearchCV(model, param_dist, cv=None, scoring='neg_mean_squared_error')
            self.way4cv = 4
            small_batch_adjust_parameter(X_train, Y_train, use_adjust_parameter, min_use_adjust_parameter,
                                         use_adjust_parameter_ratio)
            self.way4adjustParameter_score = 'MSE'
            self.best_estimator = grid.best_estimator_
            self.best_params = str(grid.best_params_)
            self.res_ajustParameter = grid.cv_results_
        if self.way4adjustParameter == "随机搜索":
            grid = RandomizedSearchCV(model, param_dist, cv=None, scoring='neg_mean_squared_error', n_iter=100)
            self.way4cv = 4
            small_batch_adjust_parameter(X_train, Y_train, use_adjust_parameter, min_use_adjust_parameter,
                                         use_adjust_parameter_ratio)
            self.way4adjustParameter_score = 'MSE'
            self.best_estimator = grid.best_estimator_
            self.best_params = str(grid.best_params_)
            self.res_ajustParameter = grid.cv_results_
        if self.way4adjustParameter == "贝叶斯搜索":
            grid = BayesSearchCV(model, [param_dist], cv=None, scoring='neg_mean_squared_error', n_iter=50)
            self.way4cv = 4
            small_batch_adjust_parameter(X_train, Y_train, use_adjust_parameter, min_use_adjust_parameter,
                                         use_adjust_parameter_ratio)
            self.way4adjustParameter_score = 'MSE'
            self.best_estimator = grid.best_estimator_
            self.best_params = str(grid.best_params_)
            self.res_ajustParameter = grid.cv_results_
        if self.way4adjustParameter == "改进的顺序搜素-决策树":

            print("适应性分步调参-决策树-start")
            #调第一个参数max_depth，另外两个参数min_samples_split、min_samples_leaf取经验值
            param_dist_temp=param_dist.copy() #不拷贝会在原值改
            #param_dist_temp['max_depth'] = range(5, 100, 1)
            param_dist_temp['min_samples_split'] = [param_dist['min_samples_split'][0]]
            param_dist_temp['min_samples_leaf'] =[ param_dist['min_samples_leaf'][0]]


            #适应性网格搜索
            grid = GridSearchCV(model, param_dist_temp, cv=None, scoring='neg_mean_squared_error')
            small_batch_adjust_parameter(X_train, Y_train, use_adjust_parameter, min_use_adjust_parameter,
                                         use_adjust_parameter_ratio)
            print(grid.cv_results_)
            print(grid.best_params_)

            #求阈值
            diff_loss =  (max(-grid.cv_results_['mean_test_score']) - min(-grid.cv_results_['mean_test_score']))   # 这组参数中损失的最大值与最小值的差
            print("diff_loss"+str(diff_loss))
            # 求阈值
            num_param = len(param_dist_temp['max_depth'])
            print("num_param    "+str(num_param))
            threshold_value = (diff_loss/num_param) *slack_variable# 阈值
            print("threshold_value  "+str(threshold_value))
            bestindex=np.where(grid.cv_results_['mean_test_score']==grid.best_score_)[0][0]
            print(bestindex)
            while bestindex>0:
                if -grid.cv_results_['mean_test_score'][bestindex-1]-(-grid.best_score_)<-1:
                    bestindex=bestindex-1
                else:
                    break
            print(bestindex)
            #确定max_depth的值
            bestvalue=(grid.cv_results_['params'][bestindex]['max_depth'])
            #print(param_dist)
            param_dist['max_depth']=[bestvalue]


            ####################################
            # 调参min_samples_split,min_samples_leaf取经验值
            param_dist_temp = param_dist.copy()
            # param_dist_temp['max_depth'] = range(5, 100, 1)
            #param_dist_temp['min_samples_split'] = [param_dist['min_samples_split'][0]]
            param_dist_temp['min_samples_leaf'] = [param_dist['min_samples_leaf'][0]]
            # 适应性网格搜索
            grid = GridSearchCV(model, param_dist_temp, cv=None, scoring='neg_mean_squared_error')
            small_batch_adjust_parameter(X_train, Y_train, use_adjust_parameter, min_use_adjust_parameter,
                                         use_adjust_parameter_ratio)
            print(grid.cv_results_)
            print(grid.best_params_)

            # 求阈值
            diff_loss = (max(-grid.cv_results_['mean_test_score']) - min(
                -grid.cv_results_['mean_test_score']))  # 这组参数中损失的最大值与最小值的差
            print("diff_loss" + str(diff_loss))
            # 求阈值
            num_param = len(param_dist_temp['min_samples_split'])
            print("num_param    " + str(num_param))
            threshold_value = (diff_loss / num_param) * slack_variable

            # 阈值
            print("threshold_value  " + str(threshold_value))
            bestindex = np.where(grid.cv_results_['mean_test_score'] == grid.best_score_)[0][0]
            print(bestindex)
            while bestindex <len(param_dist_temp['min_samples_split'])-1 :
                if -grid.cv_results_['mean_test_score'][bestindex + 1] - (-grid.best_score_) < threshold_value:
                    bestindex = bestindex + 1
                else:
                    break
            print(bestindex)
            # 确定max_depth的值
            bestvalue = (grid.cv_results_['params'][bestindex]['min_samples_split'])
            param_dist['min_samples_split'] = [bestvalue]
            print(param_dist)
            ####################################
            # 调参min_samples_leaf
            param_dist_temp = param_dist.copy()
            # param_dist_temp['max_depth'] = range(5, 100, 1)
            # param_dist_temp['min_samples_split'] = [param_dist['min_samples_split'][0]]
            #param_dist_temp['min_samples_leaf'] = [param_dist['min_samples_leaf'][0]]
            # 适应性网格搜索
            grid = GridSearchCV(model, param_dist_temp, cv=None, scoring='neg_mean_squared_error')
            small_batch_adjust_parameter(X_train, Y_train, use_adjust_parameter, min_use_adjust_parameter,
                                         use_adjust_parameter_ratio)
            print(grid.cv_results_)
            print(grid.best_params_)

            # 求阈值
            diff_loss = (max(-grid.cv_results_['mean_test_score']) - min(
                -grid.cv_results_['mean_test_score']))  # 这组参数中损失的最大值与最小值的差
            print("diff_loss" + str(diff_loss))
            # 求阈值
            num_param = len(param_dist_temp['min_samples_leaf'])
            print("num_param    " + str(num_param))
            threshold_value = (diff_loss / num_param) * slack_variable  # 阈值
            print("threshold_value  " + str(threshold_value))
            bestindex = np.where(grid.cv_results_['mean_test_score'] == grid.best_score_)[0][0]
            print(bestindex)
            while bestindex < len(param_dist_temp['min_samples_leaf']) - 1:
                if -grid.cv_results_['mean_test_score'][bestindex + 1] - (-grid.best_score_) < threshold_value:
                    bestindex = bestindex + 1
                else:
                    break
            print(bestindex)
            # 确定max_depth的值
            bestvalue = (grid.cv_results_['params'][bestindex]['min_samples_leaf'])
            param_dist['min_samples_leaf'] = [bestvalue]
            print(param_dist)


            grid = GridSearchCV(model, param_dist, cv=None, scoring='neg_mean_squared_error')
            self.way4cv = 4
            small_batch_adjust_parameter(X_train, Y_train, use_adjust_parameter, min_use_adjust_parameter,
                                         use_adjust_parameter_ratio)
            self.way4adjustParameter_score = 'MSE'
            self.best_estimator = grid.best_estimator_
            self.best_params = str(grid.best_params_)
            self.res_ajustParameter = grid.cv_results_

        # 使用最佳模型对全量数据进行预测
        if use_adjust_parameter:
            self.best_estimator.fit(X_train, Y_train)
        Y_predict = self.best_estimator.predict(X_test)
        print("MSE")
        self.MSE_score = mean_squared_error(Y_test, Y_predict)
        print(self.MSE_score)
        self.RMSE_score = mean_squared_error(Y_test, Y_predict) ** 0.5
        print("RMSE")
        print(self.RMSE_score)
        print("MAE")
        self.MAE = mean_absolute_error(Y_test, Y_predict)
        print(self.MAE)
        print("R2_score")  # 接近1最好
        self.R2_score = r2_score(Y_test, Y_predict)
        print(self.R2_score)

    def do_savedate(self):
        save_csv = use_save_csv.save_csv(num_data=self.num_data,
                                         test_per=self.test_per,
                                         how_long=self.how_long,
                                         predict_type=self.predict_type,
                                         x_type=self.x_type,
                                         y_type=self.y_type,
                                         model=self.model,
                                         dir=self.dir,
                                         file_name=self.file_name,
                                         way4divided=self.way4divided,
                                         data_class=self.data_class,
                                         way4standard=self.way4standard,
                                         way4adjustParameter=self.way4adjustParameter,
                                         ajustParameter=self.ajustParameter,
                                         way4ajustParameter=self.way4ajustParameter,
                                         way4cv=self.way4cv,
                                         way4adjustParameter_score=self.way4adjustParameter_score,
                                         best_estimator=self.best_estimator,
                                         best_params=self.best_params,
                                         MSE_score=self.MSE_score,
                                         RMSE_score=self.RMSE_score,
                                         MAE=self.MAE,
                                         R2_score=self.R2_score,
                                         res_ajustParameter=self.res_ajustParameter,
                                         csv_file=self.csv_file)
        df_savecsv = save_csv.do_savecsv()
        if model_name == 'DecisionTreeRegressor':
            save_csv.do_drow_decision_tree(self.file_name)  # 画决策树展示图
        return df_savecsv
