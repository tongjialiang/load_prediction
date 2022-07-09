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
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from skopt import BayesSearchCV#pip install scikit-optimize
from sklearn.linear_model import Ridge
import use_Ridge
import use_save_csv
import use_LightGBM
import use_Xgboost
import use_LightGBM_v2
warnings.filterwarnings("ignore")
import use_control_layer
import time
disk="D"

def do_Ridge():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Ridge"
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name="Ridge",way4adjustParameter="随机搜索",do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
# def do_Ridge_v3():
#     dir="D:\\数据采样完成new\\"
#     outdir="D:\\实验记录\\实验结果\\Ridge_v3"
#     if os.path.exists(outdir) == False:
#         os.makedirs(outdir)
#     control_layer=use_control_layer.control_layer(dir_input=dir,model_name="Ridge",way4adjustParameter="随机搜索",do_save=True,csv_file=outdir)
#     control_layer.do_load_prediction()
def do_DecisionTreeRegressor():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\DecisionTreeRegressor"
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)

    control_layer=use_control_layer.control_layer(dir_input=dir,model_name="DecisionTreeRegressor",way4adjustParameter="随机搜索",do_save=True,csv_file=outdir)
#适应性分步调参-决策树
    control_layer.do_load_prediction()

def do_RandomForestRegressor():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\RandomForestRegressor"
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)

    control_layer=use_control_layer.control_layer(dir_input=dir,model_name="RandomForestRegressor",way4adjustParameter="随机搜索",do_save=True,csv_file=outdir)
#适应性分步调参-决策树
    control_layer.do_load_prediction()



def do_SVR():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\SVR"
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name="SVR",way4adjustParameter="网格搜索",do_save=True,csv_file=outdir)
#适应性分步调参-决策树
    control_layer.do_load_prediction()

def do_KNN():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\KNN"
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name="KNN",way4adjustParameter="随机搜索",do_save=True,csv_file=outdir)
#适应性分步调参-决策树
    control_layer.do_load_prediction()

def do_LightGBM():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\LightGBM"
    model_name2='LightGBM'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
#适应性分步调参-决策树
    control_layer.do_load_prediction()

def do_LightGBM_v2():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\LightGBM_v2"
    model_name2='LightGBM_v2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
#适应性分步调参-决策树
    control_layer.do_load_prediction()
def do_LightGBM_v3():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\LightGBM_比较实验"
    model_name2='LightGBM_v3'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
#适应性分步调参-决策树
    control_layer.do_load_prediction()
def do_LightGBM_v4():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\LightGBM_v4"
    model_name2='LightGBM_v4'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
#适应性分步调参-决策树
    control_layer.do_load_prediction()
def do_Xgboost_v2():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Xgboost_v2"
    model_name2='Xgboost_v2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
#适应性分步调参-决策树
    control_layer.do_load_prediction()
def do_Xgboost_v3():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Xgboost_比较实验"
    model_name2='Xgboost_v3'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
#适应性分步调参-决策树
    control_layer.do_load_prediction()
def do_Xgboost_v4():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Xgboost_v4"
    model_name2='Xgboost_v4'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
#适应性分步调参-决策树
    control_layer.do_load_prediction()
def do_RandomForestRegressor_v2():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\RandomForestRegressor_v2"
    model_name2='RandomForestRegressor_v2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_RandomForestRegressor_v3():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\RandomForestRegressor_比较实验"
    model_name2='RandomForestRegressor_v3'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_RandomForestRegressor_v3_onepredict():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\RandomForestRegressor_一次性预测"
    model_name2='RandomForestRegressor_v3_onepredict'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_RandomForestRegressor_v3_news():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\RandomForestRegressor_新息纳入"
    model_name2='RandomForestRegressor_v3_news'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_RandomForestRegressor_v4():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\RandomForestRegressor_v4"
    model_name2='RandomForestRegressor_v4'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Ridge_v2():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Ridge_v2"
    model_name2='Ridge_v2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_KNN_v2():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\KNN_v2"
    model_name2='KNN_v2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_KNN_v3():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\KNN_v3"
    model_name2='KNN_v3'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_SVR_v2():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\SVR_v2"
    model_name2='SVR_v2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_DecisionTreeRegressor_v2():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\DecisionTreeRegressor_v2"
    model_name2='DecisionTreeRegressor_v2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_DecisionTreeRegressor_v3():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\DecisionTreeRegressor_比较实验"
    model_name2='DecisionTreeRegressor_v3'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_DecisionTreeRegressor_v4():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\DecisionTreeRegressor_v4"
    model_name2='DecisionTreeRegressor_v4'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Ridge_v3():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Ridge_比较实验"
    model_name2='Ridge_v3'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Ridge_v4():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Ridge_v4"
    model_name2='Ridge_v4'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_SVR_v3():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\SVR_比较实验"
    model_name2='SVR_v3'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_SVR_v3_oncepredict():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\SVR_一次性预测"
    model_name2='SVR_v3_oncepredict'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_SVR_v3_news():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\SVR_新息纳入"
    model_name2='SVR_v3_news'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_SVR_v4():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\SVR_v4"
    model_name2='SVR_v4'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer.control_layer(dir_input=dir,model_name=model_name2,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
try:
    start =time.clock()
except:
    start =time.perf_counter()

#do_DecisionTreeRegressor()
# do_RandomForestRegressor()def do_RandomForestRegressor():
#     dir="D:\\数据采样完成\\"
#     outdir="D:\\实验记录\\实验结果\\RandomForestRegressor"
#     if os.path.exists(outdir) == False:
#         os.makedirs(outdir)
#
#     control_layer=use_control_layer.control_layer(dir_input=dir,model_name="RandomForestRegressor",way4adjustParameter="随机搜索",do_save=True,csv_file=outdir)
# #适应性分步调参-决策树
#     control_layer.do_load_prediction()
#do_RandomForestRegressor()
#do_SVR()
#do_KNN()
#do_Ridge()
#do_Ridge_v2()
#do_KNN_v2()
# do_SVR_v2()
# do_DecisionTreeRegressor_v2()


do_Xgboost_v3()
do_LightGBM_v3()
do_Ridge_v3()
do_DecisionTreeRegressor_v3()
do_RandomForestRegressor_v3()
do_SVR_v3()



#新息纳入的消融实验

# do_RandomForestRegressor_v3()
# do_RandomForestRegressor_v3_onepredict()
# do_RandomForestRegressor_v3_news()

try:
    end =time.clock()
except:
    end =time.perf_counter()
print('Running time: %s Minutes'%((end-start)/60))
