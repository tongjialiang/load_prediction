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
import use_CNN_v2
import use_LSTM
import use_Transformer
from torch.utils import data
warnings.filterwarnings("ignore")
import use_control_layer_DL
import time
disk="C"

def do_CNN():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成\\"
    outdir=disk+":\\实验记录\\实验结果\\CNN"
    model_name_this='CNN'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_CNN_v2():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成\\"
    outdir=disk+":\\实验记录\\实验结果\\CNN_v2"
    model_name_this='CNN_v2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_LSTM():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成\\"
    outdir=disk+":\\实验记录\\实验结果\\LSTM"
    model_name_this='LSTM'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_Transformer():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer"
    model_name_this='Transformer'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_Transformer_v2():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v2"
    model_name_this='Transformer_v2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_Transformer_v3():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v3"
    model_name_this='Transformer_v3'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_v4():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v4"
    model_name_this='Transformer_v4'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_Transformer_v5():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v5"
    model_name_this='Transformer_v5'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_Transformer_v6():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v6"
    model_name_this='Transformer_v6'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_Transformer_v7():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v7"
    model_name_this='Transformer_v7'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_v8():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v8"
    model_name_this='Transformer_v8'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_v9():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v9"
    model_name_this='Transformer_v9'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_Transformer_v9_2():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v9_2"
    model_name_this='Transformer_v9_2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_v9_3():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v9_3"
    model_name_this='Transformer_v9_3'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_v9_4():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v9_4"
    model_name_this='Transformer_v9_4'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_v9_5():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v9_5"
    model_name_this='Transformer_v9_5'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_v9_6():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v9_6"
    model_name_this='Transformer_v9_6'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_v9_7():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v9_7"
    model_name_this='Transformer_v9_7'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_v9_8():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v9_8"
    model_name_this='Transformer_v9_8'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_v9_9():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v9_9"
    model_name_this='Transformer_v9_9'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_v9_10():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_v9_10"
    model_name_this='Transformer_v9_10'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_lstm():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\LSTM"
    model_name_this='LSTM'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_cnn_lstm():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\CNN_LSTM"
    model_name_this='CNN_LSTM'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_GA_cnn_lstm():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\GA_CNN_LSTM"
    model_name_this='GA_CNN_LSTM'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
try:
    start =time.clock()
except:
    start =time.perf_counter()


# do_LSTM()
# do_GA_cnn_lstm()


do_Transformer_v9()
# do_Transformer_v9_2()
# do_Transformer_v9_3()
# do_Transformer_v9_4()
# do_Transformer_v9_5()
# do_Transformer_v9_6()
# do_Transformer_v9_7()
# do_Transformer_v9_8()
# do_Transformer_v9_9()
# do_Transformer_v9_10()

# do_lstm()
# do_cnn_lstm()

try:
    end =time.clock()
except:
    end =time.perf_counter()
print('Running time: %s Minutes'%((end-start)/60))
