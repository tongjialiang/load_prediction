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

warnings.filterwarnings("ignore")
import use_control_layer_DL
import time
disk="D"

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


def do_Transformer():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer"
    model_name_this='Transformer'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_Transformer_DA():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_DA"
    model_name_this='Transformer_DA'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_Transformer_DA1():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_DA1"
    model_name_this='Transformer_DA1'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()

def do_Transformer_DA2():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_DA2"
    model_name_this='Transformer_DA2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_DA3():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_DA3"
    model_name_this='Transformer_DA3'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_DA4():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_DA4"
    model_name_this='Transformer_DA4'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_DA5():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_DA5"
    model_name_this='Transformer_DA5'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_DA6():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_DA6"
    model_name_this='Transformer_DA6'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_DA7():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_DA7"
    model_name_this='Transformer_DA7'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_DA8():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_DA8"
    model_name_this='Transformer_DA8'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_Transformer_DA9():##########################control_layer的模型名字要改
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\Transformer_DA9"
    model_name_this='Transformer_DA9'
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
def do_lstm1():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\LSTM1"
    model_name_this='LSTM1'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_lstm2():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\LSTM2"
    model_name_this='LSTM2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_lstm3():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\LSTM3"
    model_name_this='LSTM3'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_lstm4():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\LSTM4"
    model_name_this='LSTM4'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_lstm5():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\LSTM5"
    model_name_this='LSTM5'
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
def do_cnn_lstm1():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\CNN_LSTM1"
    model_name_this='CNN_LSTM1'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_cnn_lstm2():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\CNN_LSTM2"
    model_name_this='CNN_LSTM2'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_cnn_lstm3():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\CNN_LSTM3"
    model_name_this='CNN_LSTM3'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_cnn_lstm4():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\CNN_LSTM4"
    model_name_this='CNN_LSTM4'
    way4adjustParameter2='随机搜索'
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    control_layer=use_control_layer_DL.control_layer(dir_input=dir,model_name=model_name_this,way4adjustParameter=way4adjustParameter2,do_save=True,csv_file=outdir)
    control_layer.do_load_prediction()
def do_cnn_lstm5():
    dir=disk+":\\数据采样完成new\\"
    outdir=disk+":\\实验记录\\实验结果\\CNN_LSTM5"
    model_name_this='CNN_LSTM5'
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
for range in range(9999):
    do_Transformer_DA()
#do_Transformer_DA()
# do_Transformer_DA1()
# do_Transformer_DA2()
# do_Transformer_DA3()
# do_Transformer_DA4()
# do_Transformer_DA5()
# do_Transformer_DA6()
# do_Transformer_DA7()
# do_Transformer_DA8()
# do_Transformer_DA9()
# do_Transformer_v9_2()
# do_Transformer_v9_3()
# do_Transformer_v9_4()
# do_Transformer_v9_5()
# do_Transformer_v9_6()
# do_Transformer_v9_7()
# do_Transformer_v9_8()
# do_Transformer_v9_9()
# do_Transformer_v9_10()
#
# do_lstm()
# do_lstm1()
# do_lstm2()
# do_lstm3()
# do_lstm4()
# # do_lstm5()
# do_cnn_lstm()
# do_cnn_lstm1()
# do_cnn_lstm2()
# do_cnn_lstm3()
# do_cnn_lstm4()
# do_cnn_lstm5()
#do_Transformer_DA()

try:
    end =time.clock()
except:
    end =time.perf_counter()
print('Running time: %s Minutes'%((end-start)/60))
