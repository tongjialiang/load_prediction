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
import use_CNN
import use_CNN_v2
import use_LSTM
import use_Transformer
import use_Transformer_v2
import use_Transformer_v3
import use_Transformer_v4
import use_Transformer_v5
import use_Transformer_v6
import use_Transformer_v7
import use_Transformer_v8
import use_Transformer_v9
import use_LSTM
import use_CNN_LSTM_nfeatures
import use_GA_CNN_LSTM_nfeatures
#from load_prediction import use_Transformer_v9_2
import use_Transformer_v9_2
import use_Transformer_v9_3
import use_Transformer_v9_4
import use_Transformer_v9_5
import use_Transformer_v9_6
import use_Transformer_v9_7
import use_Transformer_v9_8
import use_Transformer_v9_9
import use_Transformer_v9_10
warnings.filterwarnings("ignore")


#使用模型名称到类名的映射
#model_dict={"Ridge":"use_Ridge"}
class control_layer:
    def __init__(self,dir_input,model_name,way4adjustParameter,csv_file,do_save=True,**args):
        self.dir_input=dir_input
        self.model_name=model_name

        self.model_class=''#正在调用的类名
        self.way4adjustParameter=way4adjustParameter
        self.do_save=do_save
        self.csv_file=csv_file



    def do_load_prediction(self):
        if os.path.exists(self.csv_file) == False:
            os.makedirs(self.csv_file)
        for root, dirs, filelist in os.walk(self.dir_input):
                res_csv=''
                model_res=dict()

            #读取pk数据
                for i in filelist:
                    if not i.endswith('pk'):
                        continue
                    #执行ST文件
                    if i.split("_")[-4] in ['MT','ST','RT']:
                        print(root+i)#D:\数据采样完成\按行业划分数据集_方案2-对每个企业标准化_采矿业_MT_data_4_1.pk
                        how_long=i.split("_")[-4]
                        file_name=i
                        # exec ("self.model_class="+model_dict[self.model_name])#use_Ridge
                        # print(self.model_class)
                        # my_model=self.model_class(dir=root+i)

                        if self.model_name=="CNN":
                            my_model = use_CNN.use_CNN(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()

                        if self.model_name == "CNN_v2":
                            my_model = use_CNN_v2.use_CNN_v2(dir=root + i,
                                                       how_long=how_long,
                                                       file_name=file_name,
                                                       way4adjustParameter=self.way4adjustParameter,
                                                       csv_file=self.csv_file)
                            my_model.do_model()

                        if self.model_name == "Transformer":
                            my_model = use_Transformer.use_Transformer_v1(dir=root + i,
                                                       how_long=how_long,
                                                       file_name=file_name,
                                                       way4adjustParameter=self.way4adjustParameter,
                                                       csv_file=self.csv_file)
                            my_model.do_model()

                        if self.model_name == "Transformer_v2":
                            my_model = use_Transformer_v2.use_Transformer_v2(dir=root + i,
                                                       how_long=how_long,
                                                       file_name=file_name,
                                                       way4adjustParameter=self.way4adjustParameter,
                                                       csv_file=self.csv_file)
                            my_model.do_model()

                        if self.model_name == "Transformer_v3":
                            my_model = use_Transformer_v3.use_Transformer_v3(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v4":
                            my_model = use_Transformer_v4.use_Transformer_v4(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v5":
                            my_model = use_Transformer_v5.use_Transformer_v5(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v6":
                            my_model = use_Transformer_v6.use_Transformer_v6(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()

                        if self.model_name == "Transformer_v7":
                            my_model = use_Transformer_v7.use_Transformer_v7(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v8":
                            my_model = use_Transformer_v8.use_Transformer_v8(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v9":
                            my_model = use_Transformer_v9.use_Transformer_v9(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v9_2":
                            my_model = use_Transformer_v9_2.use_Transformer_v9_2(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v9_3":
                            my_model = use_Transformer_v9_3.use_Transformer_v9_3(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v9_4":
                            my_model = use_Transformer_v9_4.use_Transformer_v9_4(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v9_5":
                            my_model = use_Transformer_v9_5.use_Transformer_v9_5(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v9_6":
                            my_model = use_Transformer_v9_6.use_Transformer_v9_6(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v9_7":
                            my_model = use_Transformer_v9_7.use_Transformer_v9_7(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v9_8":
                            my_model = use_Transformer_v9_8.use_Transformer_v9_8(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v9_9":
                            my_model = use_Transformer_v9_9.use_Transformer_v9_9(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "Transformer_v9_10":
                            my_model = use_Transformer_v9_10.use_Transformer_v9_10(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "LSTM":
                            my_model = use_LSTM.use_LSTM(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "CNN_LSTM":
                            my_model = use_CNN_LSTM_nfeatures.use_CNN_LSTM(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "GA_CNN_LSTM":
                            my_model = use_GA_CNN_LSTM_nfeatures.use_GA_CNN_LSTM(dir=root + i,
                                                                             how_long=how_long,
                                                                             file_name=file_name,
                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.do_save==True:
                            temp_df_csv,model_res_list=my_model.do_savedate()  # 方法调用的时候，会自动把前面的对象save_csv放到do_savecsv()的形参中
                            ##############################################
                            #my_model.do_drow_decision_tree()
                            if len(res_csv)==0:
                                res_csv=temp_df_csv
                                model_res.update({i: model_res_list})
                                continue
                            res_csv=res_csv.append(temp_df_csv)
                            model_res.update({i:model_res_list})

                        if len(res_csv)%10==0:#每处理10个文件down一个csv

                            nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                            csv_name = self.csv_file + "\\" + self.model_name + nowTime+".csv"
                            model_log_name = self.csv_file + "\\" + self.model_name +nowTime+ ".pk"
                            print(csv_name)
                            res_csv.to_csv(path_or_buf=csv_name, encoding="utf_8_sig", index=False)
                            #print(model_res)
                            #键：文件名，值[最优模型，调参过程日志]
                            #把最优模型和调参过程写入日志
                            with open(model_log_name, 'wb+') as f:
                                pickle.dump(model_res, f)
                            f.close()
                            res_csv=''
                            model_res=dict()

                #处理最后一批数据
                nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                csv_name = self.csv_file + "\\" + self.model_name + nowTime + ".csv"
                model_log_name = self.csv_file + "\\" + self.model_name + nowTime + ".pk"
                print(csv_name)
                res_csv.to_csv(path_or_buf=csv_name, encoding="utf_8_sig", index=False)
                # print(model_res)
                # 键：文件名，值[最优模型，调参过程日志]
                # 把最优模型和调参过程写入日志
                with open(model_log_name, 'wb+') as f:
                    pickle.dump(model_res, f)
                f.close()






