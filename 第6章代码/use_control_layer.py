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
import use_RandomForest
import use_DecisionTreeRegressor
import use_SVR
import use_KNN
import use_LightGBM
import use_Xgboost
import use_XGboost_v2
import use_CNN_v2
import use_LightGBM_v2
import use_RandomForest_v2
import use_Ridge_v2
import use_KNN_v2
import use_SVR_v2
import use_DecisionTreeRegressor_v2
import use_Ridge_v3
import use_SVR_v3_onepredict
import use_RandomForest_v3_onepredict
import use_RandomForest_v3_news
# from load_prediction import use_SVR_v3, use_DecisionTreeRegressor_v3, use_KNN_v3, use_RandomForest_v3, use_XGboost_v3, \
#     use_LightGBM_v3, use_Ridge_v4, use_SVR_v4, use_DecisionTreeRegressor_v4, use_LightGBM_v4, \
#     use_RandomForest_v4
import use_SVR_v3, use_DecisionTreeRegressor_v3, use_KNN_v3, use_RandomForest_v3, use_XGboost_v3, \
    use_LightGBM_v3, use_Ridge_v4, use_SVR_v4, use_DecisionTreeRegressor_v4, use_LightGBM_v4, \
    use_RandomForest_v4,use_RandomForest_v3_news

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

                        #model_DecisionTreeRegressor
                        if self.model_name=="Ridge":
                            my_model=use_Ridge.model_Ridge(dir=root+i,how_long=how_long,file_name=file_name,way4adjustParameter=self.way4adjustParameter,csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="DecisionTreeRegressor":
                            my_model = use_DecisionTreeRegressor.model_DecisionTreeRegressor(dir=root + i, how_long=how_long, file_name=file_name,
                                                             way4adjustParameter=self.way4adjustParameter,
                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "RandomForestRegressor":
                            my_model = use_RandomForest.model_RandomForestRegressor(dir=root + i,
                                                                                             how_long=how_long,
                                                                                             file_name=file_name,
                                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "RandomForestRegressor_v2":
                            my_model = use_RandomForest_v2.model_RandomForestRegressor_v2(dir=root + i,
                                                                                             how_long=how_long,
                                                                                             file_name=file_name,
                                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "RandomForestRegressor_v3":
                            my_model = use_RandomForest_v3.model_RandomForestRegressor_v3(dir=root + i,
                                                                                             how_long=how_long,
                                                                                             file_name=file_name,
                                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "RandomForestRegressor_v3_onepredict":
                            my_model = use_RandomForest_v3_onepredict.model_RandomForestRegressor_v3_onepredict(dir=root + i,
                                                                                             how_long=how_long,
                                                                                             file_name=file_name,
                                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "RandomForestRegressor_v3_news":
                            my_model = use_RandomForest_v3_news.model_RandomForestRegressor_v3_news(dir=root + i,
                                                                                             how_long=how_long,
                                                                                             file_name=file_name,
                                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "RandomForestRegressor_v4":
                            my_model = use_RandomForest_v4.model_RandomForestRegressor_v4(dir=root + i,
                                                                                             how_long=how_long,
                                                                                             file_name=file_name,
                                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "SVR":
                            my_model = use_SVR.model_SVR(dir=root + i,
                                                                                             how_long=how_long,
                                                                                             file_name=file_name,
                                                                                             way4adjustParameter=self.way4adjustParameter,
                                                                                             csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="KNN":
                            my_model = use_KNN.KNeighborsRegressor(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)

                            my_model.do_model()

                        if self.model_name=="LightGBM":
                            my_model = use_LightGBM.LGBMRegressor(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="LightGBM_v2":
                            my_model = use_LightGBM_v2.LGBMRegressor(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="LightGBM_v3":
                            my_model = use_LightGBM_v3.LGBMRegressor(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="LightGBM_v4":
                            my_model = use_LightGBM_v4.LGBMRegressor(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="Xgboost":
                            my_model = use_Xgboost.XGBRegressor(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()

                        if self.model_name=="Xgboost_v2":
                            my_model = use_XGboost_v2.XGBRegressor(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="Xgboost_v3":
                            my_model = use_XGboost_v3.Model_Xgboost_v3(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="Xgboost_v4":
                            my_model = use_XGboost_v3.Model_Xgboost_v4(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="Ridge_v2":
                            my_model = use_Ridge_v2.model_Ridge_v2(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="Ridge_v3":
                            my_model = use_Ridge_v3.model_Ridge_v3(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="Ridge_v4":
                            my_model = use_Ridge_v4.model_Ridge_v4(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="KNN_v2":
                            my_model = use_KNN_v2.KNeighborsRegressor_v2(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="KNN_v3":
                            my_model = use_KNN_v3.KNeighborsRegressor_v3(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="SVR_v2":
                            my_model = use_SVR_v2.model_SVR_v2(dir=root + i,
                                                         how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="DecisionTreeRegressor_v2":
                            my_model = use_DecisionTreeRegressor_v2.model_DecisionTreeRegressor_v2(dir=root + i,
                                                     how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="DecisionTreeRegressor_v3":
                            my_model = use_DecisionTreeRegressor_v3.model_DecisionTreeRegressor_v3(dir=root + i,
                                                     how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name == "DecisionTreeRegressor_v4":
                            my_model = use_DecisionTreeRegressor_v4.model_DecisionTreeRegressor_v4(dir=root + i,
                                                                                                   how_long=how_long,
                                                                                                   file_name=file_name,
                                                                                                   way4adjustParameter=self.way4adjustParameter,
                                                                                                   csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="SVR_v3":
                            my_model = use_SVR_v3.model_SVR_v3(dir=root + i,
                                                     how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="SVR_v3_oncepredict":
                            my_model = use_SVR_v3_onepredict.model_SVR_v3_oncepredict(dir=root + i,
                                                     how_long=how_long,
                                                         file_name=file_name,
                                                         way4adjustParameter=self.way4adjustParameter,
                                                         csv_file=self.csv_file)
                            my_model.do_model()
                        if self.model_name=="SVR_v4":
                            my_model = use_SVR_v4.model_SVR_v4(dir=root + i,
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






