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
import matplotlib.pyplot as plt
import gc
import json
import sys
import seaborn as sns
#设置需要保留的小数位数
pd.set_option('precision', 3)

###########################################################
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
warnings.filterwarnings("ignore")
# 解决中文和负号显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 使显示图标自适应
plt.rcParams['figure.autolayout'] = True
#改进Transformer调参_学习率##########################################################
# topath="D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第8章基于改进Transformer模型的负荷预测方法\\"+"改进Transformer调参_学习率.jpg"
# y1=[[1.02,1.013,1.027,1.01,1.02,1.004,1.002,1.002,1.0046,1.0023],
#     [0.97,0.985,0.971,0.985,0.985,0.989,0.984,0.989,0.967,0.98,0.981,0.992,0.982],
#     [0.96,0.96,0.960,0.960,0.960,0.962,0.961,0.96,0.96,0.96,0.96,0.96,0.96],
#     [0.980,0.980,0.98,0.980,0.980,0.98,0.980,0.980,0.980]]
# plt.boxplot(y1,labels=["0.001","0.005","0.01","0.03"],showmeans=1)
# plt.xlabel("学习率")
# plt.grid(True)
# #plt.legend()
# plt.ylabel("均方损失")
# plt.savefig(topath,dpi=1000)
# plt.show()
#改进Transformer调参_词嵌入维度##########################################################
# topath="D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第8章基于改进Transformer模型的负荷预测方法\\"+"改进Transformer调参_词嵌入维度.jpg"
# y1=[[1.061,1.06,1.061,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06],
#     [0.961,0.961,0.961,0.961,0.961,0.961,0.961,0.961,0.961,0.961],
#     [0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604],
#     [0.65,0.65,0.65,0.65,0.65,0.65,0.65,0.65,0.65,0.65,0.65,0.65]]
# plt.boxplot(y1,labels=["8","16","32","64"],showmeans=1)
# plt.xlabel("词嵌入维度")
# plt.grid(True)
# #plt.legend()
# plt.ylabel("均方损失")
# plt.savefig(topath,dpi=1000)
# plt.show()
#改进Transformer调参_LSTM层数##########################################################
# topath="D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第8章基于改进Transformer模型的负荷预测方法\\"+"改进Transformer调参_长短时记忆网络层数.jpg"
# y1=[[0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604],
#     [0.837,0.837,0.837,0.837,0.837,0.837,0.837,0.837,0.837,0.837],
#     [1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086],
#     ]
# plt.boxplot(y1,labels=["1","2","3"],showmeans=1)
# plt.xlabel("长短时记忆网络层数")
# plt.grid(True)
# #plt.legend()
# plt.ylabel("均方损失")
# plt.savefig(topath,dpi=1000)
# plt.show()
#改进Transformer调参_自注意力层数##########################################################
# topath="D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第8章基于改进Transformer模型的负荷预测方法\\"+"改进Transformer调参_自注意力层数.jpg"
# y1=[[0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604],
#     [0.656,0.656,0.656,0.656,0.656,0.656,0.656,0.656,0.656,0.656],
#     [0.702,0.702,0.702,0.702,0.702,0.702,0.702,0.702,0.702,0.702],
#     ]
# plt.boxplot(y1,labels=["1","2","3"],showmeans=1)
# plt.xlabel("自注意力层数")
# plt.grid(True)
# #plt.legend()
# plt.ylabel("均方损失")
# plt.savefig(topath,dpi=1000)
# plt.show()
##################################################################
# topath="D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第8章基于改进Transformer模型的负荷预测方法\\"+"改进Transformer调参_自注意力头数.jpg"
# y1=[[0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604],
#      [0.598, 0.598, 0.598, 0.598, 0.598, 0.598, 0.598, 0.598, 0.598, 0.598],
#      [0.595,0.595,0.595,0.595,0.595,0.595,0.595,0.595,0.595,0.595],
#      ]
# plt.boxplot(y1,labels=["1","2","4"],showmeans=1)
# plt.xlabel("自注意力层数")
# plt.grid(True)
# #plt.legend()
# plt.ylabel("均方损失")
# plt.savefig(topath,dpi=1000)
# plt.show()
#######################################################
topath="D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第8章基于改进Transformer模型的负荷预测方法\\"+"改进Transformer网络结构的消融实验.jpg"
y1=[[0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604,0.604],
     [1.115, 0.74, 1.113, 0.8, 1.115, 1.115, 0.77,0.804],
     [0.883,0.707,0.64,0.65,0.678,0.7,0.695,0.656,0.587],
     ]
plt.boxplot(y1,labels=["本研究","仅LSTM结构","LSTM+CNN结构"],showmeans=1)
plt.xlabel("网络结构")
plt.grid(True)
#plt.legend()
plt.ylabel("均方损失")
plt.savefig(topath,dpi=1000)
plt.show()