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
from tensorboard.backend.event_processing import event_accumulator
#设置需要保留的小数位数
pd.set_option('precision', 3)
ea = event_accumulator.EventAccumulator('D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第9章DA-Transformer\\log\\按频域分解聚类划分数据集V2_领域自适应-汇总标准化_stclass_26_ST_data_70_202022-12-09T07-11-28')
ea.Reload()
to="D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第9章DA-Transformer\\"
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
# ###########################################################
#
# sns.set_style('darkgrid',{'font.sans-serif':['SimHei','Arial']})
# sns.set_palette(palette="dark")
# #['鉴别器的训练损失_单位_批次_', '负荷预测的训练损失_单位_批次_', '训练时MMD值_单位_批次_',
# # '负荷预测_MMD_鉴别器损失的加权和_单位_批次_', '负荷预测验证集损失_单位_epoch_']
#
# data = ea.scalars.Items('负荷预测验证集损失_单位_epoch_')
# x=[]
# y=[]
# for i in data:
#     y.append(i.value)
#     x.append(i.step)
# plt.xlabel("时期(epoch)")
# plt.ylabel("负荷预测损失(MSE)")
# #plt.title("负荷预测损失（测试阶段）")
# #plt.plot(x,'-',color='black')
# sns.lineplot(x[:1000],y[:1000],ci='red',label="负荷预测损失（测试阶段）")
# plt.legend()
# plt.savefig(to+"负荷预测损失（测试阶段）.jpg",dpi=1000)
# plt.show()
#
# ################################################################
# data = ea.scalars.Items('负荷预测_MMD_鉴别器损失的加权和_单位_批次_')
# x=[]
# y=[]
# for i in data:
#     y.append(i.value)
#     x.append(i.step)
# plt.xlabel("批次")
# plt.ylabel("总损失（MSE）")
# #plt.title("总损失曲线（-域鉴别器损失+MMD+负荷预测损失）")
# #plt.plot(x,'-',color='black')
# sns.lineplot(x[:1000],y[:1000],ci='red',label="总损失曲线（-域鉴别器损失+MMD+负荷预测损失）")
# plt.legend()
# plt.savefig(to+"总损失曲线（-域鉴别器损失+MMD+负荷预测损失）.jpg",dpi=1000)
# plt.show()
#
# ################################################################
# data = ea.scalars.Items('负荷预测的训练损失_单位_批次_')
# x=[]
# y=[]
# for i in data:
#     y.append(i.value)
#     x.append(i.step)
# plt.xlabel("批次")
# plt.ylabel("负荷预测损失（MSE）")
# #plt.title("负荷预测损失曲线（训练阶段）")
# #plt.plot(x,'-',color='black')
# sns.lineplot(x[:3000],y[:3000],ci='red',label="负荷预测损失曲线（训练阶段）")
# plt.legend()
# plt.savefig(to+"负荷预测损失曲线（训练阶段）.jpg",dpi=1000)
# plt.show()
#
# ################################################################
# data = ea.scalars.Items('训练时MMD值_单位_批次_')
# x=[]
# y=[]
# for i in data:
#     y.append(i.value)
#     x.append(i.step)
# plt.xlabel("批次")
# plt.ylabel("MMD值")
# #plt.title("MMD值（训练阶段）")
# #plt.plot(x,'-',color='black')
# sns.lineplot(x[:1000],y[:1000],ci='red',label="MMD值（训练阶段）")
# plt.legend()
# plt.savefig(to+"mmd.jpg",dpi=1000)
# plt.show()
# ################################################################
# data = ea.scalars.Items('鉴别器的训练损失_单位_批次_')
# x=[]
# y=[]
# for i in data:
#     y.append(i.value)
#     x.append(i.step)
# plt.xlabel("训练批次")
# plt.ylabel("域鉴别器损失")
# #plt.title("域鉴别器的损失曲线（训练阶段）")
# #plt.plot(x,'-',color='black')
# sns.lineplot(x[:1000],y[:1000],ci='red',label="域鉴别器的损失曲线（训练阶段）")
# plt.legend()
# plt.savefig(to+"域鉴别器的损失曲线（训练阶段）.jpg",dpi=1000)
# plt.show()
#
# ################################################################
with open('D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第9章da-transformer\\yyhat.pk', 'rb') as f:
    data = pickle.load(f)
f.close()
#print(data.shape)
x=data[0][2]
print(x)
y=data[1][2]

plt.xlabel("时间")
plt.ylabel("电力负荷（KW）")
#plt.title("负荷预测与真实值对比")
#plt.plot(x,'-',color='black')
sns.lineplot(range(len(x)),x,label="真实值",ci='red')
sns.lineplot(range(len(y)),y,label="预测值",ci='red')
plt.legend()
plt.savefig(to+"真实与预测对比.jpg",dpi=1000)
plt.show()
