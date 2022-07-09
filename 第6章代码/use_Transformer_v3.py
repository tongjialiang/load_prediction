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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
from sklearn.neighbors import KNeighborsRegressor
import use_save_csv_DL
import gc
import auto_search
import torch
import torch.nn.functional as F
import copy
from torch import nn
from torch.utils import data
import torch.optim as optim
import torchvision
warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter #数据可视化

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# 设置参数搜索范围
ajustParameter ='''
手动调参
'''
#设置数据可视化日志存储目录
view_dir="D:\\实验记录\\实验结果\\Transformer_v3\\log\\"

# 设置模型
# model = KNeighborsRegressor()

# 设置model-name，记录日志用
model_name = 'Transformer_v3'

#对数据集欠采样
use_data = False
use_data_ratio = 1
min_use_data = 2000

# 调参是否欠采样？
use_adjust_parameter = False
use_adjust_parameter_ratio = 0.005
min_use_adjust_parameter = 9000
max_use_adjust_parameter = 10000

#设置CNN模型batchsize的值
batch_size=32

#如果feature_extract是True,冻住所有参数
feature_extract = False
#是否加载预训练模型
use_pretrained = False

#模型保存
filename='checkpoint.pth'

#跑多少个epochs
num_epochs=100

#学习率
lr=0.006

#多少个epoch无改善就触发earlystopping
num_epoch_quiet=8

#优化算法
myopt="AdamW"

#Transfomer，Embedding的维数 64 128 256 32
feature_size=64  #64 best

#TransformerEncoder的层数 234
num_layers=1

#使用多少头的Multi-Head Attention 4 8  10 16
nhead=8

#指定随机数种子
ran=123#23

#设置需要输出为csv结果文件的超参数
res_ajustParameter="batch_size="+str(batch_size)+"\n"+"num_epochs="+str(num_epochs)+"\n"+"lr="+str(lr)+"\n"+"num_epoch_quiet="+str(num_epoch_quiet)+"\n"+"myopt="+myopt+"\n"+"feature_size="+str(feature_size)+"\n"+"num_layers="+str(num_layers)+"\n"+"nhead="+str(nhead)+"\n"+"ran="+str(ran)+"\n"
res_ajustParameter_data=[batch_size,num_epochs,lr,num_epoch_quiet,myopt,feature_size,num_layers,nhead,ran]
# dir='D:\\数据采样完成\\按行业划分数据集_方案1-汇总标准化_住宿和餐饮业_MT_data_4_1.pk'
class use_Transformer_v3():###################################################################################################要修改
    def __init__(self, dir, how_long, file_name, way4adjustParameter, csv_file):
        #super(CNN, self).__init__()
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
        self.ajustParameter = ajustParameter # 调整的参数
        self.way4ajustParameter = self.way4adjustParameter  # 调参算法
        self.way4cv = 4  # 交叉验证的折数
        self.way4adjustParameter_score = ''  # 调参评价指标
        self.best_estimator = ''  # 最佳模型-保存
        self.best_params = ''  # 最佳参数
        self.MSE_score = 0
        self.RMSE_score = 0
        self.MAE = 0
        self.R2_score = 0
        self.res_ajustParameter = ''  # 调参结果
        self.csv_file = csv_file
        self.use_time=0
        self.opt=myopt#优化方法
        self.state=''
        self.total_epochs=0


    #writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    def do_model(self):
        dataloaders = {}    #训练数据集封装
        dataloaders_adjust={}#调参数据集封装
        writer = SummaryWriter(
        log_dir=view_dir + self.file_name.split(".")[-2])
        #C:\实验记录\实验结果\CNN_v2\log\按聚类划分数据集_方案1-汇总标准化_c25_class25_RT_data_48_1
        try:
            start = time.clock()
        except:
            start = time.perf_counter()

        print(self.how_long)
        with open(self.dir, 'rb') as f:
            data = pickle.load(f)
        num_data = len(data[0])
        self.num_data = num_data  #数据量

        # 随机打散
        print(data[0].shape)  # (33447, 9, 4)
        print(data[1].shape)  # (33447, 3, 4)

        index = np.arange(num_data)
        #固定样本随机打散的随机数种子
        np.random.seed(ran)
        np.random.shuffle(index)
        data[0] = data[0][index, :, :]  # X_train是训练集，y_train是训练标签
        data[1] = data[1][index]
        gc.collect()

        # 如果是小批量数据集：使得测试集==训练集
        short_flag=False
        if num_data<1000:
            short_flag=True

        #获取特征个数
        num_feature=data[0].shape[1]
        print(num_feature)  #9

        ###修正
        self.predict_type = self.how_long  # 修正
        #D:\数据采样完成new\按地域划分数据集_方案1-汇总标准化_临安区_MT_data_9_3.pk
        self.x_type = self.file_name.split("_")[-2]#
        self.y_type = self.file_name.split("_")[-1].split(".")[0]

        # 95%是训练数据
        if self.how_long in ["RT","MT","ST"]:
            #修正
            X_train = data[0][:int(num_data * 0.95), :, 1]
            X_train = X_train.reshape(len(X_train), -1)
            Y_train = data[1][:int(num_data * 0.95),:, 1]
            X_test = data[0][int(num_data * 0.95):, :, 1]
            X_test = X_test.reshape(len(X_test), -1)
            Y_test = data[1][int(num_data * 0.95):,:, 1]

            # print(X_train.shape)#(31774, 9)
            # print(Y_train.shape)#(31774, 3)
            # print(X_test.shape)#(1673, 9)
            # print(Y_test.shape)#(1673, 3)
            # print(Y_train[0])

            if short_flag:
                X_test=X_train[:100]
                Y_test=Y_train[:100]
            #删除
            # self.x_type = '输入:过去48*15分钟数据'
            # self.y_type = '输出：未来15-150分钟数据'

        # if self.how_long == 'MT' and self.way4standard == '方案1-汇总标准化':
        #     X_train = data[0][:int(num_data * 0.95), :, 1]  # 2供电容量
        #     X_train = X_train.reshape(len(X_train), -1)
        #     Y_train = data[1][:int(num_data * 0.95), 1]
        #     X_test = data[0][int(num_data * 0.95):, :, 1]
        #     X_test = X_test.reshape(len(X_test), -1)
        #     Y_test = data[1][int(num_data * 0.95):, 1]
        #     self.predict_type = '长期预测'
        #     self.x_type = '输入:过去4个月数据'
        #     self.y_type = '输出：未来1-3个月数据'
        #     print("X_train.shape")
        #     print(X_train.shape) #(193, 4)
        #     print("Y_train.shape")
        #     print(Y_train.shape)#(193,)
        #
        #
        # if self.how_long == 'ST' and self.way4standard == '方案1-汇总标准化':
        #     X_train = data[0][:int(num_data * 0.95), :, 1]  # 6供电容量
        #     print(data[0].shape)
        #     X_train = X_train.reshape(len(X_train), -1)
        #     Y_train = data[1][:int(num_data * 0.95), 1]
        #     X_test = data[0][int(num_data * 0.95):, :,1]
        #     X_test = X_test.reshape(len(X_test), -1)
        #     Y_test = data[1][int(num_data * 0.95):, 1]
        #     self.predict_type = '短期预测'
        #     self.x_type = '输入:过去30天数据'
        #     self.y_type = '输出：未来1-4天数据'
        #     print("X_train.shape")
        #     print(X_train.shape) #(228, 30)
        #     print("Y_train.shape")
        #     print(Y_train.shape)#(228,)
        #
        # if self.how_long == 'MT' and self.way4standard == '方案2-对每个企业标准化':
        #     X_train = data[0][:int(num_data * 0.95), :, 1]  # 2供电容量 7最小负荷 舍弃
        #     X_train = X_train.reshape(len(X_train), -1)
        #     Y_train = data[1][:int(num_data * 0.95), 1]
        #     X_test = data[0][int(num_data * 0.95):, :, 1]
        #     X_test = X_test.reshape(len(X_test), -1)
        #     Y_test = data[1][int(num_data * 0.95):, 1]
        #     self.predict_type = '长期预测'
        #     self.x_type = '输入:过去4个月数据'
        #     self.y_type = '输出：未来1-3个月数据'
        #
        # if self.how_long == 'ST' and self.way4standard == '方案2-对每个企业标准化':
        #     X_train = data[0][:int(num_data * 0.95), :, 1]  # 6供电容量 3最小负荷
        #     X_train = X_train.reshape(len(X_train), -1)
        #     Y_train = data[1][:int(num_data * 0.95), 1]
        #     X_test = data[0][int(num_data * 0.95):, :, 1]
        #     X_test = X_test.reshape(len(X_test), -1)
        #     Y_test = data[1][int(num_data * 0.95):, 1]
        #     self.predict_type = '短期预测'
        #     self.x_type = '输入:过去30天数据'
        #     self.y_type = '输出：未来1-4天数据'
        gc.collect()

        #总数据集欠采样
        len_x = len(X_train)
        if use_data and len_x > min_use_data:
            use_data_num = int(len_x * use_data_ratio)
            use_data_num = max(use_data_num, min_use_data)
            X_train = X_train[:use_data_num, :]
            Y_train = Y_train[:use_data_num]
        gc.collect()

        # 是否用GPU训练
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print('CUDA is not available.  Training on CPU ...')
        else:
            print('CUDA is available!  Training on GPU ...')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 调参欠采样
        num_sample = len(X_train)
        if use_adjust_parameter and num_sample > min_use_adjust_parameter:
            #欠采样
            num_use_sample = int(num_sample * use_adjust_parameter_ratio)
            if num_use_sample > max_use_adjust_parameter:
                num_use_sample = max_use_adjust_parameter
            if num_use_sample < min_use_adjust_parameter:
                num_use_sample = min_use_adjust_parameter
            X_train_adjust=X_train[:num_use_sample, :]
            Y_train_adjust=Y_train[:num_use_sample]
        else:
            #不使用欠采样
            X_train_adjust=X_train
            Y_train_adjust=Y_train

        #print(X_train)
        #numpy数组转为tensor

        #调整样本的数据类型和维度，适配当前的深度模型
        #转化为当前模型所需要的格式

        #处理第1组：调参数据
        #X_train_adjust [31774, 9,1]        #样本
        #Y_train_adjust [31774]             #用于预测时评估模型效果的标签
        #Y_train_adjust_seq [31774, 9,1]    #用于训练的标签

        X_train_adjust=torch.from_numpy(X_train_adjust.astype(np.float32))
        #print(X_train_adjust.size())#torch.Size([31774, 9])
        #print(X_train_adjust[0])
        Y_train_adjust=torch.from_numpy(Y_train_adjust[:,0].astype(np.float32) ) #用于预测时评估模型效果
        #print(Y_train_adjust.size())#torch.Size([31774])
        #print(Y_train_adjust[0])
        Y_train_adjust_seq=Y_train_adjust.unsqueeze(-1)    #用于训练
        #print(Y_train_adjust_seq.size())#torch.Size([31774, 1])
        Y_train_adjust_seq=torch.cat((X_train_adjust,Y_train_adjust_seq),1)[:,1:]
        #print(Y_train_adjust_seq.size())#torch.Size([31774, 9])
        #print(Y_train_adjust_seq[0])
        X_train_adjust=X_train_adjust.unsqueeze(-1)
        Y_train_adjust_seq=Y_train_adjust_seq.unsqueeze(-1)

        # 处理第2组：测试用
        #X_test         [1673, 9, 1]
        #Y_test         [1673]
        #Y_test_seq     [1673, 9, 1]
        #Y_test_real    [1673, 3]   #用于循环预测
        X_test = torch.from_numpy(X_test.astype(np.float32) )
        Y_test_real=torch.from_numpy(Y_test.astype(np.float32) )
        Y_test = torch.from_numpy(Y_test[:,0].astype(np.float32) )
        Y_test_seq=Y_test.unsqueeze(-1)
        Y_test_seq=torch.cat((X_test,Y_test_seq),1)[:,1:]
        X_test=X_test.unsqueeze(-1)
        Y_test_seq=Y_test_seq.unsqueeze(-1)

        # print(X_test.shape)
        # print(Y_test.shape)
        # print(Y_test_seq.shape)
        # print(Y_test_real.shape)
        # 1/0

        # 处理第3组：训练用
        #X_train            torch.Size([31774, 9, 1])
        #Y_train            torch.Size([31774])
        #Y_train_seq        torch.Size([31774, 9, 1])
        X_train = torch.from_numpy(X_train.astype(np.float32) )
        Y_train =torch.from_numpy(Y_train[:,0].astype(np.float32) )
        Y_train_seq=Y_train.unsqueeze(-1)
        Y_train_seq=torch.cat((X_train,Y_train_seq),1)[:,1:]
        X_train=X_train.unsqueeze(-1)
        Y_train_seq=Y_train_seq.unsqueeze(-1)

        # print(X_train.shape)
        # print(Y_train.shape)
        # print(Y_train_seq.shape)
        # 1/0

        # 用dataloaders封装调参数据
        torch_dataset_train_adjust = torch.utils.data.TensorDataset(X_train_adjust, Y_train_adjust_seq)
        torch_dataset_valid_adjust = torch.utils.data.TensorDataset(X_test, Y_test_seq)
        dataloaders_adjust['train'] = torch.utils.data.DataLoader(torch_dataset_train_adjust,batch_size=batch_size,shuffle=False)
        dataloaders_adjust['valid'] = torch.utils.data.DataLoader(torch_dataset_valid_adjust, batch_size=batch_size, shuffle=False)

        #用dataloaders封装训练数据
        torch_dataset_train = torch.utils.data.TensorDataset(X_train, Y_train_seq)
        torch_dataset_valid = torch.utils.data.TensorDataset(X_test, Y_test_seq)
        dataloaders['train'] = torch.utils.data.DataLoader(torch_dataset_train,batch_size=batch_size,shuffle=False)
        dataloaders['valid'] = torch.utils.data.DataLoader(torch_dataset_valid, batch_size=batch_size, shuffle=False)

        #如果feature_extracting为真，执行该函数会冻住所有行的参数
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False  # 冻住所有层的参数

        #深度学习建模
        # use_pretrained是否加载预训练模型
        # feature_extract 是否加载预训练模型
        #temp = X_train_adjust.size()[2]  # 输入的第三个维度 例如（b,1,30）的30
        class Transformer_Time_series(nn.Module):
            def __init__(self, feature_size=feature_size, num_layers=num_layers, dropout=0.1):
                super(Transformer_Time_series, self).__init__()
                self.model_type = 'Transformer'
                self.src_mask = None
                self.pos_encoder = PositionalEncoding(feature_size)
                self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
                self.decoder = nn.Linear(feature_size, 1)
                self.init_weights()

            def init_weights(self):
                initrange = 0.1
                self.decoder.bias.data.zero_()
                self.decoder.weight.data.uniform_(-initrange, initrange)

            def forward(self, src):  # src-->(9,32,1)   (input_window,batchsize,特征数)
                if self.src_mask is None or self.src_mask.size(0) != len(src):
                    device = src.device
                    mask = self._generate_square_subsequent_mask(len(src)).to(device)
                    self.src_mask = mask
                # print(src.shape)  #torch.Size([100, 10, 1]) torch.Size([9, 32, 1])
                # 1/0
                src = self.pos_encoder(src)  #  # (9,32,1)+([9, 32, 64])-->[9, 32, 64]
                # print(src.shape)#[9, 32, 64]
                # 1/0
                output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
                # print(output.size())#[9, 32, 64]
                # 1/0
                output = self.decoder(output)  # [9, 32, 1]
                # print(output.size())
                # 1/0
                return output

            def _generate_square_subsequent_mask(self, sz):  # 参数100
                mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
                # trui 返回矩阵上三角部分,其余部分定义为0
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                return mask

        class PositionalEncoding(nn.Module):

            def __init__(self, d_model, max_len=5000):
                super(PositionalEncoding, self).__init__()
                pe = torch.zeros(max_len, d_model)  # 5000,250的矩阵，元素全为0
                # print(pe)
                # print(pe.shape)    torch.Size([5000, 250])
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                # print(position.size())#torch.Size([5000, 1])
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                # print(math.log(10000.0))#9.21
                # print(div_term.shape)#torch.Size([125])
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).transpose(0, 1)
                #print(pe)
                #print(pe.shape)  # torch.Size([5000, 1, 250])
                # pe.requires_grad = False
                self.register_buffer('pe', pe)

            def forward(self, x):  # x-->(9,32,1)   (input_window,batchsize,特征数)
                res = x + self.pe[:x.size(0), :]
                # print(res.shape)#[9, 32, 64]
                # 1/0
                return res  # (9,32,1)+([9, 32, 64])
        #模型初始化
        def model_init():
            model_ft=Transformer_Time_series()
            # with torch.set_grad_enabled(False):
            #     writer.add_graph(model_ft, input_to_model=(X_test,), verbose=False)
            #print(model_ft.named_parameters())
            # GPU计算
            model_ft = model_ft.to(device)
            # 是否训练所有层 params_to_update存放要更新的参数
            params_to_update = model_ft.parameters()

            print("Params to learn:")
            if feature_extract:
                params_to_update = []  # 存放要更新的参数
                for name, param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        #print("\t", name)
            else:
                for name, param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        pass#print("\t", name)
                        # print(param)
            return model_ft, params_to_update
        #print(criterion)

        def train_model(model, dataloaders, params_to_update, lr,num_epochs=num_epochs,is_inception=False
                        ,filename=filename):  # is_inception=False不用
            # 优化器设置
            optimizer_ft = torch.optim.AdamW(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, 1.0, gamma=0.95)  # 学习率每7个epoch衰减成原来的1/10
            criterion = torch.nn.MSELoss(reduction='mean')

            since = time.time()
            best_loss = 9999999
            #model.to(device)

            train_losses = []
            valid_losses = []

            LRs = [optimizer_ft.param_groups[0]['lr']]  # learningrate

            best_model_wts = copy.deepcopy(model.state_dict())
            current_epoch=0
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
                self.total_epochs=epoch
                # 训练和验证
                #如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加model.train()
                for phase in ['train', 'valid']:
                    if phase == 'train':
                        model.train()  # 训练
                    else:
                        model.eval()  # 验证

                    running_loss = 0.0

                    # 把数据都取个遍
                    for inputs, labels in dataloaders[phase]:
                        #print(inputs.size(),labels.size())#[64, 30, 1]

                        #b,w,h-->w,b,h
                        inputs=inputs.permute(1,0,2)
                        labels = labels.permute(1, 0, 2)
                        # print(inputs.size(), labels.size()) #torch.Size([9, 32, 1]) torch.Size([9, 32, 1])
                        # 1/0
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        #print("从dataloader中取到的数据维度")
                        #print(inputs.size())#torch.Size([32, 1, 30])
                        #print(labels.size())#torch.Size([32])
                        # 清零
                        optimizer_ft.zero_grad()
                        # 只有训练的时候计算和更新梯度
                        with torch.set_grad_enabled(phase == 'train'):  # 如果是train阶段（true）,计算梯度，如果是valid阶段（false）,不计算梯度
                            if is_inception and phase == 'train':  # 以下几行不用看
                                print("ERROR")
                                # outputs, aux_outputs = model(inputs)
                                # loss1 = criterion(outputs, labels)
                                # loss2 = criterion(aux_outputs, labels)
                                # loss = loss1 + 0.4 * loss2
                            else:  # resnet执行的是这里
                                #print(model)
                                outputs = model(inputs) #[9, 32, 1]-->[9, 32, 1]
                                loss = criterion(outputs, labels)

                            #_, preds = torch.max(outputs, 1)

                            # 训练阶段更新权重
                            if phase == 'train':
                                loss.backward()
                                optimizer_ft.step()

                        # 计算损失
                        running_loss += loss.item() * inputs.size(0)
                        #running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                    time_elapsed = time.time() - since
                    print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                    if phase == 'valid':
                        writer.add_scalar("loss/valid", epoch_loss, epoch)
                    if phase == 'train':
                        writer.add_scalar("loss/train", epoch_loss, epoch)

                    # 得到最好那次的模型
                    if phase == 'valid' and epoch_loss+0.001 < best_loss:
                        current_epoch = epoch  # 给early_stopping提供依据
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝模型的状态字典
                        state = {
                            'state_dict': model.state_dict(),
                            'best_loss': best_loss,
                            'optimizer': optimizer_ft.state_dict(),
                            'res_ajustParameter':res_ajustParameter
                        }
                        #torch.save(state, filename)
                    if phase == 'valid':
                        #val_acc_history.append(epoch_loss)
                        valid_losses.append(epoch_loss)#单位是epoch
                        scheduler.step()
                    if phase == 'train':
                        #train_acc_history.append(epoch_loss)
                        train_losses.append(epoch_loss)

                print('Optimizer learning rate : {:.7f}'.format(optimizer_ft.param_groups[0]['lr']))
                LRs.append(optimizer_ft.param_groups[0]['lr'])
                print()

                # earlystopping
                if phase == 'valid' and (epoch - current_epoch > num_epoch_quiet):
                    print("early_stopping...start........")
                    break

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val loss: {:4f}'.format(best_loss))

            state["train_losses"]=train_losses
            state["valid_losses"]=valid_losses
            #state["params"]=params_to_update
            # 训练完后用最好的一次当做模型最终的结果
            model.load_state_dict(best_model_wts)  # 把最好的模型参数载入到模型中
            return model, valid_losses, train_losses, LRs ,state,best_loss
        model_ft, params_to_update = model_init()
        # print(model_ft)
        # print(params_to_update)
        # 1/0
        #开始训练
        print("开始训练")
        model_best, valid_losses, train_losses, LRs ,state,best_loss = \
            train_model(model_ft,dataloaders,num_epochs=num_epochs,is_inception=False,lr=lr,params_to_update=params_to_update)
        print("结束训练")

        #拿到最佳模型，计算MSE RMSE MAE R2
        with torch.set_grad_enabled(False):
            #model_best.cpu()
            model_best.eval()
            #print(next(model_best.parameters()).device)  #cpu
            #print(next(model_best.parameters())) #打印模型参数

            # print(X_test.size())#[1673, 9, 1]
            # print(X_test.permute(1,0,2).size())#[9, 1673, 1]
            # 1/0
            #print(torch.cuda.is_available())
            X_test=X_test.to(device)
#############################循环预测###############################################
            Y_predict_res = np.zeros_like(Y_test_real)#[1673, 3]
            Y_predict_res=torch.from_numpy(Y_predict_res.astype(np.float32)).to(device)
            # print(Y_predict)
            # 1/0
            # print(Y_test_real.shape) #[1673, 3]
            # 1/0
            for i in range(Y_test_real.shape[1]):  #3
                # print(X_test.shape) [1673, 9, 1]
                # 1/0
                Y = model_best(X_test.permute(1,0,2))#Y =[9, 1673, 1]
                Y = Y.to(device)
                Y = torch.squeeze(Y)#[9, 1673]
                Y = Y[-1, :]#[1673]
                #print(Y[0])
                Y_predict_res[:, i] = Y
                # print(Y_predict.shape)#(48,)
                #X_test = np.concatenate((X_test, Y.reshape(-1, 1)), axis=1)[:, 1:]
                X_test=X_test.squeeze(-1) #[1673, 9]
                X_test=torch.cat([X_test, Y.reshape(-1, 1)],dim=1)[:, 1:]
                #print(X_test.size()) [1673, 9]
                X_test=X_test.unsqueeze(-1)
                #print(X_test.size()) #[1673, 9, 1]
                #print(Y_predict_res[0])

#############################循环预测###############################################
            Y_predict_res=Y_predict_res.cpu()
            #print(Y_predict_res.shape)#torch.Size([1673, 3])
            print("预测的第一个样本值")
            print(Y_predict_res[0])
            #print(Y_test_real.shape)#torch.Size([1673, 3])
            print("真实的第一个样本值")
            print(Y_test_real[0])

            print("MSE")
            self.MSE_score = mean_squared_error(Y_test_real, Y_predict_res.numpy())
            print(self.MSE_score)
            self.RMSE_score = mean_squared_error(Y_test_real, Y_predict_res.numpy()) ** 0.5
            print("RMSE")
            print(self.RMSE_score)
            print("MAE")
            self.MAE = mean_absolute_error(Y_test_real, Y_predict_res.numpy())
            print(self.MAE)
            print("R2_score")  # 接近1最好
            self.R2_score = r2_score(Y_test_real, Y_predict_res.numpy())
            print(self.R2_score)

        self.way4adjustParameter_score = 'MSE'
        self.best_estimator = state
        self.best_params = res_ajustParameter
        self.res_ajustParameter = res_ajustParameter_data

        try:
            end = time.clock()
        except:
            end = time.perf_counter()
        self.use_time=(end-start)/60

        writer.close()
    def do_savedate(self):
        save_csv = use_save_csv_DL.save_csv(num_data=self.num_data,
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
                                         csv_file=self.csv_file,
                                         use_time=self.use_time,
                                         opt=self.opt,
                                         total_epochs=self.total_epochs)
        df_savecsv = save_csv.do_savecsv()


        return df_savecsv
