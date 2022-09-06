import os
import pickle
import sys
import platform
import numpy as np
import numpy.fft as nf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
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
with open('C:\\实验记录\\重要结果文件\\pk\\busid_RTTS_and_busid_STTS_Norm_所有长度序列.pk', 'rb') as f:
    data = pickle.load(f)
f.close()
data_res=[{},{}]
#RT数据
for index,i in enumerate(data[0]):
    ts=data[0][i]       #原始序列
    ts_fft = nf.fft(ts) #傅里叶变化后的序列
    #freqs = nf.fftfreq(len(ts), d=1)  #频率序列
    print(ts_fft)
    mask=np.abs(ts).copy()*0
    #mask[::21] = 1 #日周期分量
    mask[::3] = 1 #周周期分量
    print(ts_fft*mask)
    ts_ifft=np.fft.ifft(ts_fft*mask).real
    print(ts_ifft[:200])
    data_res[0][i]=ts_ifft
    if index==5:
        sns.lineplot(data=ts_ifft[:5000], label='日周期分量')
        #plt.plot(ts_ifft[:5000], label='降噪后')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        #plt.title("日周期分量")
        plt.xlabel("时间")
        plt.ylabel("电力负荷(归一化)")
        plt.legend()
        #plt.savefig("C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\某用电公司的日周期分量", dpi=1000)
        plt.show()
    if index == 5:
        sns.lineplot(data=ts_ifft[:5000], label='周周期分量')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        #plt.title("某用电公司降噪前的时域图")
        plt.xlabel("时间")
        plt.ylabel("电力负荷(归一化)")
        plt.legend()
        plt.savefig("C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\某用电公司的周周期分量", dpi=1000)
        plt.show()
        #plt.savefig('D:\\实验记录\\画图\\某企业周周期分量.png')
        break

#ST数据
# for i in data[1]:
#     ts=data[1][i]       #原始序列
#     ts_fft = nf.fft(ts) #傅里叶变化后的序列
#     #freqs = nf.fftfreq(len(ts), d=1)  #频率序列
#     #print(ts_fft)
#     mask=np.abs(ts).copy()*0
#     mask[::24] = 1 #周周期分量
#     #print(ts_fft*mask)
#     ts_ifft=np.fft.ifft(ts_fft*mask).real
#     print(ts_ifft[:200])
#     data_res[1][i]=ts_ifft

# with open('D:\\实验记录\\pk\\busid_RTTS_and_busid_STTS_Norm_所有长度序列_Fourier.pk', 'wb+') as f:
#     pickle.dump(data_res, f)
# f.close()