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


x=np.linspace(-3,3,100)#X的范围位于【-3，3】
y=0.5*np.sin(2*np.pi*x)
z=0.5*np.sin(2*np.pi*x+np.pi)
print(sum((y-z)*(y-z))/len(x))
#plt.title(r'欧氏距离=0.495')#这是latex的表达式，与matlplotlib兼容
#plt.plot(x,y,'ro')
sns.lineplot(x,y,label='用电企业1')
sns.lineplot(x,z,label='用电企业2')
plt.xlabel("时间")
plt.ylabel("负荷值")
plt.legend()

plt.savefig("C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\欧氏距离局限性1",dpi=1000)
plt.show()#展示图象
######################################################################
x=0.5*np.linspace(-3,3,100)#X的范围位于【-3，3】
y=0.5*np.sin(2*np.pi*x)
z=y.copy()
z[:]=0
#z[int(len(y)/2):]=y[int(len(y)/2):]/3
print(sum((y-z)*(y-z))/len(x))
#plt.title(r'欧氏距离=0.1')#这是latex的表达式，与matlplotlib兼容
#plt.plot(x,y,'ro')
sns.lineplot(x,y,label='用电企业1')
sns.lineplot(x,z,label='用电企业2')
plt.xlabel("时间")
plt.ylabel("负荷值")
plt.legend()

plt.savefig("C:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第三章数据集划分方式的研究\\欧氏距离局限性2",dpi=1000)
plt.show()#展示图象