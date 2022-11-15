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
warnings.filterwarnings("ignore")
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

#dir="D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第8章基于改进Transformer模型的负荷预测方法\\初步结果(调参以前)"
dir="d:\\实验记录\\实验结果\\Xgboost_v3"
rescsv=''
resdict={}
for root, dirs, filelist in os.walk(dir):
        for i in filelist:
            #处理CSV文件
            if i.endswith("csv"):
                print(i)
                tempcsv = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                if len(rescsv)==0:
                    rescsv=tempcsv
                    continue
                else:
                    rescsv=rescsv.append(tempcsv)
                print(len(rescsv))
            if i.endswith("pk"):
                print(root+ "\\" + i)
                with open(root + "\\" + i, 'rb') as f:
                    data = pickle.load(f)
                    #print(data)
                    resdict.update(data)

rescsv.to_csv(path_or_buf=dir+"\\"+dir.split("\\")[-1]+".csv", encoding="utf_8_sig", index=False)

with open(dir+"\\"+dir.split("\\")[-1]+".pk", 'wb+') as f:
    pickle.dump(resdict, f)
f.close()


# with open('D:\\实验记录\\实验结果\\Ridge\\Ridge.pk', 'rb') as f:
#     data = pickle.load(f)
#     print(len(data))
#     print(data["按行业划分数据集_方案1-汇总标准化_批发和零售业_ST_data_30_1.pk"])