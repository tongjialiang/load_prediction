import pickle
import numpy as np

#读
with open('D:\\实验记录\\pk\\busid_RTTS_and_busid_STTS_所有长度序列.pk', 'rb') as f:
    data = pickle.load(f)

#计算RT汇总序列做归一化
for i in data[0]:
    data_temp=data[0][i]
    #print(data_temp)
    data_temp=(data_temp-142.31591053144913)/435.8846214390647#部分数据 data_temp=(data_temp-134.8640)/425.89807
    data[0][i]=data_temp
    print(data[0][i])

#计算ST汇总序列做归一化
for i in data[1]:
    data_temp=data[1][i]
    #print(data_temp)
    data_temp=(data_temp-242.33768321389218)/576.3043046274704 #部分数据data_temp=(data_temp-141.96173)/409.27771
    data[1][i]=data_temp
    print(data[1][i])

with open('D:\\实验记录\\pk\\busid_RTTS_and_busid_STTS_Norm_所有长度序列.pk', 'wb+') as f:
    pickle.dump(data, f)
f.close()
