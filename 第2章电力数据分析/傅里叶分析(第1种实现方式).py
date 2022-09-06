import os
import pickle
import sys
import platform
import numpy as np
import numpy.fft as nf
import matplotlib
import matplotlib.pyplot as plt

with open('D:\\实验记录\\重要结果文件\\pk\\busid_RTTS_and_busid_STTS_Norm_所有长度序列.pk', 'rb') as f:
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
    #print(ts_fft*mask)
    ts_ifft=np.fft.ifft(ts_fft*mask).real
    print(ts_ifft[:200])
    data_res[0][i]=ts_ifft
    if index==6:
        plt.plot(ts_ifft[:5000], label='周周期分量')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.title("某用电公司的周周期分量")
        plt.xlabel("时间")
        plt.ylabel("电力负荷(归一化)")
        plt.legend()
        #plt.show()
        plt.savefig('D:\\实验记录\\画图\\某企业周周期分量.png')
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