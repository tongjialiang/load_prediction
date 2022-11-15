import pickle
import numpy as np

#读
with open('D:\\实验记录\\pk\\busid_RTTS_and_busid_STTS_所有长度序列.pk', 'rb') as f:
    data = pickle.load(f)
#计算rt均值和方差
rt_length=0
rt_sum=0

#计算RT序列的均值和标准差
res_rt=[]#把所有rt ts拼接成一个
for i in data[0]:
    data_temp=data[0][i]
    if len(res_rt)==0:
        res_rt=data_temp
    res_rt=np.hstack((res_rt, data_temp))
    print(res_rt.shape)

    # rt_length += len(data_temp)
    # rt_sum+=sum(data_temp)

#计算ST序列的均值和标准差
res_st=[]#把所有rt ts拼接成一个
for i in data[1]:
    data_temp=data[1][i]
    if len(res_st)==0:
        res_st=data_temp
    res_st=np.hstack((res_st, data_temp))
    print(res_st.shape)

print("打印rt序列的shape、均值和标准差")
print(res_rt.shape)
print(np.mean(res_rt))
print(np.std(res_rt))

print("打印st序列的shape、均值和标准差")
print(res_st.shape)
print(np.mean(res_st))
print(np.std(res_st))
    # rt_length += len(data_temp)
    # rt_sum+=sum(data_temp)

