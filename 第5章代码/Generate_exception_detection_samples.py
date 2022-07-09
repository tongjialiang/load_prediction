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
import GetClusteringXandBusname_rt
import GetClusteringXandBusname_long
x_rt,businessname_rt=GetClusteringXandBusname_rt.do_GetClusteringXandBusname_rt()
x_long,businessname_long=GetClusteringXandBusname_long.do_GetClusteringXandBusname_long()

#print(x_rt[:50])
#print(businessname_rt[:50])
#实时数据大于1000的公司
#print([np.where(x_rt[:,0]>1000)])
x_rt_exception=x_rt
businessname_rt_exception=businessname_rt
x_long_exception=x_long
businessname_long_exception=businessname_long
#注入异常数据
#
#print(x_rt[5]*100)
x_rt_exception=np.append(x_rt_exception,[x_rt[5]*100],axis=0)
businessname_rt_exception=np.append(businessname_rt_exception,"rt异常数据_1")
x_rt_exception=np.append(x_rt_exception,[x_rt[74]*200],axis=0)
businessname_rt_exception=np.append(businessname_rt_exception,"rt异常数据_2")
x_rt_exception=np.append(x_rt_exception,[x_rt[75]*300],axis=0)
businessname_rt_exception=np.append(businessname_rt_exception,"rt异常数据_3")
x_rt_exception=np.append(x_rt_exception,[x_rt[106]*400],axis=0)
businessname_rt_exception=np.append(businessname_rt_exception,"rt异常数据_4")
x_rt_exception=np.append(x_rt_exception,[x_rt[411]*500],axis=0)
businessname_rt_exception=np.append(businessname_rt_exception,"rt异常数据_5")
x_rt_exception=np.append(x_rt_exception,[x_rt[534]*[500,1]],axis=0)
businessname_rt_exception=np.append(businessname_rt_exception,"rt异常数据_6")
x_rt_exception=np.append(x_rt_exception,[x_rt[598]*[600,1]],axis=0)
businessname_rt_exception=np.append(businessname_rt_exception,"rt异常数据_7")
x_rt_exception=np.append(x_rt_exception,[x_rt[599]*[700,1]],axis=0)
businessname_rt_exception=np.append(businessname_rt_exception,"rt异常数据_8")
x_rt_exception=np.append(x_rt_exception,[x_rt[600]*[1,500]],axis=0)
businessname_rt_exception=np.append(businessname_rt_exception,"rt异常数据_9")
x_rt_exception=np.append(x_rt_exception,[x_rt[601]*[1,600]],axis=0)
businessname_rt_exception=np.append(businessname_rt_exception,"rt异常数据_10")
print(x_rt_exception[-15:])
print(businessname_rt_exception[-15:])
print(x_rt_exception.shape)
print(x_rt.shape)
print(businessname_rt_exception.shape)
print(businessname_rt.shape)

#对长期用电数据注入异常数据
print([np.where(x_long[:,0]>1000)])
x_long_exception=np.append(x_long_exception,[x_long[1233]*100],axis=0)
businessname_long_exception=np.append(businessname_long_exception,"long异常数据_1")
x_long_exception=np.append(x_long_exception,[x_long[1242]*200],axis=0)
businessname_long_exception=np.append(businessname_long_exception,"long异常数据_2")
x_long_exception=np.append(x_long_exception,[x_long[1361]*300],axis=0)
businessname_long_exception=np.append(businessname_long_exception,"long异常数据_3")
x_long_exception=np.append(x_long_exception,[x_long[1364]*400],axis=0)
businessname_long_exception=np.append(businessname_long_exception,"long异常数据_4")
x_long_exception=np.append(x_long_exception,[x_long[1373]*500],axis=0)
businessname_long_exception=np.append(businessname_long_exception,"long异常数据_5")
x_long_exception=np.append(x_long_exception,[x_long[1535]*[500,1]],axis=0)
businessname_long_exception=np.append(businessname_long_exception,"long异常数据_6")
x_long_exception=np.append(x_long_exception,[x_long[1564]*[600,1]],axis=0)
businessname_long_exception=np.append(businessname_long_exception,"long异常数据_7")
x_long_exception=np.append(x_long_exception,[x_long[1608]*[700,1]],axis=0)
businessname_long_exception=np.append(businessname_long_exception,"long异常数据_8")
x_long_exception=np.append(x_long_exception,[x_long[1870]*[1,500]],axis=0)
businessname_long_exception=np.append(businessname_long_exception,"long异常数据_9")
x_long_exception=np.append(x_long_exception,[x_long[1920]*[1,600]],axis=0)
businessname_long_exception=np.append(businessname_long_exception,"long异常数据_10")
print(x_long_exception[-12:])
print(businessname_long_exception[-12:])
print(x_long_exception.shape)
print(x_long.shape)
print(businessname_long_exception.shape)
print(businessname_long.shape)


#持有化
res={"rt":[x_rt,businessname_rt], #实时数据、公司名
     "long":[x_long,businessname_long],#长期数据、公司名
     "rt_exception":[x_rt_exception,businessname_rt_exception],#实时数据(带异常数据)、公司名
     "long_exception":[x_long_exception,businessname_long_exception]}#长期数据(带异常数据)、公司名
with open('d:/exception_detection_samples.pk', 'wb+') as f:
    pickle.dump(res, f)
f.close()
