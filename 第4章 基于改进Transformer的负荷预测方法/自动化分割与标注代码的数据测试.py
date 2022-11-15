import os
import pickle

import numpy as np
for root, dirs, filelist in os.walk("D:\\数据采样完成new\\"):
        for i in filelist:
            #filename=root+"\\"+i
            #print(filename)
            if i.endswith("pk"):
                with open(root + i, 'rb') as f:
                    data = pickle.load(f)
                print(i)
                print(len(data[0]))
#读


