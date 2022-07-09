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
from ShowapiRequest import ShowapiRequest

myarea='上海'
df_res = pd.DataFrame() #存放天气信息的df

r = ShowapiRequest("http://route.showapi.com/9-7","909872","8fe1b4ced8c34a1c9cfad708fd4c74d6" )
#r.addBodyPara("areaCode", "110000")
#r.addBodyPara("areaid", "101010100")
# r.addBodyPara("startDate", "20160504")
# r.addBodyPara("endDate", "20160810")
r.addBodyPara("area", myarea)


for m in ['201601','201602','201603','201604','201605','201606','201607','201608','201609','201610','201611','201612',
          '201701','201702','201703','201704','201705','201706','201707','201708','201709','201710','201711','201712',
          '201801','201802','201803','201804','201805','201806','201807','201808','201809','201810','201811','201812',
          '201901','201902','201903','201904','201905','201906','201907','201908','201909','201910','201911','201912',
          '202001','202002','202003','202004','202005','202006','202007','202008','202009','202010','202011','202012',
          '202101','202102','202103','202104','202105','202106','202107','202108','202109','202110','202111','202112']:
    r.addBodyPara("month", m)
    res = r.post() #调用接口获取嘉兴、杭州地区天气数据json字符串
    time.sleep(1)
    res_json=json.loads(res.text) #json字符串转python的字典
    print(res_json)
    res_list=res_json["showapi_res_body"]["list"]#获取list,List中存放每一天的天气信息字典

#遍历list,把天气信息保存到df中

#print(type(res.text))
#print(res.text["showapi_res_body"]["list"]) # 返回信息
    for i in res_list:
        #print(i)
        #print(type(i))
        avg_temperature=(int(i["max_temperature"])+int(i["min_temperature"]))/2#求平均气温
        i.update({'平均气温':avg_temperature})
        #i.update({'地区': myarea})
        i.update({'weather':i['weather'].split('-')[-1]})
        df_res=df_res.append([i])
        #去掉不需要的字段
        del df_res['aqi']
        del df_res['aqiInfo']
        del df_res['aqiLevel']
        del df_res['wind_direction']
        del df_res['wind_power']


df_res.to_csv(path_or_buf='D:\\用电数据集\\上海天气数据.csv', encoding="utf_8_sig",index=False)
