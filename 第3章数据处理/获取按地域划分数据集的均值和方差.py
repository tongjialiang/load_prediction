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
import time
my_dir='D:\\用电数据集\\归一化之前的数据集\\按地域划分数据集V2'
the_list=os.listdir(my_dir)
#print(the_list)#['rtclass_1', 'rtclass_10', 'rtclass_11'...

def standard_do():
    flag1=0
    flag2=0
    flag3=0
    res={}
    rtdata=''
    mtdata=''
    stdata=''
    #myclass =["class15"]
    myclass = the_list
    area=["杭州市区","淳安县","富阳区","海宁市","海盐县","嘉善县",
    "嘉兴市桐乡市","建德市","临安区","平湖市","萧山区","余杭区"]
    # area = ["杭州市区","嘉兴市桐乡市"]
    bus_type=["采矿业","电力热力燃气及水生产和供应业","金融业与房地产业与租赁和商务服务业","建筑业","交通运输仓储和邮政业",
    "教育","居民服务修理和其他服务业","科学研究和技术服务业与信息传输软件与信息技术服务业","农林牧渔业","批发和零售业"
        ,"水利环境和公共设施管理业","卫生和社会工作与公共管理社会保障和社会组织","文化体育和娱乐业"
        ,"制造业_机械电子制造业","制造业_轻纺工业","制造业_资源加工工业","住宿和餐饮业"]
    # bus_type = ["居民服务修理和其他服务业","住宿和餐饮业"]
    #赋初值
    # for i in area:
    #     exec("rtdata_"+ i +"=''")
    #     exec("mtdata_" + i + "=''")
    #     exec("stdata_" + i + "=''")

    # #计算所有数据的均值和方差，保存在字典里
    def get_res(rtdata,mtdata,stdata):
        res_temp={}
        rt_mean_ssyg = 0
        rt_std_ssyg = 0
        rt_mean_pjqw = 0
        rt_std_pjqw = 0
        rt_mean_29 = 0
        rt_std_29 = 0
        rt_mean_18 = 0
        rt_std_18 = 0
        rt_mean_163 = 0
        rt_std_163 = 0
        rt_mean_305 = 0
        rt_std_305 = 0
        rt_mean_35 = 0
        rt_std_35 = 0
        rt_mean_46 = 0
        rt_std_46 = 0
        rt_mean_72 = 0
        rt_std_72 = 0
        rt_mean_518 = 0
        rt_std_518 = 0
        rt_mean_529 = 0
        rt_std_529 = 0
        rt_mean_537_1 = 0
        rt_std_537_1 = 0
        rt_mean_537_2 = 0
        rt_std_537_2 = 0
        rt_mean_537_3 = 0
        rt_std_537_3 = 0
        mt_mean_pjfh = 0
        mt_std_pjfh = 0
        mt_mean_sdrl = 0
        mt_std_sdrl = 0
        mt_mean_qsnyscl = 0
        mt_std_qsnyscl = 0
        mt_mean_qsdlscl = 0
        mt_std_qsdlscl = 0
        mt_mean_nyscbsnzz = 0
        mt_std_nyscbsnzz = 0
        mt_mean_qsnyxfl = 0
        mt_std_qsnyxfl = 0
        mt_mean_qsdlxfl = 0
        mt_std_qsdlxfl = 0
        mt_mean_29 = 0
        mt_std_29 = 0
        mt_mean_18 = 0
        mt_std_18 = 0
        mt_mean_163 = 0
        mt_std_163 = 0
        mt_mean_305 = 0
        mt_std_305 = 0
        mt_mean_35 = 0
        mt_std_35 = 0
        mt_mean_46 = 0
        mt_std_46 = 0
        mt_mean_72 = 0
        mt_std_72 = 0
        mt_mean_518 = 0
        mt_std_518 = 0
        mt_mean_529 = 0
        mt_std_529 = 0
        mt_mean_537_1 = 0
        mt_std_537_1 = 0
        mt_mean_537_2 = 0
        mt_std_537_2 = 0
        mt_mean_537_3 = 0
        mt_std_537_3 = 0
        st_mean_pjfh = 0
        st_std_pjfh = 0
        st_mean_sdrl = 0
        st_std_sdrl = 0
        st_mean_pjqw = 0
        st_std_pjqw = 0
        st_mean_qsnyscl = 0
        st_std_qsnyscl = 0
        st_mean_qsdlscl = 0
        st_std_qsdlscl = 0
        st_mean_nyscbsnzz = 0
        st_std_nyscbsnzz = 0
        st_mean_qsnyxfl = 0
        st_std_qsnyxfl = 0
        st_mean_qsdlxfl = 0
        st_std_qsdlxfl = 0
        st_mean_29 = 0
        st_std_29 = 0
        st_mean_18 = 0
        st_std_18 = 0
        st_mean_163 = 0
        st_std_163 = 0
        st_mean_305 = 0
        st_std_305 = 0
        st_mean_35 = 0
        st_std_35 = 0
        st_mean_46 = 0
        st_std_46 = 0
        st_mean_72 = 0
        st_std_72 = 0
        st_mean_518 = 0
        st_std_518 = 0
        st_mean_529 = 0
        st_std_529 = 0
        st_mean_537_1 = 0
        st_std_537_1 = 0
        st_mean_537_2 = 0
        st_std_537_2 = 0
        st_mean_537_3 = 0
        st_std_537_3 = 0
        if len(rtdata)!=0:
            if rtdata.columns.__contains__('瞬时有功(kW)'):
                rt_mean_ssyg=rtdata.loc[:,'瞬时有功(kW)'].mean()
                rt_std_ssyg=rtdata.loc[:,'瞬时有功(kW)'].std()
                res_temp.update({'rt_mean_ssyg': rt_mean_ssyg, 'rt_std_ssyg': rt_std_ssyg})
            if rtdata.columns.__contains__('平均气温'):
                rt_mean_pjqw=rtdata.loc[:,'平均气温'].mean()
                rt_std_pjqw=rtdata.loc[:,'平均气温'].std()
                res_temp.update({'rt_mean_pjqw': rt_mean_pjqw, 'rt_std_pjqw': rt_std_pjqw})
            if rtdata.columns.__contains__('按行业分的法人单位数_p29'):
                rt_mean_29 = rtdata.loc[:, '按行业分的法人单位数_p29'].mean()
                rt_std_29 = rtdata.loc[:, '按行业分的法人单位数_p29'].std()
                res_temp.update({'rt_mean_29':rt_mean_29, 'rt_std_29':rt_std_29})
            if rtdata.columns.__contains__('按行业分的全省生产总值_亿元_p18'):
                rt_mean_18 = rtdata.loc[:, '按行业分的全省生产总值_亿元_p18'].mean()
                rt_std_18 = rtdata.loc[:, '按行业分的全省生产总值_亿元_p18'].std()
                res_temp.update({'rt_mean_18':rt_mean_18, 'rt_std_18':rt_std_18})
            if rtdata.columns.__contains__('分行业全社会单位就业人员年平均工资_p163'):
                rt_mean_163 = rtdata.loc[:, '分行业全社会单位就业人员年平均工资_p163'].mean()
                rt_std_163 = rtdata.loc[:, '分行业全社会单位就业人员年平均工资_p163'].std()
                res_temp.update({'rt_mean_163':rt_mean_163, 'rt_std_163':rt_std_163})
            if rtdata.columns.__contains__('按行业分全社会用电情况_亿千瓦时_p305'):
                rt_mean_305 = rtdata.loc[:, '按行业分全社会用电情况_亿千瓦时_p305'].mean()
                rt_std_305 = rtdata.loc[:, '按行业分全社会用电情况_亿千瓦时_p305'].std()
                res_temp.update({'rt_mean_305':rt_mean_305, 'rt_std_305':rt_std_305})
            if rtdata.columns.__contains__('按地区分组的法人单位数(万人)_p35'):
                rt_mean_35 = rtdata.loc[:, '按地区分组的法人单位数(万人)_p35'].mean()
                rt_std_35 = rtdata.loc[:, '按地区分组的法人单位数(万人)_p35'].std()
                res_temp.update({'rt_mean_35':rt_mean_35, 'rt_std_35':rt_std_35})
            if rtdata.columns.__contains__('总人口数(万人)_p46'):
                rt_mean_46 = rtdata.loc[:, '总人口数(万人)_p46'].mean()
                rt_std_46 = rtdata.loc[:, '总人口数(万人)_p46'].std()
                res_temp.update({'rt_mean_46':rt_mean_46, 'rt_std_46':rt_std_46})
            if rtdata.columns.__contains__('各市规模以上企业年末单位就业人员(万人)_p72'):
                rt_mean_72 = rtdata.loc[:, '各市规模以上企业年末单位就业人员(万人)_p72'].mean()
                rt_std_72 = rtdata.loc[:, '各市规模以上企业年末单位就业人员(万人)_p72'].std()
                res_temp.update({'rt_mean_72':rt_mean_72, 'rt_std_72':rt_std_72})
            if rtdata.columns.__contains__('生产总值(百亿元)_p518_p539'):
                rt_mean_518 = rtdata.loc[:, '生产总值(百亿元)_p518_p539'].mean()
                rt_std_518 = rtdata.loc[:, '生产总值(百亿元)_p518_p539'].std()
                res_temp.update({'rt_mean_518':rt_mean_518, 'rt_std_518':rt_std_518})
            if rtdata.columns.__contains__('全年用电量_百亿千瓦时_p529_p571'):
                rt_mean_529 = rtdata.loc[:, '全年用电量_百亿千瓦时_p529_p571'].mean()
                rt_std_529 = rtdata.loc[:, '全年用电量_百亿千瓦时_p529_p571'].std()
                res_temp.update({'rt_mean_529':rt_mean_529, 'rt_std_529':rt_std_529})
            if rtdata.columns.__contains__('第一产业(百亿元)_p537_p539'):
                rt_mean_537_1 = rtdata.loc[:, '第一产业(百亿元)_p537_p539'].mean()
                rt_std_537_1 = rtdata.loc[:, '第一产业(百亿元)_p537_p539'].std()
                res_temp.update({'rt_mean_537_1':rt_mean_537_1, 'rt_std_537_1':rt_std_537_1})
            if rtdata.columns.__contains__('第二产业(百亿元)_p537_p539'):
                rt_mean_537_2 = rtdata.loc[:, '第二产业(百亿元)_p537_p539'].mean()
                rt_std_537_2 = rtdata.loc[:, '第二产业(百亿元)_p537_p539'].std()
                res_temp.update({'rt_mean_537_2':rt_mean_537_2, 'rt_std_537_2':rt_std_537_2})
            if rtdata.columns.__contains__('第三产业(百亿元)_p537_p539'):
                rt_mean_537_3 = rtdata.loc[:, '第三产业(百亿元)_p537_p539'].mean()
                rt_std_537_3 = rtdata.loc[:, '第三产业(百亿元)_p537_p539'].std()
                res_temp.update({'rt_mean_537_3':rt_mean_537_3, 'rt_std_537_3':rt_std_537_3})
        if len(mtdata)!=0:
            if mtdata.columns.__contains__('平均负荷(kW)'):
                mt_mean_pjfh = mtdata.loc[:,'平均负荷(kW)'].mean()
                mt_std_pjfh = mtdata.loc[:,'平均负荷(kW)'].std()
                res_temp.update({'mt_mean_pjfh':mt_mean_pjfh, 'mt_std_pjfh':mt_std_pjfh})
            if mtdata.columns.__contains__('受电容量(KVA)'):
                mt_mean_sdrl = mtdata.loc[:,'受电容量(KVA)'].mean()
                mt_std_sdrl = mtdata.loc[:,'受电容量(KVA)'].std()
                res_temp.update({'mt_mean_sdrl':mt_mean_sdrl, 'mt_std_sdrl':mt_std_sdrl})
            if mtdata.columns.__contains__('全省能源生产量(百万吨标准煤)'):
                mt_mean_qsnyscl = mtdata.loc[:,'全省能源生产量(百万吨标准煤)'].mean()
                mt_std_qsnyscl = mtdata.loc[:,'全省能源生产量(百万吨标准煤)'].std()
                res_temp.update({'mt_mean_qsnyscl':mt_mean_qsnyscl, 'mt_std_qsnyscl':mt_std_qsnyscl})
            if mtdata.columns.__contains__('全省电力生产量(百亿千瓦小时)'):
                mt_mean_qsdlscl = mtdata.loc[:,'全省电力生产量(百亿千瓦小时)'].mean()
                mt_std_qsdlscl = mtdata.loc[:,'全省电力生产量(百亿千瓦小时)'].std()
                res_temp.update({'mt_mean_qsdlscl':mt_mean_qsdlscl, 'mt_std_qsdlscl':mt_std_qsdlscl})
            if mtdata.columns.__contains__('能源生产比上年增长(%)'):
                mt_mean_nyscbsnzz = mtdata.loc[:,'能源生产比上年增长(%)'].mean()
                mt_std_nyscbsnzz = mtdata.loc[:,'能源生产比上年增长(%)'].std()
                res_temp.update({'mt_mean_nyscbsnzz':mt_mean_nyscbsnzz, 'mt_std_nyscbsnzz':mt_std_nyscbsnzz})
            if mtdata.columns.__contains__('全省能源消费量(百万吨标准煤)'):
                mt_mean_qsnyxfl = mtdata.loc[:,'全省能源消费量(百万吨标准煤)'].mean()
                mt_std_qsnyxfl = mtdata.loc[:,'全省能源消费量(百万吨标准煤)'].std()
                res_temp.update({'mt_mean_qsnyxfl':mt_mean_qsnyxfl, 'mt_std_qsnyxfl':mt_std_qsnyxfl})
            if mtdata.columns.__contains__('全省电力消费量(百亿千瓦小时)'):
                mt_mean_qsdlxfl = mtdata.loc[:,'全省电力消费量(百亿千瓦小时)'].mean()
                mt_std_qsdlxfl = mtdata.loc[:,'全省电力消费量(百亿千瓦小时)'].std()
                res_temp.update({'mt_mean_qsdlxfl':mt_mean_qsdlxfl, 'mt_std_qsdlxfl':mt_std_qsdlxfl})
            if mtdata.columns.__contains__('按行业分的法人单位数_p29'):
                mt_mean_29 = mtdata.loc[:,'按行业分的法人单位数_p29'].mean()
                mt_std_29 = mtdata.loc[:,'按行业分的法人单位数_p29'].std()
                res_temp.update({'mt_mean_29':mt_mean_29, 'mt_std_29':mt_std_29})
            if mtdata.columns.__contains__('按行业分的全省生产总值_亿元_p18'):
                mt_mean_18 = mtdata.loc[:,'按行业分的全省生产总值_亿元_p18'].mean()
                mt_std_18 = mtdata.loc[:,'按行业分的全省生产总值_亿元_p18'].std()
                res_temp.update({'mt_mean_18':mt_mean_18, 'mt_std_18':mt_std_18})
            if mtdata.columns.__contains__('分行业全社会单位就业人员年平均工资_p163'):
                mt_mean_163 = mtdata.loc[:,'分行业全社会单位就业人员年平均工资_p163'].mean()
                mt_std_163 = mtdata.loc[:,'分行业全社会单位就业人员年平均工资_p163'].std()
                res_temp.update({'mt_mean_163':mt_mean_163, 'mt_std_163':mt_std_163})
            if mtdata.columns.__contains__('按行业分全社会用电情况_亿千瓦时_p305'):
                mt_mean_305 = mtdata.loc[:,'按行业分全社会用电情况_亿千瓦时_p305'].mean()
                mt_std_305 = mtdata.loc[:,'按行业分全社会用电情况_亿千瓦时_p305'].std()
                res_temp.update({'mt_mean_305':mt_mean_305, 'mt_std_305':mt_std_305})
            if mtdata.columns.__contains__('按地区分组的法人单位数(万人)_p35'):
                mt_mean_35 = mtdata.loc[:,'按地区分组的法人单位数(万人)_p35'].mean()
                mt_std_35 = mtdata.loc[:,'按地区分组的法人单位数(万人)_p35'].std()
                res_temp.update({'mt_mean_35':mt_mean_35, 'mt_std_35':mt_std_35})
            if mtdata.columns.__contains__('总人口数(万人)_p46'):
                mt_mean_46 = mtdata.loc[:,'总人口数(万人)_p46'].mean()
                mt_std_46 = mtdata.loc[:,'总人口数(万人)_p46'].std()
                res_temp.update({'mt_mean_46':mt_mean_46, 'mt_std_46':mt_std_46})
            if mtdata.columns.__contains__('各市规模以上企业年末单位就业人员(万人)_p72'):
                mt_mean_72 = mtdata.loc[:,'各市规模以上企业年末单位就业人员(万人)_p72'].mean()
                mt_std_72 = mtdata.loc[:,'各市规模以上企业年末单位就业人员(万人)_p72'].std()
                res_temp.update({'mt_mean_72':mt_mean_72, 'mt_std_72':mt_std_72})
            if mtdata.columns.__contains__('生产总值(百亿元)_p518_p539'):
                mt_mean_518 = mtdata.loc[:, '生产总值(百亿元)_p518_p539'].mean()
                mt_std_518 = mtdata.loc[:, '生产总值(百亿元)_p518_p539'].std()
                res_temp.update({'mt_mean_518':mt_mean_518, 'mt_std_518':mt_std_518})
            if mtdata.columns.__contains__('全年用电量_百亿千瓦时_p529_p571'):
                mt_mean_529 = mtdata.loc[:, '全年用电量_百亿千瓦时_p529_p571'].mean()
                mt_std_529 = mtdata.loc[:, '全年用电量_百亿千瓦时_p529_p571'].std()
                res_temp.update({'mt_mean_529':mt_mean_529, 'mt_std_529':mt_std_529})
            if mtdata.columns.__contains__('第一产业(百亿元)_p537_p539'):
                mt_mean_537_1 = mtdata.loc[:, '第一产业(百亿元)_p537_p539'].mean()
                mt_std_537_1 = mtdata.loc[:, '第一产业(百亿元)_p537_p539'].std()
                res_temp.update({'mt_mean_537_1':mt_mean_537_1, 'mt_std_537_1':mt_std_537_1})
            if mtdata.columns.__contains__('第二产业(百亿元)_p537_p539'):
                mt_mean_537_2 = mtdata.loc[:, '第二产业(百亿元)_p537_p539'].mean()
                mt_std_537_2 = mtdata.loc[:, '第二产业(百亿元)_p537_p539'].std()
                res_temp.update({'mt_mean_537_2':mt_mean_537_2, 'mt_std_537_2':mt_std_537_2})
            if mtdata.columns.__contains__('第三产业(百亿元)_p537_p539'):
                mt_mean_537_3 = mtdata.loc[:, '第三产业(百亿元)_p537_p539'].mean()
                mt_std_537_3 = mtdata.loc[:, '第三产业(百亿元)_p537_p539'].std()
                res_temp.update({'mt_mean_537_3':mt_mean_537_3, 'mt_std_537_3':mt_std_537_3})

        if len(stdata)!=0:
            if stdata.columns.__contains__('平均负荷(kW)'):
                st_mean_pjfh = stdata.loc[:, '平均负荷(kW)'].mean()
                st_std_pjfh = stdata.loc[:, '平均负荷(kW)'].std()
                res_temp.update({'st_mean_pjfh':st_mean_pjfh, 'st_std_pjfh':st_std_pjfh})
            if stdata.columns.__contains__('受电容量(KVA)'):
                st_mean_sdrl = stdata.loc[:, '受电容量(KVA)'].mean()
                st_std_sdrl = stdata.loc[:, '受电容量(KVA)'].std()
                res_temp.update({'st_mean_sdrl':st_mean_sdrl, 'st_std_sdrl':st_std_sdrl})
            if stdata.columns.__contains__('平均气温'):
                st_mean_pjqw = stdata.loc[:, '平均气温'].mean()
                st_std_pjqw = stdata.loc[:, '平均气温'].std()
                res_temp.update({'st_mean_pjqw':st_mean_pjqw, 'st_std_pjqw':st_std_pjqw})
            if stdata.columns.__contains__('全省能源生产量(百万吨标准煤)'):
                st_mean_qsnyscl = stdata.loc[:, '全省能源生产量(百万吨标准煤)'].mean()
                st_std_qsnyscl = stdata.loc[:, '全省能源生产量(百万吨标准煤)'].std()
                res_temp.update({'st_mean_qsnyscl':st_mean_qsnyscl, 'st_std_qsnyscl':st_std_qsnyscl})
            if stdata.columns.__contains__('全省电力生产量(百亿千瓦小时)'):
                st_mean_qsdlscl = stdata.loc[:, '全省电力生产量(百亿千瓦小时)'].mean()
                st_std_qsdlscl = stdata.loc[:, '全省电力生产量(百亿千瓦小时)'].std()
                res_temp.update({'st_mean_qsdlscl':st_mean_qsdlscl, 'st_std_qsdlscl':st_std_qsdlscl})
            if stdata.columns.__contains__('能源生产比上年增长(%)'):
                st_mean_nyscbsnzz = stdata.loc[:, '能源生产比上年增长(%)'].mean()
                st_std_nyscbsnzz = stdata.loc[:, '能源生产比上年增长(%)'].std()
                res_temp.update({'st_mean_nyscbsnzz':st_mean_nyscbsnzz, 'st_std_nyscbsnzz':st_std_nyscbsnzz})
            if stdata.columns.__contains__('全省能源消费量(百万吨标准煤)'):
                st_mean_qsnyxfl = stdata.loc[:, '全省能源消费量(百万吨标准煤)'].mean()
                st_std_qsnyxfl = stdata.loc[:, '全省能源消费量(百万吨标准煤)'].std()
                res_temp.update({'st_mean_qsnyxfl':st_mean_qsnyxfl, 'st_std_qsnyxfl':st_std_qsnyxfl})
            if stdata.columns.__contains__('全省电力消费量(百亿千瓦小时)'):
                st_mean_qsdlxfl = stdata.loc[:, '全省电力消费量(百亿千瓦小时)'].mean()
                st_std_qsdlxfl = stdata.loc[:, '全省电力消费量(百亿千瓦小时)'].std()
                res_temp.update({'st_mean_qsdlxfl':st_mean_qsdlxfl, 'st_std_qsdlxfl':st_std_qsdlxfl})
            if stdata.columns.__contains__('按行业分的法人单位数_p29'):
                st_mean_29 = stdata.loc[:, '按行业分的法人单位数_p29'].mean()
                st_std_29 = stdata.loc[:, '按行业分的法人单位数_p29'].std()
                res_temp.update({'st_mean_29':st_mean_29, 'st_std_29':st_std_29})
            if stdata.columns.__contains__('按行业分的全省生产总值_亿元_p18'):
                st_mean_18 = stdata.loc[:, '按行业分的全省生产总值_亿元_p18'].mean()
                st_std_18 = stdata.loc[:, '按行业分的全省生产总值_亿元_p18'].std()
                res_temp.update({'st_mean_18':st_mean_18, 'st_std_18':st_std_18})
            if stdata.columns.__contains__('分行业全社会单位就业人员年平均工资_p163'):
                st_mean_163 = stdata.loc[:, '分行业全社会单位就业人员年平均工资_p163'].mean()
                st_std_163 = stdata.loc[:, '分行业全社会单位就业人员年平均工资_p163'].std()
                res_temp.update({'st_mean_163':st_mean_163, 'st_std_163':st_std_163})
            if stdata.columns.__contains__('按行业分全社会用电情况_亿千瓦时_p305'):
                st_mean_305 = stdata.loc[:, '按行业分全社会用电情况_亿千瓦时_p305'].mean()
                st_std_305 = stdata.loc[:, '按行业分全社会用电情况_亿千瓦时_p305'].std()
                res_temp.update({'st_mean_305':st_mean_305, 'st_std_305':st_std_305})
            if stdata.columns.__contains__('按地区分组的法人单位数(万人)_p35'):
                st_mean_35 = stdata.loc[:, '按地区分组的法人单位数(万人)_p35'].mean()
                st_std_35 = stdata.loc[:, '按地区分组的法人单位数(万人)_p35'].std()
                res_temp.update({'st_mean_35':st_mean_35, 'st_std_35':st_std_35})
            if stdata.columns.__contains__('总人口数(万人)_p46'):
                st_mean_46 = stdata.loc[:, '总人口数(万人)_p46'].mean()
                st_std_46 = stdata.loc[:, '总人口数(万人)_p46'].std()
                res_temp.update({'st_mean_46':st_mean_46, 'st_std_46':st_std_46})
            if stdata.columns.__contains__('各市规模以上企业年末单位就业人员(万人)_p72'):
                st_mean_72 = stdata.loc[:, '各市规模以上企业年末单位就业人员(万人)_p72'].mean()
                st_std_72 = stdata.loc[:, '各市规模以上企业年末单位就业人员(万人)_p72'].std()
                res_temp.update({'st_mean_72':st_mean_72, 'st_std_72':st_std_72})
            if stdata.columns.__contains__('生产总值(百亿元)_p518_p539'):
                st_mean_518 = stdata.loc[:, '生产总值(百亿元)_p518_p539'].mean()
                st_std_518 = stdata.loc[:, '生产总值(百亿元)_p518_p539'].std()
                res_temp.update({'st_mean_518':st_mean_518, 'st_std_518':st_std_518})
            if stdata.columns.__contains__('全年用电量_百亿千瓦时_p529_p571'):
                st_mean_529 = stdata.loc[:, '全年用电量_百亿千瓦时_p529_p571'].mean()
                st_std_529 = stdata.loc[:, '全年用电量_百亿千瓦时_p529_p571'].std()
                res_temp.update({'st_mean_529':st_mean_529, 'st_std_529':st_std_529})
            if stdata.columns.__contains__('第一产业(百亿元)_p537_p539'):
                st_mean_537_1 = stdata.loc[:, '第一产业(百亿元)_p537_p539'].mean()
                st_std_537_1 = stdata.loc[:, '第一产业(百亿元)_p537_p539'].std()
                res_temp.update({'st_mean_537_1':st_mean_537_1, 'st_std_537_1':st_std_537_1})
            if stdata.columns.__contains__('第二产业(百亿元)_p537_p539'):
                st_mean_537_2 = stdata.loc[:, '第二产业(百亿元)_p537_p539'].mean()
                st_std_537_2 = stdata.loc[:, '第二产业(百亿元)_p537_p539'].std()
                res_temp.update({'st_mean_537_2':st_mean_537_2, 'st_std_537_2':st_std_537_2})
            if stdata.columns.__contains__('第三产业(百亿元)_p537_p539'):
                st_mean_537_3 = stdata.loc[:, '第三产业(百亿元)_p537_p539'].mean()
                st_std_537_3 = stdata.loc[:, '第三产业(百亿元)_p537_p539'].std()
                res_temp.update({'st_mean_537_3':st_mean_537_3, 'st_std_537_3':st_std_537_3})

        # res={"rt_mean":rt_mean,"rt_std":rt_std,"mt_mean":mt_mean,"mt_std":mt_std,"mt_mean_max":mt_mean_max,
        #      "mt_std_max":mt_std_max,"mt_mean_min":mt_mean_min,"mt_std_min":mt_std_min,"st_mean":st_mean,
        #      "st_std":st_std,"st_mean_max":st_mean_max,"st_std_max":st_std_max,"st_mean_min":st_mean_min
        #     ,"st_std_min":st_std_min}
        return res_temp

   # 按地域分类
    #把所有公司的df拼在一起
    for k in myclass:
        #print(k)
        for root, dirs, filelist in os.walk(my_dir+"\\"+k):
            for i in filelist:
                filename=root+"\\"+i
                if i =="RT_data.csv":
                    print(root+'\\'+i)
                    try:
                        df = pd.read_csv(filename, encoding='utf-8', sep=',')
                    except:
                        df = pd.read_csv(filename, encoding='gbk', sep=',')

                    if flag1==0:
                        rtdata=df

                        if len(rtdata)!=0:
                            flag1=1
                        continue
                    if len(df)==0:
                        continue
                    rtdata=pd.concat([rtdata, df], axis=0)
                if i =="MT_data.csv":
                    print(root + '\\' + i)
                    try:
                        df = pd.read_csv(filename, encoding='utf-8', sep=',')
                    except:
                        df = pd.read_csv(filename, encoding='gbk', sep=',')

                    if flag2==0:
                        mtdata=df

                        if len(mtdata)!=0:
                            flag2=1
                        continue
                    if len(df)==0:
                        continue
                    mtdata=pd.concat([mtdata, df], axis=0)
                if i =="ST_data.csv":
                    print(root + '\\' + i)
                    try:
                        df = pd.read_csv(filename, encoding='utf-8', sep=',')
                    except:
                        df = pd.read_csv(filename, encoding='gbk', sep=',')

                    if flag3==0:
                        stdata=df

                        if len(stdata)!=0:
                            flag3=1
                        continue
                    if len(df)==0:
                        continue
                    stdata=pd.concat([stdata, df], axis=0)
        #一个地区、一个行业、一个类别遍历完毕后，计算均值和方差
        # exec("rtdata_"+k+"=rtdata")
        # exec("mtdata_" + k + "=mtdata")
        # exec("stdata_" + k + "=stdata")
        #print(mtdata)
        res[k]=get_res(rtdata, mtdata, stdata)
        #print(k)
        #print(res[k])
        flag1=0
        flag2 =0
        flag3 =0
        rtdata=''
        stdata = ''
        mtdata = ''
    #print(res)
    return res




start =time.clock()
res=standard_do()
with open('D:\\实验记录\\重要结果文件\pk\\各特征的均值和方差_按地域划分数据集V2.pk', 'wb+') as f:
    pickle.dump(res, f)
f.close()

end =time.clock()
print('Running time: %s Minutes'%((end-start)/60))


