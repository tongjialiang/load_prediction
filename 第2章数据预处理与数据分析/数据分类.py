#df.drop( index = df.age[df1.age == 0].index )
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)#把列显示全
region_name = "滨江供电分公司"
def do(region_name,dir3):
    min_time_link = ''
    day_time_link = ''
    flag=False#循环只执行一次的标记

    #os.walk()函数返回一个元组，该元组有3个元素，这3个元素分别表示每次遍历的路径名，目录列表和文件列表
    for root, dirs, filelist in os.walk(dir3+region_name):
        # if root != r'C:\用电数据集_已完成数据清洗\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\城北供电分公司\170':
        #     continue
        print("--------------------------------------------------------------------------------------")
        print("当前在遍历的文件夹为"+root)
        print(filelist)
        if '用户基本信息.xlsx' in filelist:
            continue

        #取得两个csv文件的地址
        for i in filelist:

            if i=='2021.csv':
                min_time_link = root+'\\2021.csv'
                print(min_time_link)#C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\滨江供电分公司\1\2021.csv
            if i=='负荷特性.csv':
                day_time_link = root+'\\负荷特性.csv'
                print(day_time_link)#C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\滨江供电分公司\1\负荷特性.csv
                flag=True  #只遍历一遍的标记

        # 数据处理
        if day_time_link.endswith('csv'):
            # 读取日数据
            df_day_time = pd.read_csv(day_time_link, encoding='utf-8', sep=',')

            # 读取实时用电数据
            df_min_time = pd.read_csv(min_time_link, encoding='utf-8', sep=',')

            #处理df_day_time为空或df_min_time为空的情况下，数据Merge报错：
            if len(df_day_time)==0:
                continue
            if len(df_min_time) == 0:
                continue
            # # 取得供电单位、户名
            # region = df_min_time.loc[:, '供电单位'][0]

            #中长期数据分类
            #assign函数：向df添加一列
            #添加一列 “年-月”
            df_mt_time=df_day_time
            df_mt_time=df_mt_time.astype({'数据时间': 'datetime64[ns]'}).assign(数据时间2=lambda x: x['数据时间'].dt.strftime('%Y-%m'))
            #print(df_mt_time.head())

            #月维度的负荷值-->取当月均值
            df_mt_time['平均负荷(kW)'] = df_mt_time.groupby('数据时间2')['平均负荷(kW)'].transform(np.mean)

            #月维度的受电容量-->取当月均值
            df_mt_time['受电容量(KVA)'] = df_mt_time.groupby('数据时间2')['受电容量(KVA)'].transform(np.mean)

            #月维度的最大最小负荷值
            df_mt_time['max_month']=df_mt_time.groupby('数据时间2')['最大负荷(kW)'].transform(np.max)
            df_mt_time['min_month']=df_mt_time.groupby('数据时间2')['最小负荷(kW)'].transform(np.min)

            # 月维度的最大负荷-->取当月最大负荷产生的日期
                #取得当月最大值-series
            max_the_month=df_mt_time.groupby('数据时间2')['最大负荷(kW)'].max()
            #print("XXXXXXXX")
            #print(max_the_month)
            df_mt_time2=df_mt_time
                #原df_mt_time2中增加一列:当月的最大值
            df_mt_time2=pd.merge(df_mt_time2, max_the_month, how='inner',
                                on=None, left_on="数据时间2", right_on="数据时间2")
            df_mt_time2=df_mt_time2.sort_values(by=['数据时间'], ascending=[True])
            df_mt_time2.rename(columns={'最大负荷(kW)_x': '最大负荷(kW)', '最大负荷(kW)_y': '本月最大负荷'}, inplace=True)


            max_load_filter=df_mt_time2['最大负荷(kW)']==df_mt_time2['本月最大负荷']
            #print(max_load_filter)

            month_max_load=df_mt_time2[max_load_filter==True]
            month_max_load=month_max_load[['数据时间2','数据时间']]

            month_max_load.rename(columns={'数据时间2': '月份', '数据时间': '最大负荷发生日'}, inplace=True)
            #print(month_max_load.head())

            df_mt_time = pd.merge(df_mt_time, month_max_load, how='inner',
                                   on=None, left_on="数据时间2", right_on="月份")

            df_mt_time=df_mt_time.drop(columns=['月份'])

            # 月维度的最小负荷-->取当月最小负荷产生的日期
            # 取得当月最小值-series
            min_the_month = df_mt_time.groupby('数据时间2')['最小负荷(kW)'].min()
            #print("XXXXXXXX")
            #print(min_the_month)
            df_mt_time3 = df_mt_time
            # 原df_mt_time2中增加一列:当月的最大值
            df_mt_time3 = pd.merge(df_mt_time3, min_the_month, how='inner',
                                   on=None, left_on="数据时间2", right_on="数据时间2")
            df_mt_time3 = df_mt_time3.sort_values(by=['数据时间'], ascending=[True])
            df_mt_time3.rename(columns={'最小负荷(kW)_x': '最小负荷(kW)', '最小负荷(kW)_y': '本月最小负荷'}, inplace=True)

            min_load_filter = df_mt_time3['最小负荷(kW)'] == df_mt_time3['本月最小负荷']
            # print(max_load_filter)

            month_min_load = df_mt_time3[min_load_filter == True]
            month_min_load = month_min_load[['数据时间2', '数据时间']]

            month_min_load.rename(columns={'数据时间2': '月份', '数据时间': '最小负荷发生日'}, inplace=True)
            #print(month_min_load.head())

            df_mt_time = pd.merge(df_mt_time, month_min_load, how='inner',
                                  on=None, left_on="数据时间2", right_on="月份")

            df_mt_time = df_mt_time.drop(columns=['月份'])


            #去重
            df_mt_time.drop_duplicates(keep='first', inplace=True, subset=['数据时间2'])  # 去重

            df_mt_time = df_mt_time.drop(columns=['最小负荷(kW)'])
            df_mt_time = df_mt_time.drop(columns=['最大负荷(kW)'])
            df_mt_time = df_mt_time.drop(columns=['最大负荷发生时间'])
            df_mt_time = df_mt_time.drop(columns=['最小负荷发生时间'])
            df_mt_time.rename(columns={'max_month':'最大负荷(kW)'},inplace=True)
            df_mt_time.rename(columns={'min_month': '最小负荷(kW)'},inplace=True)
            #
            print("000000000000000000000000000000000000000000000000000000000000000")

            print(df_min_time.head())
            print(len(df_min_time))
            print(df_day_time.head())
            print(len(df_day_time))
            print(df_mt_time.head())
            print(len(df_mt_time))
#如果3个数据的数据量都太小，不用建目录
            if ((len(df_min_time)) < 50) and ((len(df_day_time)) < 50) and ((len(df_mt_time)) < 50):
                continue

    #将处理完的数据转成csv文件
            #拼接新目录
            min_time_link_temp=min_time_link
            min_time_link_temp=min_time_link_temp.replace('用电数据集_已完成数据清洗','用电数据集_已完成数据分类')
            print(min_time_link_temp)

            day_time_link_temp = day_time_link
            day_time_link_temp = day_time_link_temp.replace('用电数据集_已完成数据清洗', '用电数据集_已完成数据分类')
            print(day_time_link_temp)
            #把csv保存到本地之前先创建目录
            list_min_time_link_temp=min_time_link_temp.split(sep='\\')
            print(list_min_time_link_temp)
            list_min_time_link_temp.pop()
            print(list_min_time_link_temp)
            dir_list_min_time_link_temp="\\".join(list_min_time_link_temp)
            if os.path.exists(dir_list_min_time_link_temp)==False:
                os.makedirs(dir_list_min_time_link_temp)
            #保存csv
            print(day_time_link_temp)
            day_time_link_temp_list=day_time_link_temp.split('\\')
            day_time_link_temp_list.pop()
            day_time_link_temp_list.append("ST_data.csv")
            day_time_link_temp="\\".join(day_time_link_temp_list)
            print(day_time_link_temp)

            min_time_link_temp_list=min_time_link_temp.split('\\')
            min_time_link_temp_list.pop()
            min_time_link_temp_list.append("RT_data.csv")
            min_time_link_temp="\\".join(min_time_link_temp_list)
            print(min_time_link_temp)

            mt_time_link_temp_list=min_time_link_temp.split('\\')
            mt_time_link_temp_list.pop()
            mt_time_link_temp_list.append("MT_data.csv")
            mt_time_link_temp="\\".join(mt_time_link_temp_list)
            print(mt_time_link_temp)

            if day_time_link.endswith('csv'):
                if (len(df_min_time)) >= 50:
                    df_min_time.to_csv(path_or_buf=min_time_link_temp, encoding="utf_8_sig",index=False) #实时数据
                if (len(df_day_time)) >= 50:
                    df_day_time.to_csv(path_or_buf=day_time_link_temp, encoding="utf_8_sig",index=False)#短期数据
                if (len(df_mt_time)) >= 24:
                    df_mt_time.to_csv(path_or_buf=mt_time_link_temp, encoding="utf_8_sig", index=False)#中长期
            print("处理数据量太小后的数据量")







        # if flag==True:
        #     break


