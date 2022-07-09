#df.drop( index = df.age[df1.age == 0].index )
import os
import pandas as pd
region_name = "滨江供电分公司"
def do(region_name,dir2):
    min_time_link = ''
    day_time_link = ''
    flag=False#循环只执行一次的标记
    #xlsx转csv
    # def xlsx_to_csv_pd(xlsx):
    #     data_xls = pd.read_excel(xlsx, index_col=0)
    #     data_xls.to_csv('csv_business_type.csv', encoding='utf-8')


    #os.walk()函数返回一个元组，该元组有3个元素，这3个元素分别表示每次遍历的路径名，目录列表和文件列表
    for root, dirs, filelist in os.walk(dir2+region_name):
        #剔除数据量太小的公司
        # if day_time_link.endswith('csv'):
        #     df_min_time = pd.read_csv(min_time_link, encoding='gbk', sep=',')
        #     df_day_time = pd.read_csv(day_time_link, encoding='gbk', sep=',')
        #     if (len(df_min_time))<10:
        #         continue
        #     if (len(df_day_time))<10:
        #         continue

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
            print(df_day_time.head())

            # 读取实时用电数据
            df_min_time = pd.read_csv(min_time_link, encoding='utf-8', sep=',')
            print(df_min_time.head())

            # # 取得供电单位、户名
            # region = df_min_time.loc[:, '供电单位'][0]

            #只取需要的字段
            df_min_time=df_min_time.loc[:, ['日期','瞬时有功(kW)','局号(终端/表计)','供电单位','户名','business_type']]
            #print(df_min_time.head())
            df_day_time = df_day_time.loc[:, ['数据时间', '平均负荷(kW)', '最大负荷(kW)', '最小负荷(kW)',
                                              '最小负荷发生时间', '最大负荷发生时间','受电容量(KVA)',
                                              'region','company','business_type','局号(终端/表计)']]
            #去除为0值的数据
            print("处理0值前的数据量")
            print(len(df_min_time))
            print(len(df_day_time))

            df_min_time=df_min_time[df_min_time['瞬时有功(kW)']!=0]
            df_min_time = df_min_time[df_min_time['局号(终端/表计)'] != 0]

            df_day_time = df_day_time[df_day_time['平均负荷(kW)'] != 0]
            df_day_time = df_day_time[df_day_time['最大负荷(kW)'] != 0]
            df_day_time = df_day_time[df_day_time['最小负荷(kW)'] != 0]
            df_day_time = df_day_time[df_day_time['受电容量(KVA)'] != 0]
            df_day_time = df_day_time[df_day_time['局号(终端/表计)'] != 0]



            print("处理0值后的数据量")
            print(len(df_min_time))
            print(len(df_day_time))

            #去除为空值的数据
            print("处理空值前的数据量")
            print(len(df_min_time))
            print(len(df_day_time))

            df_min_time=df_min_time[df_min_time['瞬时有功(kW)'].notnull()]
            df_min_time = df_min_time[df_min_time['局号(终端/表计)'] .notnull()]
            df_min_time = df_min_time[df_min_time['日期'].notnull()]
            df_min_time = df_min_time[df_min_time['供电单位'].notnull()]
            df_min_time = df_min_time[df_min_time['户名'].notnull()]
            df_min_time = df_min_time[df_min_time['business_type'].notnull()]

            df_day_time = df_day_time[df_day_time['平均负荷(kW)'] .notnull()]
            df_day_time = df_day_time[df_day_time['最大负荷(kW)'] .notnull()]
            df_day_time = df_day_time[df_day_time['最小负荷(kW)'] .notnull()]
            df_day_time = df_day_time[df_day_time['受电容量(KVA)'] .notnull()]
            df_day_time = df_day_time[df_day_time['business_type'] .notnull()]
            df_day_time = df_day_time[df_day_time['数据时间'].notnull()]
            df_day_time = df_day_time[df_day_time['最小负荷发生时间'].notnull()]
            df_day_time = df_day_time[df_day_time['最大负荷发生时间'].notnull()]

            print("处理空值后的数据量")
            print(len(df_min_time))
            print(len(df_day_time))

            #去重
            # data.drop_duplicates(subset=['A', 'B'], keep='first', inplace=True)
            # 代码中subset对应的值是列名，表示只考虑这两列，将这两列对应值相同的行进行去重。
            # 默认值为subset = None表示考虑所有列。keep = 'first'
            # 表示保留第一次出现的重复行，是默认值。keep另外两个取值为"last"和False，
            # 分别表示保留最后一次出现的重复行和去除所有重复行。
            # inplace = True表示直接在原来的DataFrame上删除重复项，而默认值False表示生成一个副本。

            #去重前的数据量
            print("处理空值前的数据量")
            print(len(df_min_time))
            print(len(df_day_time))
            df_min_time.drop_duplicates(keep='first', inplace=True)#去重
            df_day_time.drop_duplicates(keep='first', inplace=True)

            #处理同一时刻，同一企业，有多个电表的问题——分组、聚合后对组内的负荷值求和、去重
            # 使用transform函数对groupby对象进行变换，transform的计算结果和原始数据的形状保持一致。
            print("处理多表问题前的数据量")
            print(len(df_min_time))
            print(len(df_day_time))
            # ['数据时间', '平均负荷(kW)', '最大负荷(kW)', '最小负荷(kW)',
            # '最小负荷发生时间', '最大负荷发生时间', '受电容量(KVA)',
            # 'region', 'company', 'business_type', '局号(终端/表计)']]
            df_min_time['瞬时有功(kW)']=df_min_time.groupby('日期')['瞬时有功(kW)'].transform(sum)

            df_day_time['平均负荷(kW)']=df_day_time.groupby('数据时间')['平均负荷(kW)'].transform(sum)
            df_day_time['最大负荷(kW)'] = df_day_time.groupby('数据时间')['最大负荷(kW)'].transform(sum)
            df_day_time['最小负荷(kW)'] = df_day_time.groupby('数据时间')['最小负荷(kW)'].transform(sum)
            #去重
            df_min_time.drop_duplicates(keep='first', inplace=True,subset=['日期'])  # 去重
            df_day_time.drop_duplicates(keep='first', inplace=True, subset=['数据时间'])  # 去重
        #    df_day_time.drop_duplicates(keep='first', inplace=True)
            print("处理多表问题后的数据量")
            print(len(df_min_time))
            print(len(df_day_time))
            #去负值
            df_min_time=df_min_time[df_min_time['瞬时有功(kW)']>0]
            #df_min_time = df_min_time[df_min_time['局号(终端/表计)'] != 0]

            df_day_time = df_day_time[df_day_time['平均负荷(kW)'] >= 0]
            df_day_time = df_day_time[df_day_time['最大负荷(kW)'] >= 0]
            df_day_time = df_day_time[df_day_time['最小负荷(kW)'] >= 0]
            df_day_time = df_day_time[df_day_time['受电容量(KVA)'] >= 0]
            #df_day_time = df_day_time[df_day_time['局号(终端/表计)'] != 0]
            #剔除数据量太小的公司

            print("去重后的数据量")
            print(len(df_min_time))
            print(len(df_day_time))

            # if day_time_link.endswith('csv'):
            #     if (len(df_min_time))<50:
            #         continue
            #     if (len(df_day_time))<50:
            #         continue
            #
            # print("处理数据量太小后的数据量")
            # print(len(df_min_time))
            # print(len(df_day_time))
    #将处理完的数据转成csv文件
            #拼接新目录
            min_time_link_temp=min_time_link
            min_time_link_temp=min_time_link_temp.replace('用电数据集_已合并','用电数据集_已完成数据清洗')
            print(min_time_link_temp)

            day_time_link_temp = day_time_link
            day_time_link_temp = day_time_link_temp.replace('用电数据集_已合并', '用电数据集_已完成数据清洗')
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
            df_day_time.to_csv(path_or_buf=day_time_link_temp, encoding="utf_8_sig",index=False)
            df_min_time.to_csv(path_or_buf=min_time_link_temp, encoding="utf_8_sig",index=False)

        # if flag==True:
        #     break


