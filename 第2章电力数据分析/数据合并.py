import os
import pandas as pd
# region_name = "滨江供电分公司"
def do(region_name,dir1):
    min_time_link = ''
    day_time_link = ''
    business_type_link=dir1 +region_name+"\\用户基本信息.xlsx"

    #python中，如果你的字符串最后一位是斜杠（slash）字符，那么即使字符串前面加了r表示regular的普通字符串，
    #也是无法通过编译的，也是会导致SyntaxError的。

    #xlsx转csv
    def xlsx_to_csv_pd(xlsx):
        data_xls = pd.read_excel(xlsx, index_col=0)
        data_xls.to_csv('csv_business_type.csv', encoding='utf-8')

    #读取企业类型数据
    xlsx_to_csv_pd(business_type_link)#csv_business_type.csv
    df_business_type = pd.read_csv('csv_business_type.csv',encoding='utf-8',names=['区域','企业类型'],sep=',')
    #print(df_business_type)
    #print(type(df_business_type))



    #剔除问题数据
    #os.walk()函数返回一个元组，该元组有3个元素，这3个元素分别表示每次遍历的路径名，目录列表和文件列表
    for root, dirs, filelist in os.walk(dir1+region_name):

        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\淳安县供电分公司1\31':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\富阳区供电分公司\122':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\富阳区供电分公司\3':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\建德市供电分公司\198':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\临安区供电分公司\185':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\余杭区供电分公司6\110':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉善县供电分公司4\106':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉善县供电分公司4\113':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉善县供电分公司5\89':
            continue

        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\1':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\87':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\107':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\1':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\1':
            continue
        if root == r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\1':
            continue
        # if root != r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\淳安县供电分公司4\127':
        #     continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\淳安县供电分公司1\31':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\富阳区供电分公司\122':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\富阳区供电分公司\3':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\建德市供电分公司\198':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\临安区供电分公司\185':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\余杭区供电分公司6\110':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉善县供电分公司4\106':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉善县供电分公司4\113':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉善县供电分公司5\89':
            continue

        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\1':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\87':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\107':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\1':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\1':
            continue
        if root == r'C:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司\嘉兴供电分公司1\1':
            continue
        # 剔除数据量太小的公司
        if day_time_link.endswith('csv'):
            df_min_time = pd.read_csv(min_time_link, encoding='gbk', sep=',')
            df_day_time = pd.read_csv(day_time_link, encoding='gbk', sep=',')
            # if (len(df_min_time))<10:
            #     continue
            # if (len(df_day_time))<10:
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
                #flag=True
        # 取得企业类型
        #print(root)
        getlinklist=root.split(sep='\\')
        company_index=getlinklist.pop()
        #print(company_index)
        bus_type = df_business_type.iloc[int(company_index)-1, 1]

        print(bus_type)  # Series
        # 合并
        if day_time_link.endswith('csv'):
            # 读取日数据
            df_day_time = pd.read_csv(day_time_link, encoding='gbk', sep=',')
            print(df_day_time.head())

            # 读取实时用电数据
            try:
                df_min_time = pd.read_csv(min_time_link, encoding='gbk', sep=',')
            except:
                df_min_time = pd.read_csv(min_time_link, encoding='utf-8', sep=',')

            #print(len(df_min_time))
            #print(df_min_time.head())

            # 取得供电单位、户名
            region = df_min_time.loc[:, '供电单位'][0]
            print(df_min_time.loc[:, '供电单位'][0])
            print(df_min_time.loc[:, '户名'][0])
            company = df_min_time.loc[:, '户名'][0]

            # 把公司名和行政区域添加到日负荷数据文件中
            df_day_time['region'] = region
            df_day_time['company'] = company

            # 把企业类型加到日负荷数据文件中
            df_day_time['business_type'] = bus_type

            # 把企业类型加到实时数据文件中
            df_min_time['business_type'] = bus_type
            print(df_min_time)



    #将处理完的数据转成csv文件
            #拼接新目录
            min_time_link_temp=min_time_link
            min_time_link_temp=min_time_link_temp.replace('用电数据集','用电数据集_已合并')
            print(min_time_link_temp)

            day_time_link_temp = day_time_link
            day_time_link_temp = day_time_link_temp.replace('用电数据集', '用电数据集_已合并')
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


