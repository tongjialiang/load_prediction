import os
import shutil
import pandas as pd
MT_data_old=0  #重新划行业分类别之前的MT_data.csv，RT_data.csv,ST_data.csv计数
RT_data_old=0
ST_data_old=0
MT_data_new=0 #重新划行业分类别后前的MT_data.csv，RT_data.csv,ST_data.csv计数
RT_data_new=0
ST_data_new=0
#1.对重新划行业分类别前后的MT_data.csv，RT_data.csv,ST_data.csv分别计数，若两者一致，则测试通过，否则，测试不通过。
for root, dirs, filelist in os.walk("D:\\用电数据集_已完成数据分类\\浙江省电力公司2021\\浙江省电力公司2021\\杭州供电公司\\"):
    for i in filelist:
        if i=='MT_data.csv':
            MT_data_old=MT_data_old+1
        if i == 'ST_data.csv':
            ST_data_old=ST_data_old+1
        if i == 'RT_data.csv':
            RT_data_old=RT_data_old+1

for root, dirs, filelist in os.walk("D:\\按行业划分数据集2\\"):
    for i in filelist:
        if i=='MT_data.csv':
            MT_data_new=MT_data_new+1
        if i == 'ST_data.csv':
            ST_data_new=ST_data_new+1
        if i == 'RT_data.csv':
            RT_data_new=RT_data_new+1

print("重新划行业分类别之前的MT_data.csv，RT_data.csv,ST_data.csv计数")
print("MT_data_old:     "+str(MT_data_old))
print("RT_data_old:     "+str(RT_data_old))
print("ST_data_old:     "+str(ST_data_old))
print("重新划行业分类别之后的MT_data.csv，RT_data.csv,ST_data.csv计数")
print("MT_data_new:     "+str(MT_data_new))
print("RT_data_new:     "+str(RT_data_new))
print("ST_data_new:     "+str(ST_data_new))


#2.把分类前后的供电公司名称分别放在两个集合里，比较两个集合的差集，从而得到未被正确分类的企业名称，以便排查数据。
company_name=[]

for root, dirs, filelist in os.walk("D:\\用电数据集_已完成数据分类\\浙江省电力公司2021\\浙江省电力公司2021\\杭州供电公司\\"):
    for i in filelist:
        if i in ['MT_data.csv','ST_data.csv','RT_data.csv']:#'MT_data.csv','ST_data.csv',
           res=root.split("\\")[-2]+"\\"+root.split("\\")[-1]
           #print(res)
           company_name.append(res)
           #company_name.add(res)


for root, dirs, filelist in os.walk("D:\\按行业划分数据集2\\"):
    for i in filelist:
        if i in ['MT_data.csv','ST_data.csv','RT_data.csv']:#'MT_data.csv','ST_data.csv',
           res=root.split("\\")[-2]+"\\"+root.split("\\")[-1]
           if res in company_name:
                company_name.remove(res)
print("以下是没有被正确分类的企业")
print(company_name)