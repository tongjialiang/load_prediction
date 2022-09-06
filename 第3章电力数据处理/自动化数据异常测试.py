region_list=['滨江供电分公司','本市级高压','城北供电分公司','城北供电分公司2','城南供电分公司','淳安县供电分公司1','淳安县供电分公司2',
 '淳安县供电分公司3','淳安县供电分公司4','淳安县供电分公司5','富阳区供电分公司','富阳区供电分公司2','富阳区供电分公司3','富阳区供电分公司4'
,'富阳区供电分公司5','建德市供电分公司','建德市供电分公司2','建德市供电分公司3','建德市供电分公司4','建德市供电分公司5','建德市供电分公司6',
    '临安区供电分公司','临安区供电分公司2','临安区供电分公司3','临安区供电分公司4','临安区供电分公司5','钱塘新区供电公司','西湖供电分公司'
    ,'萧山区供电分公司','萧山区供电分公司2','萧山区供电分公司3','萧山区供电分公司4','萧山区供电分公司5','余杭区供电分公司','余杭区供电分公司2'
    ,'余杭区供电分公司3','余杭区供电分公司4','余杭区供电分公司5','余杭区供电分公司6','余杭区供电分公司7',
    '海宁市供电分公司','海盐县供电分公司1','海盐县供电分公司2','海盐县供电分公司3','海盐县供电分公司4','海盐县供电分公司5',
    '嘉善县供电分公司1','嘉善县供电分公司2','嘉善县供电分公司3','嘉善县供电分公司4','嘉善县供电分公司5','嘉善县供电分公司6'
    ,'嘉兴供电分公司1','平湖市供电分公司1','平湖市供电分公司2','平湖市供电分公司3','平湖市供电分公司4','平湖市供电分公司5',
    '桐乡市供电分公司1']
before_company_count={}  #处理前，供电分公司的企业数量。 键：供电分公司，值：企业数量
after_company_count={}  #处理后，供电分公司的企业数量。 键：供电分公司，值：企业数量
after_power_supply_company_count=0
before_power_supply_company_count=0

import os
import pandas as pd


print ("1.检查 原始数据的'供电公司数量'和处理后的'供电公司数量'是否一致？")
# 1.检查原始数据的"供电公司数量"和"经合并、清洗、分类后的"供电公司数量"是否一致

for root, dirs, filelist in os.walk("D:\\用电数据集\\浙江省电力公司2021\\浙江省电力公司2021\\"):
    if root==r'D:\用电数据集\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司':
        #print(root + '__下面的目录个数为' + str(len(dirs)))
        before_power_supply_company_count=len(dirs)
        print("处理前，供电公司数量为"+str(before_power_supply_company_count))# log

for root, dirs, filelist in os.walk("D:\\用电数据集_已完成数据分类\\浙江省电力公司2021\\浙江省电力公司2021\\"):
    if root == r'D:\用电数据集_已完成数据分类\浙江省电力公司2021\浙江省电力公司2021\杭州供电公司':
        #print(root + '__下面的目录个数为' + str(len(dirs)))  # log
        after_power_supply_company_count = len(dirs)
        print("处理后，供电公司数量为" + str(after_power_supply_company_count))# log
if after_power_supply_company_count-before_power_supply_company_count==0:
    print("数据正常")

print ("2.检查同一个供电公司下面，“企业数量”和原始数据相比，是否差异10%以上")
# 2.遍历每个供电公司下的每个企业，“企业数量”和原始数据相比，是否差异10%以上，如是，手工确认是否正常
for root, dirs, filelist in os.walk("D:\\用电数据集\\浙江省电力公司2021\\浙江省电力公司2021\\杭州供电公司\\"):
    root_sp=root.split(sep='\\').pop()
    if root_sp in region_list:
        #print(root_sp+'   处理前的企业数量为：'+str(len(dirs)))
        before_company_count[root_sp]=len(dirs)
for root, dirs, filelist in os.walk("D:\\用电数据集_已完成数据分类\\浙江省电力公司2021\\浙江省电力公司2021\\杭州供电公司\\"):
    root_sp=root.split(sep='\\').pop()
    if root_sp in region_list:
        #print(root_sp+'   处理前的企业数量为：'+str(len(dirs)))
        after_company_count[root_sp]=len(dirs)
flag=0
for k1 in before_company_count:
    for k2 in after_company_count:
        if k1==k2:
            # print(k1+"处理前的企业数"+str(before_company_count[k1])+"处理后的企业数"+str(after_company_count[k2])
            #       +"差异百分比为"+str((before_company_count[k1]-after_company_count[k2])/before_company_count[k1]))
            if ((before_company_count[k1]-after_company_count[k2])/before_company_count[k1])> 0.1:
                print("发现异常数据："+"\n"+k1 + "处理前的企业数" + str(before_company_count[k1]) + "处理后的企业数" + str(after_company_count[k2])
                      + "差异百分比为" + str((before_company_count[k1] - after_company_count[k2]) / before_company_count[k1]))
                flag=1
if flag==0:
    print("数据正常")


count_MT_data={}#键：供电公司名称  值：该供电公司下mt_data文件的数量
count_ST_data={}#键：供电公司名称  值：该供电公司下st_data文件的数量
count_RT_data={}#键：供电公司名称  值：该供电公司下rt_data文件的数量
print("3.检查每个供电公司下每个企业分类后所生成的RT_data.csv、ST_data.csv、MT_data.csv的数据量是否正常？")
print("每个供电公司MT_data文件的数量")
for root, dirs, filelist in os.walk("D:\\用电数据集_已完成数据分类\\浙江省电力公司2021\\浙江省电力公司2021\\杭州供电公司\\"):
    if 'MT_data.csv' in filelist:
        #print(filelist)
        root_list=root.split(sep='\\')
        root_list.pop()
        key=root_list.pop()
        if key not in count_MT_data:
            count_MT_data[key]=0
        else:
            count_MT_data[key]=count_MT_data[key]+1
print(count_MT_data)

print("每个供电公司RT_data文件的数量")
for root, dirs, filelist in os.walk("D:\\用电数据集_已完成数据分类\\浙江省电力公司2021\\浙江省电力公司2021\\杭州供电公司\\"):
    if 'RT_data.csv' in filelist:
        #print(filelist)
        root_list=root.split(sep='\\')
        root_list.pop()
        key=root_list.pop()
        if key not in count_RT_data:
            count_RT_data[key]=0
        else:
            count_RT_data[key]=count_RT_data[key]+1
print(count_RT_data)

print("每个供电公司ST_data文件的数量")
for root, dirs, filelist in os.walk("D:\\用电数据集_已完成数据分类\\浙江省电力公司2021\\浙江省电力公司2021\\杭州供电公司\\"):
    if 'ST_data.csv' in filelist:
        #print(filelist)
        root_list=root.split(sep='\\')
        root_list.pop()
        key=root_list.pop()
        if key not in count_ST_data:
            count_ST_data[key]=0
        else:
            count_ST_data[key]=count_ST_data[key]+1
print(count_ST_data)

    #print(root+str(filelist))
print("MT_data文件数量的异常数据：")
print("以下供电公司下，所有企业均没有MT_data文件")
mt_list_all=region_list
for key in count_MT_data:
    if key in mt_list_all:
        mt_list_all.remove(key)
print(mt_list_all)
print("以下供电公司下，MT_data文件较少")
for key in count_MT_data:
    if count_MT_data[key] <50 :
        print(key)

print("RT_data文件数量的异常数据：")
print("以下供电公司下，所有企业均没有RT_data文件")
Rt_list_all=region_list
for key in count_RT_data:
    if key in Rt_list_all:
        Rt_list_all.remove(key)
print(Rt_list_all)
print("以下供电公司下，RT_data文件较少")
for key in count_RT_data:
    if count_RT_data[key] <50 :
        print(key)

print("ST_data文件数量的异常数据：")
print("以下供电公司下，所有企业均没有ST_data文件")
St_list_all=region_list
for key in count_ST_data:
    if key in St_list_all:
        St_list_all.remove(key)
print(St_list_all)
print("以下供电公司下，ST_data文件较少")
for key in count_ST_data:
    if count_ST_data[key] <50 :
        print(key)

print("4.检查每个RT_data.csv、ST_data.csv、MT_data.csv文件，各个字段是否有0值，空值")
flag2=0
flag3=0
flag4=0
for root, dirs, filelist in os.walk("D:\\用电数据集_已完成数据分类\\浙江省电力公司2021\\浙江省电力公司2021\\杭州供电公司\\"):
    for i in filelist:
        #print(root+"\\"+i)
        link=root + "\\" + i
        if i == "RT_data.csv":
            try:
                RT_data = pd.read_csv(link, encoding='utf-8', sep=',')
            except:
                RT_data = pd.read_csv(link, encoding='gbk', sep=',')
            if (pd.isnull(RT_data['瞬时有功(kW)'])==True).any():
                print(link+" 瞬时有功(kW) 存在空数据")
                flag2 = 1
            if ((RT_data['瞬时有功(kW)'])==0).any():
                print(link+"  存在零值，个数为：  "+str(len(RT_data[RT_data['瞬时有功(kW)']==0])))
                flag2 = 1
            if (pd.isnull(RT_data['供电单位'])==True).any():
                print(link+" 供电单位 存在空数据")
                flag2 = 1
            if (pd.isnull(RT_data['户名'])==True).any():
                print(link + "  户名 存在空数据")
                flag2 = 1
            if (pd.isnull(RT_data['business_type'])==True).any():
                print(link+"  business_type 存在空数据")
                flag2 = 1
if flag2==0:
   print("RT_data.csv无异常数据")


for root, dirs, filelist in os.walk("D:\\用电数据集_已完成数据分类\\浙江省电力公司2021\\浙江省电力公司2021\\杭州供电公司\\"):
    for i in filelist:
        #print(root+"\\"+i)
        link=root + "\\" + i
        if i == "ST_data.csv":
            try:
                ST_data = pd.read_csv(link, encoding='utf-8', sep=',')
            except:
                ST_data = pd.read_csv(link, encoding='gbk', sep=',')
            if (pd.isnull(ST_data['business_type'])==True).any():
                print(link+" business_type 存在空数据")
                flag3 = 1
            if ((ST_data['平均负荷(kW)'])==0).any():
                print(link+"  存在零值，个数为：  "+str(len(ST_data[ST_data['平均负荷(kW)']==0])))
                flag3 = 1
            if ((ST_data['最大负荷(kW)'])==0).any():
                print(link+"  存在零值，个数为：  "+str(len(ST_data[ST_data['最大负荷(kW)']==0])))
                flag3 = 1
            if ((ST_data['最小负荷(kW)'])==0).any():
                print(link+"  存在零值，个数为：  "+str(len(ST_data[ST_data['最小负荷(kW)']==0])))
                flag3 = 1
            if ((ST_data['受电容量(KVA)'])==0).any():
                print(link+"  存在零值，个数为：  "+str(len(ST_data[ST_data['受电容量(KVA)']==0])))
                flag3 = 1
            if (pd.isnull(ST_data['数据时间'])==True).any():
                print(link+" 数据时间 存在空数据")
                flag3 = 1
            if (pd.isnull(ST_data['company'])==True).any():
                print(link+" company 存在空数据")
                flag3 = 1
            if (pd.isnull(ST_data['region'])==True).any():
                print(link+" region 存在空数据")
            if (pd.isnull(ST_data['最小负荷发生时间']) == True).any():
                print(link + " 最小负荷发生时间 存在空数据")
            if (pd.isnull(ST_data['最大负荷发生时间']) == True).any():
                print(link + " 最大负荷发生时间 存在空数据")
                flag3 == 1

if flag3==0:
   print("ST_data.csv无异常数据")

for root, dirs, filelist in os.walk("D:\\用电数据集_已完成数据分类\\浙江省电力公司2021\\浙江省电力公司2021\\杭州供电公司\\"):
    for i in filelist:
        #print(root+"\\"+i)
        link=root + "\\" + i
        if i == "MT_data.csv":
            try:
                MT_data = pd.read_csv(link, encoding='utf-8', sep=',')
            except:
                MT_data = pd.read_csv(link, encoding='gbk', sep=',')
        if (pd.isnull(MT_data['business_type']) == True).any():
            print(link + " business_type 存在空数据")
            flag4 = 1
        if ((MT_data['平均负荷(kW)']) == 0).any():
            print(link + "  存在零值，个数为：  " + str(len(MT_data[MT_data['平均负荷(kW)'] == 0])))
            flag4 = 1
        if ((MT_data['最大负荷(kW)']) == 0).any():
            print(link + "  存在零值，个数为：  " + str(len(MT_data[MT_data['最大负荷(kW)'] == 0])))
            flag4 = 1
        if ((MT_data['最小负荷(kW)']) == 0).any():
            print(link + "  存在零值，个数为：  " + str(len(MT_data[MT_data['最小负荷(kW)'] == 0])))
            flag4 = 1
        if ((MT_data['受电容量(KVA)']) == 0).any():
            print(link + "  存在零值，个数为：  " + str(len(MT_data[MT_data['受电容量(KVA)'] == 0])))
            flag4 = 1
        if (pd.isnull(MT_data['数据时间2']) == True).any():
            print(link + " 数据时间2 存在空数据")
            flag4 = 1
        if (pd.isnull(MT_data['company']) == True).any():
            print(link + " company 存在空数据")
            flag4 = 1
        if (pd.isnull(MT_data['region']) == True).any():
            print(link + " region 存在空数据")
        if (pd.isnull(MT_data['最小负荷发生日']) == True).any():
            print(link + " 最小负荷发生日 存在空数据")
        if (pd.isnull(MT_data['最大负荷发生日']) == True).any():
            print(link + " 最大负荷发生日 存在空数据")
            flag4 == 1
if flag4==0:
   print("MT_data.csv无异常数据")