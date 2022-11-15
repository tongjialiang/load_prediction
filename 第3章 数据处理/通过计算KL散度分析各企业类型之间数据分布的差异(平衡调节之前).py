
import os
import shutil
import pandas as pd
#对每个企业分类构建一个列表，列表内存放该企业类别下40家企业的瞬时功率值，每家企业截取200条数据
import scipy.stats
import pickle


list1=[] #'C:\按行业划分数据集\采矿业'
list2=[]#'C:\按行业划分数据集\电力热力燃气及水生产和供应业'
list3=[]#'C:\按行业划分数据集\房地产业'
list4=[]#'C:\按行业划分数据集\公共管理社会保障和社会组织'
list5=[]#'C:\按行业划分数据集\租赁和商务服务业'
list6=[]#'C:\按行业划分数据集\建筑业'
list7=[]#'C:\按行业划分数据集\交通运输仓储和邮政业'
list8=[]#'C:\按行业划分数据集\教育'
list9=[]#'C:\按行业划分数据集\金融业'
list10=[]#'C:\按行业划分数据集\居民服务修理和其他服务业'
list11=[]#'C:\按行业划分数据集\科学研究和技术服务业'
list12=[]#'C:\按行业划分数据集\农林牧渔业'
list13=[]#'C:\按行业划分数据集\批发和零售业'
list14=[]#'C:\按行业划分数据集\水利环境和公共设施管理业'
list15=[]#'C:\按行业划分数据集\卫生和社会工作'
list16=[]#'C:\按行业划分数据集\文化体育和娱乐业'
list17=[]#'C:\按行业划分数据集\信息传输、软件和信息技术服务业'
list18=[]#'C:\按行业划分数据集\制造业_机械电子制造业'
list19=[]#'C:\按行业划分数据集\住宿和餐饮业'
list20=[]#制造业_轻纺工业
list21=[]#制造业_资源加工工业

count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
count9=0
count10=0
count11=0
count12=0
count13=0
count14=0
count15=0
count16=0
count17=0
count18=0
count19=0
count20=0
count21=0

def do_sampling():
    for root, dirs, filelist in os.walk("D:\\按行业划分数据集\\"):
        for i in filelist:
            global count1
            global count2
            global count3
            global count4
            global count5
            global count6
            global count7
            global count8
            global count9
            global count10
            global count11
            global count12
            global count13
            global count14
            global count15
            global count16
            global count17
            global count18
            global count19
            global count20
            global count21
            if i == 'RT_data.csv':
                try:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='utf-8', sep=',')
                except:
                    RT_data = pd.read_csv(root + "\\" + i, encoding='gbk', sep=',')
                #当前遍历到哪个类
                type_at_present=root.split("\\")[-3]
                describe_at_present = RT_data['瞬时有功(kW)'][:500]#.describe()#[["mean", "std", "25%", "50%", "75%"]]
                if type_at_present=="采矿业":
                    count1=count1+1
                    if len(list1)<40:
                        list1.append(describe_at_present)
                if type_at_present=="电力热力燃气及水生产和供应业" :
                    count2 = count2 + 1
                    if len(list2)<40:
                        list2.append(describe_at_present)
                if type_at_present=="房地产业":
                    count3=count3+1
                    if len(list3)<40:
                        list3.append(describe_at_present)
                if type_at_present=="公共管理社会保障和社会组织":
                    count4=count4+1
                    if len(list4)<40:
                        list4.append(describe_at_present)
                if type_at_present == "租赁和商务服务业":
                    count5=count5+1
                    if len(list5)<40:
                        list5.append(describe_at_present)
                if type_at_present=="建筑业":
                    count6=count6+1
                    if len(list6)<40:
                        list6.append(describe_at_present)
                if type_at_present=="交通运输仓储和邮政业":

                    count7=count7+1
                    if len(list7)<40:
                        list7.append(describe_at_present)
                if type_at_present=="教育":
                    count8=count8+1
                    if len(list8)<40:
                        list8.append(describe_at_present)
                if type_at_present=="金融业":
                    count9=count9+1
                    if len(list9)<40:
                        list9.append(describe_at_present)
                if type_at_present=="居民服务修理和其他服务业":
                    count10=count10+1
                    if len(list10)<40:
                        list10.append(describe_at_present)
                if type_at_present=="科学研究和技术服务业":
                    count11=count11+1
                    if len(list11)<40:
                        list11.append(describe_at_present)
                if type_at_present=="农林牧渔业":
                    count12=count12+1
                    if len(list12)<40:
                        list12.append(describe_at_present)
                if type_at_present=="批发和零售业":
                    count13=count13+1
                    if len(list13)<40:
                        list13.append(describe_at_present)
                if type_at_present=="水利环境和公共设施管理业":
                    count14=count14+1
                    if len(list14)<40:
                        list14.append(describe_at_present)
                if type_at_present=="卫生和社会工作":
                    count15=count15+1
                    if len(list15)<40:
                        list15.append(describe_at_present)
                if type_at_present=="文化体育和娱乐业":
                    count16=count16+1
                    if len(list16)<40:
                        list16.append(describe_at_present)
                if type_at_present=="信息传输软件和信息技术服务业":
                    count17=count17+1
                    if len(list17)<40:
                        list17.append(describe_at_present)
                if type_at_present=="制造业_机械电子制造业":
                    count18=count18+1
                    if len(list18)<40:
                        list18.append(describe_at_present)
                if type_at_present=="住宿和餐饮业" :
                    count19 = count19 + 1
                    if len(list19)<40:
                        list19.append(describe_at_present)
                if type_at_present=="制造业_轻纺工业":
                    count20=count20+1
                    if len(list20)<40:
                        list20.append(describe_at_present)
                if type_at_present=="制造业_资源加工工业":
                    count21=count21+1
                    if len(list21)<40:
                        list21.append(describe_at_present)



    print("属于类别1的采样个数为： "+str(len(list1))+"属于类别1的企业个数为： "+str(count1))
    print("属于类别2的采样个数为： "+str(len(list2))+"属于类别2的企业个数为： "+str(count2))
    print("属于类别3的采样个数为： "+str(len(list3))+"属于类别3的企业个数为： "+str(count3))
    print("属于类别4的采样个数为： "+str(len(list4))+"属于类别4的企业个数为： "+str(count4))
    print("属于类别5的采样个数为： "+str(len(list5))+"属于类别5的企业个数为： "+str(count5))
    print("属于类别6的采样个数为： "+str(len(list6))+"属于类别6的企业个数为： "+str(count6))
    print("属于类别7的采样个数为： "+str(len(list7))+"属于类别7的企业个数为： "+str(count7))
    print("属于类别8的采样个数为： "+str(len(list8))+"属于类别8的企业个数为： "+str(count8))
    print("属于类别9的采样个数为： "+str(len(list9))+"属于类别9的企业个数为： "+str(count9))
    print("属于类别10的采样个数为： "+str(len(list10))+"属于类别10的企业个数为： "+str(count10))
    print("属于类别11的采样个数为： "+str(len(list11))+"属于类别11的企业个数为： "+str(count11))
    print("属于类别12的采样个数为： "+str(len(list12))+"属于类别12的企业个数为： "+str(count12))
    print("属于类别13的采样个数为： "+str(len(list13))+"属于类别13的企业个数为： "+str(count13))
    print("属于类别14的采样个数为： "+str(len(list14))+"属于类别14的企业个数为： "+str(count14))
    print("属于类别15的采样个数为： "+str(len(list15))+"属于类别15的企业个数为： "+str(count15))
    print("属于类别16的采样个数为： "+str(len(list16))+"属于类别16的企业个数为： "+str(count16))
    print("属于类别17的采样个数为： "+str(len(list17))+"属于类别17的企业个数为： "+str(count17))
    print("属于类别18的采样个数为： "+str(len(list18))+"属于类别18的企业个数为： "+str(count18))
    print("属于类别19的采样个数为： "+str(len(list19))+"属于类别19的企业个数为： "+str(count19))
    print("属于类别20的采样个数为： " + str(len(list20)) + "属于类别20的企业个数为： " + str(count20))
    print("属于类别21的采样个数为： " + str(len(list21)) + "属于类别21的企业个数为： " + str(count21))
    mylist=[list1,list2,list3,list4,list5,list6,list7,list8,list9,list10,list11,list12,list13,list14,list15,list16,list17,list18,list19,list20,list21]
    #持久化
    with open('d:/tmp.pk', 'wb+') as f:
        pickle.dump(mylist, f)
    f.close()


do_sampling() #采样计算耗时较长，为提高后续计算效率，将采样结果持久化（保存为tmp.pk）
with open('d:/tmp.pk', 'rb') as f:
    data = pickle.load(f)
#把持久化的数据赋值给变量List1-list21
for i in range(21):
    print("list" + str(i + 1) + "=data[" + str(i) + "]")
    exec ("list"+str(i+1)+"=data["+str(i)+"]")
#企业类型1 对 企业类型2的kl散度=企业类型1中每个企业的 对 企业类型2 中每个企业的kl散度累加，除以计算次数得到均值
def get_kl(x,y):
    res=0
    for i in range(40):
            res=res+scipy.stats.entropy(x[i][:200],y[i][:200])
    print (res/(40))
    return res/(40)


def get_kl_all(list):
    for i in range(21):
        print("get_kl(list"+ str(list)+",list" + str(i + 1) + ")")
        exec("get_kl(list"+ str(list)+",list" + str(i + 1) + ")")

#企业类型16对所有企业类型的散度
get_kl_all(16)






