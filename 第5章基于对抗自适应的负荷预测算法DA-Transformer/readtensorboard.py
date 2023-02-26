from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns
# 解决中文和负号显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 使显示图标自适应
plt.rcParams['figure.autolayout'] = True
# 加载日志数据
ea = event_accumulator.EventAccumulator('D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第9章DA-Transformer\\log\\按频域分解聚类划分数据集V2_领域自适应-汇总标准化_rtclass_12_RT_data_96_202022-12-08T23-33-16')
ea.Reload()
#print(ea.scalars.Keys())
#['鉴别器的训练损失_单位_批次_', '负荷预测的训练损失_单位_批次_', '训练时MMD值_单位_批次_',
# '负荷预测_MMD_鉴别器损失的加权和_单位_批次_', '负荷预测验证集损失_单位_epoch_']
sns.set_style('darkgrid',{'font.sans-serif':['SimHei','Arial']})
data = ea.scalars.Items('负荷预测验证集损失_单位_epoch_')
x=[]
y=[]
for i in data:
    y.append(i.value)
    x.append(i.step)
plt.title = ("验证集损失")
plt.xlabel=("测试样本")
plt.ylabel=("均方损失(MSE)")

#plt.plot(x,y,label="丰富的发顺丰")
sns.lineplot(x,y,ci='red',label="丰富的发顺丰")
#plt.savefig('D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第9章DA-Transformer')
plt.legend()
#plt.savefig('D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第9章DA-Transformer\\fsf.jpg')
plt.show()


with open('D:\\myfiles\\毕业论文\\论文撰写\\实验结果与配图\\第9章DA-Transformer\\yyhat.pk', 'rb') as f:
    data = pickle.load(f)
f.close()