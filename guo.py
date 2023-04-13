import numpy as np
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
import scipy.io as scio
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tchdata
from torch.utils.tensorboard import SummaryWriter

from time import time
from datetime import datetime, time
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tsmoothie.utils_func import sim_randomwalk, sim_seasonal_data
from tsmoothie.smoother import LowessSmoother, WindowWrapper
from tsmoothie.smoother import ConvolutionSmoother
from tsmoothie.bootstrap import BootstrappingWrapper



def data_sampling(data_set, n_samples):
    """
    param data_set: 数据集格式csv
    param n_samples: 采样窗口的长度
    return:  数据， 标签
    """
    seq_data, seq_labels = [], []
    dt = data_set.iloc[::, :5].values
    d_label = data_set.iloc[::, 5].values

    d = np.array(list(dt)).reshape(-1, 5)  # 一组数据一共 2 个变量
    d_lab = np.array(list(d_label)).reshape(-1, 1)  # 取出标签 单个变量 0~7 , 设备共 8 种状态


    data = []
    labels = []
    length = d.shape[0] - n_samples + 1

    for j in range(n_samples):
        data.append(d[j: j + length])
        labels.append(d_lab[j:j + length])

    data = np.hstack(data)
    last_data = data[-1,::]
    add_data = np.tile(last_data, (n_samples-1, 1))
    data = np.vstack((data, add_data))

    labels = np.hstack(labels)
    last_labels = labels[-1,::]
    add_labels = np.tile(last_labels, (n_samples-1, 1))
    labels = np.vstack((labels, add_labels))

    labels[::,0][labels[::,0]>=1]=1
    labels = labels[::, 0]

    seq_data.append(data)
    seq_labels.append(labels)

    return np.vstack(seq_data),np.concatenate(seq_labels)


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_features, n_samples, is_bn=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.n_samples = n_samples
        self.is_bn = is_bn

        self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.fc = torch.nn.Linear(100, self.out_features)  #这里200需要根据不同的参数进行更改
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()
        if self.is_bn:
            self._bn = torch.nn.BatchNorm1d(hidden_size)
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

    def forward(self, x):
        seq_data = x.reshape(-1, self.n_samples, self.input_size)
        seq_feature, (_, _) = self.lstm(seq_data)
        x = seq_feature.reshape(seq_feature.size(0), -1)
        x = self.fc(x)

        return x



class AccMectric(object):
    def __init__(self):
        self._count = None
        self._sum = None
        self.reset()

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, targets, outputs):
        pred = outputs.argmax(axis=1)

        self._sum += (pred == targets).sum()
        self._count += targets.shape[0]

    def get(self):
        return self._sum / self._count
class LossMectric(object):
    def __init__(self):
        self._count = None
        self._sum = None
        self.reset()

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, loss):
        self._sum += loss.sum()
        self._count += 1

    def get(self):
        return self._sum / self._count



def train(model, optimizer, train_loader, writer, total_train_step):
    model.train()
    acc = AccMectric()
    loss_log = LossMectric()
    for i_batch, (data, labels) in enumerate(train_loader):
        x = torch.autograd.Variable(data.cpu())
        y = torch.autograd.Variable(labels.cpu()).long()
        o = model(x)
        loss = torch.nn.NLLLoss()(torch.nn.LogSoftmax(dim=1)(o), y)  # 这里使用的是交叉熵
        acc.update(labels.numpy(), o.data.cpu().numpy())  # 更新预测精度
        loss_log.update(loss.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 1000 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    return acc.get(), loss_log.get(), total_train_step



def validate(model, test_loader, para_Rem):
    model.eval()
    acc = AccMectric()
    loss_log = LossMectric()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for data, labels in test_loader:
        x = torch.autograd.Variable(data.cpu())
        y = torch.autograd.Variable(labels.cpu()).long()
        o = model(x)
        prob_o = F.softmax(o, dim=1)  # 按行 和为 1
        # 判断无故障概率>0.5时，认为确实没有概率，否则取剩余标签1-7中最大概率的
        mask = prob_o.data[:, 0] < 0.5
        Fault = torch.max(prob_o.data[:, 1:2], 1)[1] + 1  # 返回的小标从 0 开始，所以需要再 加 1
        predict = (mask*Fault).cpu().numpy()
        labels = y.data.cpu()
        # predict = torch.max(o.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predict)
        loss = torch.nn.NLLLoss()(torch.nn.LogSoftmax(dim=1)(o), y)
        acc.update(labels.numpy(), o.data.cpu().numpy())
        loss_log.update(loss.data.cpu().numpy())
    target_names = ['Fault-Free', 'Faulty ']
    confusion_matrix_result = metrics.confusion_matrix(labels_all, predict_all)  # True, Pred
    print(metrics.classification_report(labels_all, predict_all, target_names=target_names, digits=6))
    print('The confusion matrix result:\n', confusion_matrix_result)

    plot_confusion_matrix(confusion_matrix_result, target_names, title=para_Rem)
    return acc.get(), loss_log.get()

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        matrix = cm
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # plt.figure()
    # 设置输出的图片大小
    figsize = 8, 6
    figure, ax = plt.subplots(figsize=figsize)
    plt.imshow(torch.from_numpy(cm).t(), interpolation='nearest', cmap=plt.cm.Blues)
    # 设置title的大小以及title的字体
    font_title = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 15,
                  }
    plt.title(title, fontdict=font_title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print(labels)
    [label.set_fontname('Times New Roman') for label in labels]
    if normalize:
        fm_int = 'd'
        fm_float = '.1%'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[j, i], fm_float),
                     horizontalalignment="center", verticalalignment='top', family="Times New Roman",
                     weight="normal", size=12,
                     color="white" if cm[j, i] > thresh else "black")
            plt.text(j, i, format(matrix[j, i], fm_int),
                     horizontalalignment="center", verticalalignment='bottom', family="Times New Roman",
                     weight="normal",
                     size=12,
                     color="white" if cm[j, i] > thresh else "black")
    else:
        fm_int = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fm_int),
                     horizontalalignment="center", verticalalignment='bottom',
                     color="white" if cm[i, j] > thresh else "black")
    # 设置横纵坐标的名称以及对应字体格式
    font_lable = {'family': 'Times New Roman',
                  'weight': 'bold',
                  'size': 14,
                  }
    plt.xlabel('True label', font_lable)
    plt.ylabel('Predicted label', font_lable)
    plt.tight_layout()
    figtitle = str
    plt.savefig(title+'.eps', dpi=600, format='eps')
    plt.savefig(title + 'lstm_result.png', dpi=600, format='png')
    # plt.show()

n_samples = 10 # 采样窗口的长度
Bath_size = 32
n_hidden = 10   # 隐层数目
learn_rate = 0.001  # 学习率
weight_de = 0.003  # 衰减权重
input_features = 5
out_features = 2
test_size = 0.3 #测试集的划分比例



#加载数据集
dataset = pd.read_csv(r"C:/Users/28570/Desktop/航迹关联与预测/1/813 副本.csv", usecols=['height','speed',
                                                                                       'angle','longitude','latitude','labels'
])
#对数据进行采样
data, labels = data_sampling(dataset,n_samples)
#对数据的进行训练集和测试集的划分
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, random_state=42)

#  数据标准化
scaler = preprocessing.StandardScaler().fit(train_data)  #用于计算训练数据的均值和方差， 后面就会用均值和方差来
#转换训练数据
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

train_dataset = tchdata.TensorDataset(torch.from_numpy(train_data.astype(np.float32)),
                                      torch.from_numpy(train_labels))
test_dataset = tchdata.TensorDataset(torch.from_numpy(test_data.astype(np.float32)),
                                     torch.from_numpy(test_labels.astype(np.float16)))

train_loader = tchdata.DataLoader(train_dataset, batch_size=Bath_size, shuffle=True)
test_loader = tchdata.DataLoader(test_dataset, batch_size=Bath_size, shuffle=False)
# 添加tensorboard
writer = SummaryWriter("../logs_train")
total_train_step = 0  # 记录训练总步数
#模型初始化
model = LSTM(input_features,n_hidden,out_features,n_samples,True)
torch.save(model, './LSTM.pkl')
torch.backends.cudnn.benchmark = True
model.cpu()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_de)   # 优化参数配置
para_Rem = 'S_'+str(n_samples)+'-B_'+str(Bath_size)+'-H_'+str(n_hidden)+'-L_'+str(learn_rate)
max_test_acc = 0

for i in range(20):
    train_acc, train_loss, total_train_step = train(model, optimizer, train_loader, writer, total_train_step)
    test_acc, test_loss = validate(model, test_loader, para_Rem)
    if max_test_acc <= test_acc:
        max_test_acc = test_acc
        best_model = copy.deepcopy(model)
    print(format(datetime.now()))
    print('epoch = {}\ttrain accuracy: {:0.3f}\ttrain loss: {:0.3f}\ttest accuracy: {:0.3f}\ttest loss: {:0.3f}'\
          .format(i, train_acc, train_loss, test_acc, test_loss))
    print(f'————————————————————结束第{total_train_step+1}次迭代—————————————————————————————')
torch.save(best_model,'LSTM.pkl')
