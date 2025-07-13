import torch
from torch.ao.nn.qat import Conv2d
from torch.nn.init import kaiming_uniform, xavier_uniform

print(torch.__version__)

from torch.nn import Module, MaxPool2d, Softmax

class CNN(Module):
    #构造函数
    def __init__(self, n_channels):
        super().__init__()
        #卷积层1
        self.hidden1 = Conv2d(n_channels, 32, (3,3))
        kaiming_uniform(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        #池化层1
        self.pool1 = MaxPool2d((2,2), stride=(2,2))
        #卷积层2
        self.hidden2 = Conv2d(32, 32, (3,3))
        kaiming_uniform(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        #池化层2
        self.pool2 = MaxPool2d((2,2), stride=(2,2))
        #全连接层
        self.hidden3 = Linear(5*5*32, 100)
        kaiming_uniform(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        #输出层
        self.hidden4 = Linear(100, 10)
        xavier_uniform(self.hidden4.weight)
        self.act4 = Softmax(dim=1)

#前向传播Forward Propagation

#训练模型
