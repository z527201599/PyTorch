import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
#import matplotlib.pyplot as plt

print(torch.__version__, torchvision.__version__)

#跨硬件 Cuda 或 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备：{device}")

#设置seed 复现性
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

#超参数
BATCH_SIZE = 128
EPOCHS = 10    #可增大epochs超参-增加准确度
LEARNING_RATE = 3e-4
PATCH_SIZE = 4
NUM_CLASSES = 10
IMAGE_SIZE = 32    #或可调节imageSize超参
CHANNELS = 3
EMBED_DIM = 256
NUM_HEADS = 8    #或增加注意力头
DEPTH = 6
MLP_DIM = 512    #多层感知器维度
DROP_RATE =  0.1    #正则化

#定义图片转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))  #加速模型收敛
])

# #在训练完成后进行数据增强微调（扩充）
# transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5), (0.5))
# ])

#数据集（PyTorch数据集）
train_dataset = datasets.CIFAR10(
    root = "data",
    train = True,
    download = True,
    transform = transform  #定义的图片转换
)

test_dataset = datasets.CIFAR10(
    root = "data",
    train = True,
    download = True,
    transform = transform  #定义的图片转换
)

print(train_dataset)
print(len(train_dataset))

#构建数据加载器(PyTorch Datasets → 加载器)(转换为批次batches)(break增加计算效率，增加每轮epoch更新梯度的次数)
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)
test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False
)
print(f"数据加载器：{train_loader, test_loader}")
print(f"""训练集长度：{len(train_loader)} batches of {BATCH_SIZE}
测试集长度：{len(test_loader)} batches of {BATCH_SIZE}
""")

# 构造Vision Transformer from Scratch
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels = in_channels,
            out_channels = embed_dim,
            kernel_size = patch_size,  #patch size 定义就是卷积核的大小
            stride = patch_size
        )
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x)
        x= x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token(B, -1, -1)
        x = torch.cat((cls_token, x), dim = 1)
        x = x + self.pos_embed
        return x

class MLP(nn.Module):  #MLP
    def __init__(
        self,
        in_features,
        hidden_features,
        drop_rate
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features = in_features, out_features = hidden_features)
        self.fc2 = nn.Linear(in_features = hidden_features, out_features = in_features)
        self.dropout = nn.Dropout(drop_rate)  #减少过拟合
    
    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))  #fc -> ReLU -> Dropout正则化
        x = self.dropout(self.fc2(x))
        return x

class TransformerEncoderLayer(nn.Module):  #Encoder顺序： Input -> (Multi-Head Attention -> Add & Norm) -> (MLP -> Add & Norm) -> Output
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):  #Build graph
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder = nn.Sequential([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, drop_rate)
            for _ in range(depth)  #匿名的临时变量，用来复制和组装_个Encoder层
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)  #分类器, embed_dim -> 每个类10个概率
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

#VT实例化/重实例化
model = VisionTransformer(
    IMAGE_SIZE, PATCH_SIZE, CHANNELS, NUM_CLASSES, 
    EMBED_DIM, DEPTH, NUM_HEADS, MLP_DIM, DROP_RATE
).to(device)  #DEPTH = 6, 6个Transformer编码层

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  #交叉熵用于多种类分类
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)

#模型训练函数
def train(model, loader, optimizer, criterion):
    model.train()  #训练模式

    total_loss, correct = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()  #在每一次都需要重置归零梯度
        out = model(x)  #Forward前向传播
        loss = criterion(out, y)  #在每个batch计算loss
        loss.backward()  #反向传播
        optimizer.step()  #梯度下降

        total_loss += loss.item * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    #标准化
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

#定义评估函数
def evaluate(model, loader):
    model.eval()  #评估模式
    correct = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(dim=1) == y).sum().item()
            #按行计算、使...最大值的索引
    return correct / len(loader.dataset)

#训练
train_accuracies, test_accuracies = [], []

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_acc = evaluate(model, test_loader)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    print(f"Epoch: {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# #PLOT
# plt.plot(train_accuracies, label = "Train Acc")
# plt.plot(test_accuracies, label = "Test Acc")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.title("训练与测试集准确度")
# plt.show()

#预测
import random
def predict_and_plot_grid(
    model,
    dataset,
    classes,
    grid_size = 3
):
    model.eval()
    fig, axes = plt.subplot(grid_size, grid_size, figsize = (9,9))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = random.randint(0, len(dataset) -1)
            img, true_label = dataset[idx]
            input_tensor = img.unsqueeze(dim=0).to(device)
            with torch.inference_mode():
                #在预测-推理模式中会关闭反向传播算法
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
            img = img / 2 + 0.5  #Unnormalize for plot
            npimg = img.cpu().numpy()
            axes[i, j].imshow(np.transpose(npimg, (1,2,0)))  #需要转置（对于np）
            truth = classes[true_label] == classes[predicted.item()]
            if truth:
                color = "g"
            else:
                color = "r"

            axes[i, j].set_title(f"真实标签：{classes[true_label]}\n预测：{classes[predicted.item()]}", fontsize = 10, c = color)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()

#Call
predict_and_plot_grid(model, test_dataset, classes = train_dataset.classes, grid_size = 3)
