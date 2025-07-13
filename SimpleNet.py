import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(784, 10)  # 输入 784 维，输出 10 维（例如手写数字分类）

    def forward(self, x):
        out = self.fc(x)
        return out


# 实例化模型
model = SimpleNet()
print(model)

# 设置设备（如果有 GPU 就用 GPU，否则用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 假设我们有一个批次的输入数据（如手写数字图像，已展平为784维向量）
batch_size = 32
dummy_input = torch.randn(batch_size, 784).to(device)  # 随机生成一批数据并迁移到 device
dummy_labels = torch.randint(0, 10, (batch_size,)).to(device)  # 随机生成对应的标签并迁移到 device

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 前向传播
outputs = model(dummy_input)
loss = criterion(outputs, dummy_labels)
print("初始损失:", loss.item())

# 反向传播和优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 再次输出损失（仅作为示例，损失值可能不会明显下降）
outputs_after = model(dummy_input)
loss_after = criterion(outputs_after, dummy_labels)
print("更新后损失:", loss_after.item())

# 对一条测试数据进行预测
test_sample = torch.randn(1, 784).to(device)
pred_logits = model(test_sample)
pred_label = torch.argmax(pred_logits, dim=1)
print("预测类别:", pred_label.item())