import torch
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as dataloader
import torch.nn as nn


class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28).astype(float)
        self.Y = np.array(data.iloc[:, 0])
        self.len = len(self.X)
        del data

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item = self.X[idx]
        label = self.Y[idx]
        return (item, label)


class CNN(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(CNN, self).__init__()
        # 卷积层conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        # 池化层pool1
        self.pool1 = nn.MaxPool2d(2)
        # 卷积层conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU())
        # 卷积层conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        # 池化层pool2
        self.pool2 = nn.MaxPool2d(2)
        # 全连接层fc(输出层)
        self.fc = nn.Linear(5 * 5 * 64, 10)

    # 前向传播
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool2(out)
        # 压缩成向量以供全连接
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


DATA_PATH = Path('./data/')
train_dataset = FashionMNISTDataset(csv_file=DATA_PATH / "fashion-mnist_train.csv")
test_dataset = FashionMNISTDataset(csv_file=DATA_PATH / "fashion-mnist_test.csv")
BATCH_SIZE = 256
train_loader = dataloader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = dataloader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)
cnn = CNN()
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
cnn = cnn.to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
LEARNING_RATE = 0.01
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
TOTAL_EPOCHS = 5
losses = []
for epoch in range(TOTAL_EPOCHS):
    # 在每个批次下,遍历每个训练样本
    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(DEVICE)
        labels = labels.to(DEVICE)
        # 清零
        optimizer.zero_grad()
        outputs = cnn(images)
        # 计算损失函数
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item());
        if (i + 1) % 100 == 0:
            print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (
            epoch + 1, TOTAL_EPOCHS, i + 1, len(train_dataset) // BATCH_SIZE, loss.data.item()))

plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.plot(losses)
plt.show()
torch.save(cnn.state_dict(), "fm-cnn3.pth")
cnn.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = images.float().to(DEVICE)
    outputs = cnn(images).cpu()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('准确率: %.4f %%' % (100 * correct / total))
