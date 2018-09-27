import numpy as np

from deepsense import neptune
ctx = neptune.Context()

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import torch.nn.functional as F

import pandas as pd

from matplotlib import pyplot as plt



planes=np.load("/home/user/data/airplane.npy")
onions=np.load("/home/user/data/onion.npy")
#Zmiana

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = train_loss / len(train_loader.dataset)
    avg_accuracy = train_correct / len(train_loader.dataset)



    return avg_loss, avg_accuracy


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    avg_loss_val = test_loss
    avg_accuracy_val = test_correct / len(test_loader.dataset)



    return avg_loss_val, avg_accuracy_val


pltest = planes[1::100].reshape(-1, 1, 28, 28).astype('float32')
ontest = onions[1::100].reshape(-1, 1, 28, 28).astype('float32')


pltest /= 255
ontest /= 255


pltest_t = torch.from_numpy(pltest)
ontest_t = torch.from_numpy(ontest)

X_test = torch.cat((pltest_t, ontest_t), 0)
Y_test = torch.LongTensor(pltest_t.size(0)*[0]+ontest_t.size(0)*[1])


pl2 = planes[::100].reshape(-1, 1, 28, 28).astype('float32')
on2 = onions[::100].reshape(-1, 1, 28, 28).astype('float32')


pl2 /= 255
on2 /= 255


pl2t = torch.from_numpy(pl2)
on2t = torch.from_numpy(on2)



X_train = torch.cat((pl2t, on2t), 0)
Y_train = torch.LongTensor(pl2t.size(0)*[0]+on2t.size(0)*[1])

model = Net()
device = torch.device("cpu")

train_loader = DataLoader(TensorDataset(X_train, Y_train),
                          batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, Y_test),
                         batch_size=32, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



for epoch in range(10):
    avg_loss, avg_accuracy = train(model, device, train_loader, optimizer, epoch)
    avg_loss_val, avg_accuracy_val = test(model, device, test_loader)

    ctx.channel_send('Log-loss training', avg_loss)
    ctx.channel_send('Accuracy training', avg_accuracy)
    ctx.channel_send('Log-loss validation', avg_loss_val)
    ctx.channel_send('Accuracy validation', avg_accuracy_val)
