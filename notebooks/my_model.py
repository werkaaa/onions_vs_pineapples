#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import torch.nn.functional as F

import pandas as pd

from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from livelossplot import PlotLosses


# In[2]:


def data_load(names, amount): #pobiera dane, argumenty: tablica nazw klas, ilośc danych
    data_t = {}
    X_train = torch.tensor([])
    Y_train = torch.LongTensor([])
    X_test = torch.tensor([])
    Y_test = torch.LongTensor([])
    for i in range(len(names)):
        data = np.load("/home/user/data/"+names[i]+".npy")
        data_t =torch.from_numpy(data[:amount].reshape(-1, 1, 28, 28).astype('float32')/255)
        X_train = torch.cat((X_train, data_t), 0)
        Y_train = torch.cat((Y_train, torch.LongTensor(amount*[i])),0)
        
        data_t =torch.from_numpy(data[amount:2*amount].reshape(-1, 1, 28, 28).astype('float32')/255)
        X_test = torch.cat((X_test, data_t), 0)
        Y_test = torch.cat((Y_test, torch.LongTensor(amount*[i])),0)
        
    return X_train, Y_train, X_test, Y_test


# In[3]:


class Net(nn.Module):
    def __init__(self, classes_num):
        self.classes_num = classes_num
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, self.classes_num)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# In[4]:


def conv_train_step(model, device, train_loader, optimizer, epoch):
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
        
        train_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        train_correct += pred.eq(target.view_as(pred)).sum().item()
    
    avg_loss = train_loss / len(train_loader.dataset)
    avg_accuracy = train_correct / len(train_loader.dataset)
    

        
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    return avg_loss, avg_accuracy


def conv_test_step(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            test_correct += pred.eq(target.view_as(pred)).sum().item()
    

    test_loss /= len(test_loader.dataset)
    avg_loss_val = test_loss
    avg_accuracy_val = test_correct / len(test_loader.dataset)

    
        

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, len(test_loader.dataset),
        100. * test_correct / len(test_loader.dataset)))
    
    return avg_loss_val, avg_accuracy_val


# In[5]:


names = ['airplane', 'onion', 'apple', 'pineapple', 'ant', 'banana', 'ambulance', 'angel', 'cat', 'cow', 'broccoli', 'bus']
amount = 1000
device = torch.device("cpu")
model = Net(len(names))
X_train, Y_train, X_test, Y_test = data_load(names, amount)        
train_loader  = DataLoader(TensorDataset(X_train,Y_train),
                        batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, Y_test),
                        batch_size=32, shuffle=False)  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

liveloss = PlotLosses()


# In[6]:


def conv_train_model(epoch): #stosuje wcześniej zdefiniowane funkcje, argument: 
                                   #epoch-ilość przejśc przez dane treningowe, 
                                   #names-tablica nazw klas,
                                   #amount-ilość danych,
    for epoch in range(epoch):

        avg_loss, avg_accuracy = conv_train_step(model, device, train_loader, optimizer, epoch)  
        avg_loss_val, avg_accuracy_val = conv_test_step(model, device, test_loader)
    
        liveloss.update({
            'val_log loss': avg_loss_val,
            'val_accuracy': avg_accuracy_val,
            'log loss': avg_loss,
            'accuracy': avg_accuracy,
        })
    
        liveloss.draw()
        
    torch.save(model.state_dict(), '/home/user/ex/notebooks/model_state_dict/model_data.pt')


# In[7]:


#conv_train_model(10)


# In[8]:


def print_kernels(names):
    #kernels = conv_train_model()
    #model = Net(len(names))   
    #model.load_state_dict(torch.load('/home/user/ex/notebooks/model_state_dict/model_data.pt'))
    #model.eval()
    kernels = model.conv1.weight.detach().cpu().numpy()
    column = 10
    rows = np.int(kernels.shape[0]/column)
    fig, axs = plt.subplots(rows, column, figsize=(column, rows))
    j = 0
    axs = axs.reshape(rows, column)
    for i in range(rows):
        for k in range(column):
            ax = axs[i, k]
            ax.imshow(kernels[j,0])
            #plt.colorbar()
            ax.axis('off')
            j = j+1


# In[9]:


#print_kernels(names)


# In[10]:


def print_false_predictions(names, how_many):
    res = model(X_test).max(dim=1)[1]
    eq = (Y_test != res)
    good = Y_test[eq]
#class_names = ['airplane', 'onion', 'apple', 'pineapple', 'ant', 'banana', 'ambulance', 'angel']
    test_fail = X_test[eq] #Tylko złe wyniki
    predictions = F.softmax(model(test_fail), dim = 1)

    rows = how_many
    fig, axs = plt.subplots(rows, 2, figsize=(8, 1.5 * rows))
    for i in range(rows):
        ax = axs[i,0]
        idx = np.random.randint(len(test_fail))
    
        ax.imshow(test_fail[idx,0].reshape(28,28),
              cmap='Greys', interpolation='none')
    
        ax.axis('off')
        
        pd.Series(predictions[idx].detach().numpy(), index=names).plot('barh', ax=axs[i,1], xlim=[0,1], title=names[good[idx]])
        
    plt.tight_layout()


# In[11]:


#print_false_predictions(names, 16)


# In[12]:


def show_layers_action():
#Rysunek losowy
    id_rys = np.random.randint(len(X_train))
    print("Rysunek numer", id_rys,":")
    plt.imshow(X_train[id_rys].reshape(28, 28), cmap='Greys')
    plt.show()

#Rysuje wszystkie kanały po przejściu pierwszej warstwy
    Conv_l = F.relu(F.max_pool2d(model.conv1(X_train), 2))
    print("Pierwsza warstwa:")
    rows_r = 1
    columns_r = Conv_l.shape[1]
    fig, axs = plt.subplots(rows_r, columns_r, figsize=(columns_r, rows_r))
    axs = axs.reshape(rows_r, columns_r)
    for kernel_id in range(columns_r):
    #letters = Conv_l[Y_train == class_id]
        for i in range(rows_r):
            ax = axs[i, kernel_id]
            ax.imshow(Conv_l[id_rys,kernel_id].reshape(12,12).detach().numpy(),
                  cmap='Greys', interpolation='none')
            ax.axis('off')
        
    plt.show()
        
#Rysuje wszystkie kanały po przejściu pierwszej i drugiej warstwy
    Conv_l2 = F.relu(F.max_pool2d(model.conv2_drop(model.conv2(Conv_l)), 2))
    print("Druga warstwa:")
    rows = 1
    columns = Conv_l2.shape[1]
    fig, axs = plt.subplots(rows, columns, figsize=(columns, rows))
    axs = axs.reshape(rows,columns)

    for kernel_id in range(columns):
    #letters = Conv_l[Y_train == class_id]
        for i in range(rows):
            ax = axs[i, kernel_id]
            ax.imshow(Conv_l2[id_rys,kernel_id].reshape(4,4).detach().numpy(),
                  cmap='Greys', interpolation='none')
            ax.axis('off')
    plt.show()


# In[13]:


#show_layers_action()


# In[14]:


def confusion_matrix():
    class_num = len(names)
    conf_m = torch.tensor(np.array((class_num)**2*[0]).reshape(class_num,class_num), dtype=torch.int32)
    p = 0    
    res = model(X_test).max(dim=1)[1]
    #iterator = torch.tensor([pltest_t.size(0), ontest_t.size(0), aptest_t.size(0), pinetest_t.size(0),
                         #antest_t.size(0), batest_t.size(0), amtest_t.size(0), angtest_t.size(0)])
    for k in range(class_num):
        for j in range(class_num):
            conf_m[k][j] = 0
            for i in range(amount):
                if (res[p + i] == j):
                    conf_m[k][j] += 1
        p += amount

    predictions = []
    
    for i in names:
        predictions.append('predicted '+i)
    
    fig, ax = plt.subplots()
    im = ax.imshow(conf_m)

# Labels place
    ax.set_xticks(np.arange(len(predictions)))
    ax.set_yticks(np.arange(len(names)))
# Labels
    ax.set_xticklabels(predictions)
    ax.set_yticklabels(names)

# Rotation and alignment of the labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
    for i in range(len(names)):
        for j in range(len(predictions)):
            text = ax.text(j, i, conf_m[i][j].item(), #taking a value out of the tensor
                       ha="center", va="center", color="w")

    ax.set_title("Confusion matrix")
#fig.tight_layout() #Making the image smaller
    plt.show()


# In[15]:


#confusion_matrix()


# In[16]:


def image_confusion_matrix():
    
    res = model(X_test).max(dim=1)[1]
    class_num = len(names)
    conf_pic = conf_pic = torch.tensor(np.zeros((len(names)*28)**2).reshape((len(names),len(names),28,28)).astype('float32'))
    p = 0
    
    
    for k in range(class_num):
        for j in range(class_num):
            conf_pic[k][j] = 0
            for i in range(amount):
                if (res[p + i] == j):
                    conf_pic[k][j] = X_test[p+i][0]
                    break;
        p += amount
    
    predictions = []
    for i in names:
        predictions.append('predicted '+i)

    rows = np.size(names)
    columns = np.size(predictions)


    fig, axs = plt.subplots(rows, columns, figsize=(2*(columns), 2*rows))
    axs = axs.reshape(rows, columns)

#Place for pictures
    for ax, predictions in zip(axs[0], predictions):
        ax.set_title(predictions)

    fig.tight_layout()

#Drawing the first picture that fits 
    for i in range(columns):
        for j in range(rows):
            ax = axs[j, i]
            ax.imshow(conf_pic[j,i],
                      cmap='Greys', interpolation='none')
            ax.axis('off')
                
    plt.show()


# In[17]:


#image_confusion_matrix()


# In[ ]:





# In[ ]:




