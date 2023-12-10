import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from tqdm import tqdm
import random
from VanillaRNN import VanillaRNN
from data_loader import data_loader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')


## parameters, dataset

train_datasize = 30
test_datasize = 1
database = data_loader(train_datasize=train_datasize, test_datasize=test_datasize, device=device)

## models
input_size = database.train_input_seq[0].size(2) #3
num_layers = 2
hidden_size = 8
model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=database.sequence_length,
                   num_layers=num_layers,
                   device=device).to(device)
criterion = nn.MSELoss()

## training
lr = 1e-3
num_epochs = 300
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(database.train_loader[0])

for epoch in range(num_epochs):
    running_loss1 = 0.0
    running_loss2 = 0.0
    train_loader = database.train_loader[random.randrange(0,train_datasize)]
    for data in train_loader:
        
        # loss 1 calculate
        seq, target = data #seq, target은 20개, seq는 3차원 배열 20*8*3 , target은 2차원 배열 20*2
        target[:,0] += np.pi/2
        out = model(seq)
        loss1 = criterion(out, target)
        
        # loss 2 calculate
        dt = 20/300.
        expected_theta_dot = torch.zeros(out[:,0].shape[0]).to(device)
        bbefore_theta = 0
        before_theta = 0
        for idx, theta in enumerate(out[:,0]):
            if idx > 1:
                theta_dot = (theta - bbefore_theta)/dt 
                expected_theta_dot[idx-1] = theta_dot
            bbefore_theta = before_theta
            before_theta = theta
        
        loss2 = criterion(expected_theta_dot[1:-1], target[1:-1,1])

        loss = 100*loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
    running_loss = running_loss1 + running_loss2
    loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
    if epoch % 10 == 0:
      print('[epoch: %d] loss1: %.4f loss2: %.4f'%(epoch, running_loss1 / n, running_loss2 / n))

plt.figure()
plt.plot(loss_graph)
plt.show()

## model wieght save
PATH = "model/train_direct_dict_sl"+str(database.sequence_length)+".pt"
torch.save(model.state_dict(), PATH)




