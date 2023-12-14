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
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')


## parameters, dataset

database = data_loader(device=device)

## models
input_size = 1 
num_layers = 2
hidden_size = 8
model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   
                   num_layers=num_layers,
                   device=device).to(device)
criterion = nn.MSELoss()

## training
lr = 1e-3
num_epochs = database.num_epochs
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(database.train_loader)

for epoch in range(num_epochs):
    running_loss = 0.0
    for data in database.train_loader:
        
        # loss 1 calculate
        seq_batch, target_batch = data 
        target_batch[:,:,0] += np.pi/2
        h0 = torch.zeros(num_layers, seq_batch.size()[0], hidden_size).to(device) # 초기 hidden state 설정하기.
        out, _ = model(seq_batch, h0)
        loss = criterion(out, target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
    if epoch % 10 == 0:
        print('[epoch: %d] loss: %.4f '%(epoch, running_loss / n))
plt.figure()
plt.plot(loss_graph)
plt.show()

## model weight save
PATH = "model/train_direct_dict_batch_"+str(database.batch_size)+"_epoch_"+str(num_epochs)+".pt"
torch.save(model.state_dict(), PATH)




