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

train_datasize = 1
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
num_epochs = 500
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(database.train_loader[0])

for epoch in range(num_epochs):
    running_loss = 0.0
    train_loader = database.train_loader[random.randrange(0,train_datasize)]
    for batch_idx, data in enumerate(train_loader):
        
        seq_batch, target_batch = data #seq, target은 20개, seq는 3차원 배열 20*8*3 , target은 2차원 배열 20*2
        target_batch[:,0] += np.pi/2

        for seq_idx, seq in enumerate(seq_batch):
           if seq_idx == 0:
                out = model(seq.unsqueeze(0))
                seq_buffer = seq.unsqueeze(0)
                out_batch = out
           else:
                seq_buffer = seq_buffer[:,1:,:] # pop
                seq_buffer = torch.cat(seq[:,:,0])
                print(seq_idx)

              

        out = model(seq_batch)
        loss = criterion(out, target_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
    if epoch % 10 == 0:
      print('[epoch: %d] loss: %.4f'%(epoch, running_loss / n))

plt.figure()
plt.plot(loss_graph)
plt.show()

## model wieght save
PATH = "model/train_direct_dict_sl"+str(database.sequence_length)+".pt"
torch.save(model.state_dict(), PATH)




