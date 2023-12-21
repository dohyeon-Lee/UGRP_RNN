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
sequence_length = 8
batch_size = 20
train_datasize = 30
test_datasize = 5
database = data_loader(train_datasize=train_datasize, test_datasize=test_datasize, sequence_length=sequence_length, batch_size=batch_size, device=device)

## models
input_size = database.train_input_seq[0].size(2) #3
num_layers = 2
hidden_size = 8
model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=sequence_length,
                   num_layers=num_layers,
                   device=device).to(device)
criterion = nn.MSELoss()

## training
lr = 1e-3
num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(database.train_loader[0])
for epoch in range(num_epochs):
    running_loss = 0.0
    train_loader = database.train_loader[random.randrange(0,train_datasize)]
    
    b=0
    last_out = torch.tensor([[0.,0.]]).to(device)
    last_input_buffer = torch.ones([1, sequence_length, 3])

    for data in train_loader:
        out = torch.tensor([[0.,0.]]).to(device)
        out_save = torch.tensor([[0.,0.]]).to(device)
        target_save = torch.tensor([[0.,0.]]).to(device)
        seq, target = data #seq, target은 20개, seq는 3차원 배열 20*8*3 , target은 2차원 배열 20*2
        for i in range(0,seq.shape[0]):#seq.shape[0] ==20
            
            seq_in_loop = seq[i][:][:] #한번할때마다 8*3 짜리를 보겟다
            target_in_loop = target[i][:]
    
            if b == 0:#첫번째 배치에서만 실행
                out = model(seq_in_loop.unsqueeze(0)) #처음 8*3짜리 모델 들어감
               
                input_buffer = seq_in_loop.unsqueeze(0)

                out_save = out
                target_save = target_in_loop.unsqueeze(0)
    
                b=1
            else:
                
                input_buffer_u = seq_in_loop.unsqueeze(0) #다음 8*3짜리
                
                if (i==0): #두번째 배치의 첫번째 데이터, out이 없음!
                    input_buffer_pop = last_input_buffer[-1][1:][:] #제일 첫번재 꺼 뺀 거
                    insert = torch.cat((input_buffer_u[-1][-1][0].unsqueeze(0), last_out.squeeze(0))) #squeeze(0)에서 바꿈
                else:
                    input_buffer_pop = input_buffer[-1][1:][:] #제일 첫번재 꺼 뺀 거
                    insert = torch.cat((input_buffer_u[-1][-1][0].unsqueeze(0), out.squeeze(0)))
                
                input_buffer = torch.cat((input_buffer_pop, insert.unsqueeze(0)))
                input_buffer = input_buffer.unsqueeze(0)

                out = model(input_buffer)
               
                if (i==0):
                    out_save = out.clone().detach()
                    target_save = target_in_loop.unsqueeze(0)
               
                    
                elif (i==seq.shape[0]-1):
                    
                    target_save = torch.cat((target_save, target_in_loop.unsqueeze(0)), dim=0)
                    out_save = torch.cat((out_save, out), dim=0)
                    last_out = out.clone().detach()
                    last_input_buffer = input_buffer.clone().detach()
                  
                else:
                    out_save = torch.cat((out_save, out), dim=0)
                  
                    target_save = torch.cat((target_save, target_in_loop.unsqueeze(0)), dim=0)
        loss = criterion(out_save, target_save)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
    if epoch % 10 == 0:
      print('[epoch: %d] loss: %.4f'%(epoch, running_loss / n))

plt.figure()
plt.plot(loss_graph)
plt.show()

## model wieght save
PATH = "model/model_dict.pt"
torch.save(model.state_dict(), PATH)




