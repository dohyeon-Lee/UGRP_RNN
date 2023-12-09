#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')


traindata = pd.read_csv('train/train0.csv')
train_input = traindata[['u(t)', 'theta', 'theta_dot']].values
train_input = train_input[:-1]  
train_output = traindata[['theta','theta_dot']].values


testdata = pd.read_csv('train/train0.csv')
test_input = testdata[['u(t)', 'theta', 'theta_dot']].values
test_input = test_input[:-1]  
test_output = testdata[['theta','theta_dot']].values

# 뭉텅이로 묶어주는 함수
def seq_data(input, output, sequence_length):

    x_seq = []
    y_seq = []
    
    for i in range(len(input)-sequence_length):
        x_seq.append(input[i:i+sequence_length])
        y_seq.append(output[i+sequence_length])        
    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device)

sequence_length = 8
train_input_seq, train_output_seq = seq_data(train_input,train_output,sequence_length)
train = torch.utils.data.TensorDataset(train_input_seq, train_output_seq)
batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
input_size = train_input_seq.size(2) #3
num_layers = 2
hidden_size = 8

class VanillaRNN(nn.Module):

  def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
    super(VanillaRNN, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size * sequence_length, 2)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) # 초기 hidden state 설정하기.
    out, _ = self.rnn(x, h0) # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
    out = out.reshape(out.shape[0], -1) # many to many 전략
    out = self.fc(out)
    return out
model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=sequence_length,
                   num_layers=num_layers,
                   device=device).to(device)
criterion = nn.MSELoss()

lr = 1e-4
num_epochs = 300
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(train_loader)


# In[8]:


for epoch in range(num_epochs):
    running_loss = 0.0
    for data in train_loader:
        b=0
        
        out_save = torch.tensor([[0.,0.]])
        target_save = torch.tensor([[0.,0.]])
        seq, target = data #seq, target은 20개, seq는 3차원 배열 20*8*3 , target은 2차원 배열 20*2
        for i in range(0,seq.shape[0]):
            
            seq_in_loop = seq[i][:][:]
            target_in_loop = target[i][:]
    
            if b == 0:
                out = model(seq_in_loop.unsqueeze(0))
                input_buffer = seq_in_loop.unsqueeze(0)
                out_save = out
                target_save = target_in_loop.unsqueeze(0)
                #print(target_save)
                out_list_graph = out
                b=1
            else:
                input_buffer_u = seq_in_loop.unsqueeze(0)
                input_buffer_pop = input_buffer[-1][1:][:]

                insert = torch.cat((input_buffer_u[-1][-1][0].unsqueeze(0), out.squeeze(0)))
                
                input_buffer = torch.cat((input_buffer_pop, insert.unsqueeze(0)))
                input_buffer = input_buffer.unsqueeze(0)

                out = model(input_buffer)
               
                if (i==0):
                    out_save = out
                    target_save = target_in_loop.unsqueeze(0)
                else:
                    out_save = torch.cat((out_save, out), dim=0)
                    target_save = torch.cat((target_save, target_in_loop.unsqueeze(0)), dim=0)
                out_list_graph = torch.cat((out_list_graph, out), dim = 0)
#         print(target_save.shape)
#         print(out_save.shape)
        loss = criterion(out_save, target_save)
        running_loss += loss.item()
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         print("error")
    loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
    if epoch % 10 == 0:
      print('[epoch: %d] loss: %.4f'%(epoch, running_loss/n))

plt.figure()
plt.plot(loss_graph)
plt.show()



# In[9]:


test_input_seq, test_output_seq = seq_data(test_input, test_output, sequence_length)

test = torch.utils.data.TensorDataset(test_input_seq, test_output_seq)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

def plotting(test_loader, actual_theta, actual_theta_dot):
  out_list = torch.tensor([[0,0]])
  with torch.no_grad():
    test_pred = []
    a = 0

    for data in test_loader:
      seq, target = data
      for sample in seq:
        if a == 0:
            out = model(sample.unsqueeze(dim=0))
            input_buffer = sample.unsqueeze(dim=0)
            out_list = out
            a = 1
        else:
            input_buffer_u = sample.unsqueeze(dim=0)
            input_buffer_pop = input_buffer[-1][1:][:] # 맨 윗줄 없애는거

            insert = torch.cat((input_buffer_u[-1][-1][0].unsqueeze(0),out.squeeze(0)))
            input_buffer = torch.cat((input_buffer_pop, insert.unsqueeze(0)))
            input_buffer = input_buffer.unsqueeze(0)

            out = model(input_buffer)
            out_list = torch.cat((out_list, out), dim = 0)
                 
  time = np.linspace(0,20,300)
  out_list = np.array(out_list.cpu())

  plt.figure(figsize=(20,5))

  plt.plot(time[:292], actual_theta, '--') #theta
  plt.plot(time[:290],out_list[0:-1,0], 'b', linewidth=0.6)
  plt.legend([r'$\theta$',r'$\hat{\theta}$'])
  plt.xlabel('t(s)')
  plt.ylabel(r'$\theta$')
  plt.grid(True)

  plt.figure(figsize=(20,5))
  plt.plot(time[:292],actual_theta_dot,'r' '--') #theta_dot  
  plt.plot(time[:290],out_list[0:-1,1], 'r', linewidth=0.6)
  plt.grid(True)

  plt.legend([r'$\dot{\theta}$', r'$\hat{\dot{\theta}}$'])
  plt.xlabel('t(s)')
  plt.ylabel(r'$\dot{\theta}$')
  plt.show()

plotting(test_loader, testdata['theta'][sequence_length:],testdata['theta_dot'][sequence_length:])


# In[ ]:




