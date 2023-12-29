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

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'{device} is available')


## parameters, dataset

database = data_loader(num_epochs=200, device=device) #200 

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
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0
    for data in database.train_loader:
        
        # loss 1 calculate
        seq_batch, target_batch = data 
        target_batch[:,:,0] += np.pi/2
        h0 = torch.zeros(num_layers, seq_batch.size()[0], hidden_size).to(device) # 초기 hidden state 설정하기.
        out, _ = model(seq_batch, h0)
        loss1 = criterion(out, target_batch)
        
        # loss 2 calculate (use only 0th batch)
        dt = 600/30000.
        expected_theta_dot = torch.zeros(out.shape[1]).to(device)
        bbefore_theta = 0
        before_theta = 0
        for idx, theta in enumerate(out[0,:,0]):
            if idx > 1:
                theta_dot = (theta - bbefore_theta)/(2*dt) 
                expected_theta_dot[idx-1] = theta_dot
            bbefore_theta = before_theta
            before_theta = theta       
        loss2 = criterion(expected_theta_dot[1:-1], target_batch[0,1:-1,1])
        
        # loss 3 calculate (use only 0th batch)
        # g = 9.8 
        # L = 1.5 
        # m = 1.0
        # k = 8 # coefficients c/m
        g = 9.8 
        L = 1./21.  
        m = 1.0 
        k = 0.58 # coefficients c/m
        x_ddot = seq_batch[0,:-1,0]
        true_theta_ddot = -k*target_batch[0,:-1,1]*torch.cos(target_batch[0,:-1,0])-(g/L)*torch.sin(target_batch[0,:-1,0])-(x_ddot/L)*torch.cos(target_batch[0,:-1,0])
        expected_theta_ddot = torch.zeros(out.shape[1]).to(device)
        bbefore_theta_dot = 0
        before_theta_dot = 0
        for idx, theta_dot in enumerate(out[0,:,1]):
            if idx > 1:
                theta_ddot = (theta_dot - bbefore_theta_dot)/(2*dt) 
                expected_theta_ddot[idx-1] = theta_ddot
            bbefore_theta_dot = before_theta_dot
            before_theta_dot = theta_dot       
        loss3 = criterion(expected_theta_ddot[1:-1], true_theta_ddot[1:])

        loss1 = 100*loss1 # 100
        loss2 = 0.01*loss2 # 0.01
        loss3 = 0.001*loss3 # 0.001

        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        running_loss3 += loss3.item()
        

        loss = loss1 + loss2 + loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    running_loss = running_loss1 + running_loss2 + running_loss3
    writer.add_scalar("tot_Loss/epoch", running_loss/n, epoch)
    writer.add_scalar("Loss1/epoch", running_loss1/n, epoch)
    writer.add_scalar("Loss2/epoch", running_loss2/n, epoch)
    writer.add_scalar("Loss3/epoch", running_loss3/n, epoch)
    loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
    if epoch % 10 == 0:
        print('[epoch: %d] loss: %.4f '%(epoch, running_loss / n))
writer.close()
plt.figure()
plt.plot(loss_graph)
plt.show()

## model weight save
PATH = "weight/trace_direct_dict_real_batch_"+str(database.batch_size)+"_epoch_"+str(num_epochs)+"_loss123.pt"
torch.save(model.state_dict(), PATH)

hn = torch.rand(num_layers, 1, hidden_size).to(device)
example = torch.rand(1).unsqueeze(0).unsqueeze(0).to(device)

traced_script_module = torch.jit.trace(model, (example, hn))
traced_script_module.save("model/ugrp_traced_model.pt")




