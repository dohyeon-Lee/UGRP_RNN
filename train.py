import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import yaml
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
from network.VanillaRNN import VanillaRNN
from network.data_loader import data_loader
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser(
    prog="training PIRNN",
    description="training",
    epilog="2023_UGRP"
)
parser.add_argument("--dataset", type=str, help="dataset file name",required=True)
args = parser.parse_args()

# read yaml file
with open('setting.yaml') as f:
    param = yaml.full_load(f)

MODE = param['PIRNN_mode']
CPU = param['device']
Hz = param['Hz']

writer = SummaryWriter()
if CPU == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f'{device} is available')


## parameters, dataset
database = data_loader(num_epochs=param['learning_param']['epoch'], device=device, dataset=args.dataset)  

## models
input_size = param['model_param']['input_size'] 
num_layers = param['model_param']['num_layer']
hidden_size = param['model_param']['hidden_size']
model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   num_layers=num_layers,
                   device=device).to(device)
criterion = nn.MSELoss()

## training
lr = param['learning_param']['learning_rate']
num_epochs = database.num_epochs
sequence_length = database.sequence_length
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_graph = [] # loss for draw graph
n = len(database.train_loader)

for epoch in range(num_epochs):
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0
    for data in database.train_loader:
        
        # loss 1 calculate
        seq_batch, target_batch = data 
        target_batch[:,:,0] += np.pi/2
        h0 = torch.zeros(num_layers, seq_batch.size()[0], hidden_size).to(device) # inital hidden state
        out, _ = model(seq_batch, h0)
        loss1 = criterion(out, target_batch)
        
        if MODE == 2 or MODE == 3:
        # loss 2 calculate (use only 0th batch)
            dt = 1/Hz 
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
        
        if MODE == 3:
            # loss 3 calculate (use only 0th batch)
            g = param['physics_param']['g']
            L = param['physics_param']['L']  
            m = param['physics_param']['m']
            k = param['physics_param']['k'] # coefficients c/m
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

        if MODE == 1:
            loss1 = param['PIRNN_physics_param']['loss1_gain']*loss1 
            running_loss1 += loss1.item()
            loss = loss1
        elif MODE == 2:
            loss1 = param['PIRNN_physics_param']['loss1_gain']*loss1 
            loss2 = param['PIRNN_physics_param']['loss2_gain']*loss2 
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            loss = loss1 + loss2
        elif MODE == 3:
            loss1 = param['PIRNN_physics_param']['loss1_gain']*loss1 
            loss2 = param['PIRNN_physics_param']['loss2_gain']*loss2 
            loss3 = param['PIRNN_physics_param']['loss3_gain']*loss3 
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss3 += loss3.item()
            loss = loss1 + loss2 + loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # logging (tensorboard)
    running_loss = running_loss1 + running_loss2 + running_loss3
    writer.add_scalar("tot_Loss/epoch", running_loss/n, epoch)
    writer.add_scalar("Loss1/epoch", running_loss1/n, epoch)
    writer.add_scalar("Loss2/epoch", running_loss2/n, epoch)
    writer.add_scalar("Loss3/epoch", running_loss3/n, epoch)
    loss_graph.append(running_loss / n) 
    
    # logging (terminal)
    if epoch % 10 == 0:
        print('[epoch: %d] loss: %.4f '%(epoch, running_loss / n))

#ploting
writer.close()
plt.figure()
plt.plot(loss_graph)
plt.show()


## model weight save && tracing model
tracing_device = param['tracing_device']

if MODE == 1:
    PATH = "testtraced_model_loss1_epoch"+str(num_epochs)+"_"+tracing_device+"_withcontrol_seq"+str(sequence_length)+"_Hz_"+str(Hz)+".pt"
elif MODE == 2:
    PATH = "testtraced_model_loss12_epoch"+str(num_epochs)+tracing_device+"_withcontrol_seq"+str(sequence_length)+"_Hz_"+str(Hz)+".pt"
elif MODE == 3:
    PATH = "testtraced_model_loss123_epoch"+str(num_epochs)+tracing_device+"_withcontrol_seq"+str(sequence_length)+"_Hz_"+str(Hz)+".pt"
torch.save(model.state_dict(), "weight/"+PATH)

traced_model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   num_layers=num_layers,
                   device=tracing_device).to(tracing_device)

hn = torch.rand(num_layers, 1, hidden_size).to(tracing_device)
example = torch.rand(1).unsqueeze(0).unsqueeze(0).to(tracing_device)

traced_model.load_state_dict(torch.load("weight/"+PATH))
traced_script_module = torch.jit.trace(traced_model, (example, hn))
traced_script_module.save("extracted_model/"+PATH)




