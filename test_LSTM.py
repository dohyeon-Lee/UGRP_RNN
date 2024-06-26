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
from LSTM import LSTM
from data_loader import data_loader

CPU = 0

if CPU == 1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

database = data_loader(num_epochs=200, device=device)

## models
input_size = 1
num_layers = 2
hidden_size = 8
model = LSTM(input_size=input_size,
                   hidden_size=hidden_size,
                  
                   num_layers=num_layers,
                   device=device).to(device)

# PATH = "weight/trace_direct_dict_real_batch_"+str(database.batch_size)+"_epoch_"+str(database.num_epochs)+"_loss123.pt"
PATH = "model/LSTM_ugrp_traced_model_loss123_200_gpu_dataset4_seq500.pt"
# model.load_state_dict(torch.load(PATH))
model = torch.load(PATH)
model.eval()

def plotting(model, test_loader):

  with torch.no_grad():
    theta = []
    theta_dot = []
    target_theta = []
    target_theta_dot = []
    hn = torch.zeros(num_layers, 1, hidden_size).to(device)
    cn = torch.zeros(num_layers, 1, hidden_size).to(device)
    for data in test_loader:
      seq_batch, target_batch = data
      
      for seq in seq_batch:
        # input_ = torch.zeros(1, 1, 1).to(device) ## TODO check
        out, (hn,cn) = model(seq.unsqueeze(0), (hn,cn))
        theta.extend(out[:,:,0].view([-1,]).cpu().numpy().tolist())
        theta_dot.extend(out[:,:,1].view([-1,]).cpu().numpy().tolist())           

      target_theta.extend(target_batch[:,:,0].view([-1,]).cpu().numpy().tolist())
      target_theta_dot.extend(target_batch[:,:,1].view([-1,]).cpu().numpy().tolist())

  time = np.linspace(0,1200,1200*50)
  print(len(target_theta))
  for i in range(0,len(target_theta)):
    target_theta[i] += np.pi/2 

  
  
  plt.figure(figsize=(20,5))
  plt.plot(time[:len(target_theta)], target_theta, '--') #theta
  plt.plot(time[:len(theta)],theta , 'b', linewidth=0.6)
  plt.legend([r'$\theta$',r'$\hat{\theta}$'])
  plt.xlabel('t(s)')
  plt.ylabel(r'$\theta$')
  plt.grid(True)

  plt.figure(figsize=(20,5))
  plt.plot(time[:len(target_theta_dot)],target_theta_dot,'r' '--') #theta_dot  
  plt.plot(time[:len(theta_dot)], theta_dot, 'r', linewidth=0.6)
  plt.grid(True)

  plt.legend([r'$\dot{\theta}$', r'$\hat{\dot{\theta}}$'])
  plt.xlabel('t(s)')
  plt.ylabel(r'$\dot{\theta}$')
  plt.show()

plotting(model, database.test_loader)