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

PATH = "model/train_direct_dict_batch_"+str(database.batch_size)+".pt"
model.load_state_dict(torch.load(PATH))
model.eval()

testdata = pd.read_csv('test/test4.csv')
test_input = testdata[['u(t)', 'theta', 'theta_dot']].values
test_input = test_input[:-1]  
test_output = testdata[['theta','theta_dot']].values

test_input_seq, test_output_seq = database.seq_data(test_input, test_output, database.sequence_length)
test = torch.utils.data.TensorDataset(test_input_seq, test_output_seq)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=database.batch_size, shuffle=False)

def plotting(model, test_loader, actual_theta, actual_theta_dot):
  out_list = torch.tensor([[0,0]]).to(device)
  with torch.no_grad():
    theta = []
    theta_dot = []
    for data in test_loader:
      seq, target = data
      out = model(seq)
      theta.extend(out[:,0].squeeze(0).cpu().numpy().tolist())
      theta_dot.extend(out[:,1].squeeze(0).cpu().numpy().tolist())           
  time = np.linspace(0,20,300)

  out_list = np.array(out_list.cpu())
  actual_theta[:] += np.pi/2
  plt.figure(figsize=(20,5))
  plt.plot(time[:len(actual_theta)], actual_theta, '--') #theta
  plt.plot(time[:len(theta)],theta , 'b', linewidth=0.6)
  plt.legend([r'$\theta$',r'$\hat{\theta}$'])
  plt.xlabel('t(s)')
  plt.ylabel(r'$\theta$')
  plt.grid(True)

  plt.figure(figsize=(20,5))
  plt.plot(time[:len(actual_theta_dot)],actual_theta_dot,'r' '--') #theta_dot  
  plt.plot(time[:len(theta_dot)], theta_dot, 'r', linewidth=0.6)
  plt.grid(True)

  plt.legend([r'$\dot{\theta}$', r'$\hat{\dot{\theta}}$'])
  plt.xlabel('t(s)')
  plt.ylabel(r'$\dot{\theta}$')
  plt.show()

plotting(model, test_loader, testdata['theta'][database.sequence_length:],testdata['theta_dot'][database.sequence_length:])