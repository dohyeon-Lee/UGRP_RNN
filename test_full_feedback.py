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

def seq_data(input, output, sequence_length):

    x_seq = []
    y_seq = []
    
    for i in range(len(input)-sequence_length):
        x_seq.append(input[i:i+sequence_length])
        y_seq.append(output[i+sequence_length])        
    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_size = 3#3
num_layers = 2
hidden_size = 8
sequence_length = 8
batch_size = 1
model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=sequence_length,
                   num_layers=num_layers,
                   device=device).to(device)
PATH = "model/train_direct_dict.pt"
model.load_state_dict(torch.load(PATH))
model.eval()

testdata = pd.read_csv('train/train0.csv')
test_input = testdata[['u(t)', 'theta', 'theta_dot']].values
test_input = test_input[:-1]  
test_output = testdata[['theta','theta_dot']].values

test_input_seq, test_output_seq = seq_data(test_input, test_output, sequence_length)
test = torch.utils.data.TensorDataset(test_input_seq, test_output_seq)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

def plotting(model, test_loader, actual_theta, actual_theta_dot):
  out_list = torch.tensor([[0,0]]).to(device)
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

plotting(model, test_loader, testdata['theta'][sequence_length:],testdata['theta_dot'][sequence_length:])