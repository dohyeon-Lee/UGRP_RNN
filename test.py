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
import argparse

parser = argparse.ArgumentParser(
    prog="test PIRNN",
    description="test",
    epilog="2023_UGRP"
)
parser.add_argument("--dataset", type=str, help="dataset file name",required=True)
args = parser.parse_args()

# read yaml file
with open('setting.yaml') as f:
    param = yaml.full_load(f)

CPU = param['device']
Hz = param['Hz']
if CPU == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

database = data_loader(num_epochs=param['learning_param']['epoch'], device=device, dataset=args.dataset)

input_size = param['model_param']['input_size'] 
num_layers = param['model_param']['num_layer']
hidden_size = param['model_param']['hidden_size']

## use with weight
# model = VanillaRNN(input_size=input_size,
#                    hidden_size=hidden_size,
                  
#                    num_layers=num_layers,
#                    device=device).to(device)
# PATH = "weight/trace_direct_dict_real_batch_"+str(database.batch_size)+"_epoch_"+str(database.num_epochs)+"_loss123.pt"
# model.load_state_dict(torch.load(PATH))
# model.eval()

## use with traced model
PATH = "extracted_model/model_dataset4.pt"
model = torch.load(PATH)
model.eval()

def lpf(x_k,  y_km1, Ts, tau) :
  y_k = (tau * y_km1 + Ts * x_k) / (Ts + tau)
  return y_k
   

def plotting(model, test_loader):
  
  with torch.no_grad():
    theta = []
    theta_dot = []
    target_theta = []
    target_theta_dot = []
    hn = torch.zeros(num_layers, 1, hidden_size).to(device)
    for data in test_loader:
      seq_batch, target_batch = data
      
      for seq in seq_batch:
        out, hn = model(seq.unsqueeze(0), hn)
        theta.extend(out[:,:,0].view([-1,]).cpu().numpy().tolist())
        theta_dot.extend(out[:,:,1].view([-1,]).cpu().numpy().tolist())           

      target_theta.extend(target_batch[:,:,0].view([-1,]).cpu().numpy().tolist())
      target_theta_dot.extend(target_batch[:,:,1].view([-1,]).cpu().numpy().tolist())

  time = np.linspace(0,1200,1200*100)
  print(len(target_theta))
  for i in range(0,len(target_theta)):
    target_theta[i] += np.pi/2 
  
  f_cut = 10 #0.5
  tau = 1/(2*np.pi*f_cut)
  filtered_theta = []
  before_lpf_theta = 0
  filtered_thetadot = []
  before_lpf_thetadot = 0
  for i in range(0, len(theta)):
    lpf_theta = lpf(theta[i], before_lpf_theta, 1/Hz, tau)
    before_lpf_theta = lpf_theta
    filtered_theta.append(lpf_theta)

  for i in range(0, len(theta_dot)):
    lpf_thetadot = lpf(theta_dot[i], before_lpf_thetadot, 1/Hz, tau)
    before_lpf_thetadot = lpf_thetadot
    filtered_thetadot.append(lpf_thetadot)

  plt.subplots(constrained_layout=True)
  plt.subplot(211)
  plt.plot(time[:len(target_theta)], target_theta, '--') #theta
  plt.plot(time[:len(theta)],theta , 'b', linewidth=0.6)
  plt.plot(time[:len(filtered_theta)],filtered_theta , 'g', linewidth=0.6)
  plt.title(r'$\theta$')
  plt.legend([r'$\theta$',r'$\hat{\theta}$', r'$\hat{\theta}_{lpf}$'])
  plt.xlabel('t(s)')
  plt.ylabel(r'$\theta$')
  plt.grid(True)

  plt.subplot(212)
  plt.plot(time[:len(target_theta_dot)],target_theta_dot,'r' '--') #theta_dot  
  plt.plot(time[:len(theta_dot)], theta_dot, 'r', linewidth=0.6)
  plt.plot(time[:len(filtered_thetadot)],filtered_thetadot , 'g', linewidth=0.6)
  plt.title(r'$\dot{\theta}$')
  plt.legend([r'$\dot{\theta}$', r'$\hat{\dot{\theta}}$', r'$\hat{\dot{\theta}}_{lpf}$'])
  plt.xlabel('t(s)')
  plt.ylabel(r'$\dot{\theta}$')
  plt.grid(True)
  plt.show()
  
plotting(model, database.test_loader)