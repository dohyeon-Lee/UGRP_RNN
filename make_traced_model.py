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

MODE = 3
CPU = 1

if CPU == 1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

hn = torch.rand(num_layers, 1, hidden_size).to(device)
example = torch.rand(1).unsqueeze(0).unsqueeze(0).to(device)

PATH = "extracted_model/model_dataset4_100Hz.pt"

weight_path = "weight/traced_model_loss123_epoch200_gpu_dataset4_seq2000_Hz_100.pt"

model.load_state_dict(torch.load(weight_path))
traced_script_module = torch.jit.trace(model, (example, hn))
traced_script_module.save(PATH)