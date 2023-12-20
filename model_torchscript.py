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

device = torch.device('cpu')
print(f'{device} is available')


## parameters, dataset

database = data_loader(num_epochs=200, device=device)

## models
input_size = 1 
num_layers = 2
hidden_size = 8
model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   num_layers=num_layers,
                   device=device).to(device)


hn = torch.rand(num_layers, 1, hidden_size).to(device)
example = torch.rand(1).unsqueeze(0).unsqueeze(0).to(device)

traced_script_module = torch.jit.trace(model, (example, hn))
traced_script_module.save("model/ugrp_rnn_model.pt")
# import torch
# import torchvision

# # 모델 인스턴스 생성
# model = torchvision.models.resnet18()

# # 일반적으로 모델의 forward() 메소드에 넘겨주는 입력값
# example = torch.rand(1, 3, 224, 224)

# # torch.jit.trace를 사용하여 트레이싱을 이용해 torch.jit.ScriptModule 생성
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("model/traced_resnet_model.pt")
