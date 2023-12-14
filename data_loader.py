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

class data_loader():
    
    def __init__(self, device):
        self.device = device
        
        self.sequence_length = 100
        self.batch_size = 20
        self.num_epochs = 100
        
        print("total_epochs: ",self.num_epochs)
        print("batch_size :",self.batch_size)
        print("sequence_length :",self.sequence_length)

        # make train dataset
    
        filename = "train/train0.csv"
        traindata = pd.read_csv(filename)
        train_full_data = traindata.values
    
        self.train_input_seq = train_full_data[:-1,1] # delete last data
        self.train_output_seq = train_full_data[1:,1:3] # delete first data

        self.train_input_seq, self.train_output_seq = self.seq_data(self.train_input_seq, self.train_output_seq, self.sequence_length)
        self.train_input_seq = self.train_input_seq.unsqueeze(-1)
        train = torch.utils.data.TensorDataset(self.train_input_seq, self.train_output_seq)
        self.train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=False)
        
    
        #make test dataset

        filename = "test/test1.csv"
        testdata = pd.read_csv(filename)
        test_full_data = testdata.values

        self.test_input_seq = test_full_data[:-1,1] # delete last data
        self.test_output_seq = test_full_data[1:,1:3]
        
        self.test_input_seq = torch.FloatTensor(self.test_input_seq).to(self.device).unsqueeze(-1).unsqueeze(1)
        self.test_output_seq = torch.FloatTensor(self.test_output_seq).to(self.device).unsqueeze(1)
        test = torch.utils.data.TensorDataset(self.test_input_seq, self.test_output_seq)
        self.test_loader = torch.utils.data.DataLoader(test, batch_size=self.test_output_seq.shape[0], shuffle=False)

           

    def seq_data(self, input, output, sequence_length):

        x_seq = []
        y_seq = []

        for i in range(len(input)-sequence_length):
            x_seq.append(input[i:i+sequence_length])
            y_seq.append(output[i:i+sequence_length])        
        return torch.FloatTensor(x_seq).to(self.device), torch.FloatTensor(y_seq).to(self.device)

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available')
    database = data_loader(device)
    # print(database.test_output_seq.shape)
    print(len(database.train_loader))
    for data in database.test_loader:
        seq, target = data
        print("seq:", seq.shape)
        print("target:", target.shape)
