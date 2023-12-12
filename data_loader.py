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
    
    def __init__(self, train_datasize, test_datasize, device):
        self.device = device
        self.sequence_length = 32
        self.batch_size = 50
        print("batch_size :",self.batch_size)
        # make train dataset
        self.train_datasize = train_datasize 
        for i in range(0,self.train_datasize):
            filename = "train/train"+str(i)+".csv"
            traindata = pd.read_csv(filename)
            train_full_data = traindata.values
            if i == 0:
                self.train_input = np.expand_dims(train_full_data[:-1],axis = 0) # delete last data
                self.train_output = np.expand_dims(train_full_data[:,1:3],axis = 0)
                train_input_seq, train_output_seq = self.seq_data(self.train_input[i], self.train_output[i], self.sequence_length)
                self.train_input_seq = train_input_seq.unsqueeze(0)
                self.train_output_seq = train_output_seq.unsqueeze(0)
                train = torch.utils.data.TensorDataset(self.train_input_seq[i], self.train_output_seq[i])
                self.train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=False)
                self.train_loader = [self.train_loader]
            else:
                self.train_input = np.append(self.train_input, np.expand_dims(train_full_data[:-1],axis = 0),axis = 0) # delete last data
                self.train_output = np.append(self.train_output,  np.expand_dims(train_full_data[:,1:3],axis = 0),axis = 0)
                train_input_seq, train_output_seq = self.seq_data(self.train_input[i], self.train_output[i], self.sequence_length)
                self.train_input_seq = torch.cat((self.train_input_seq, train_input_seq.unsqueeze(0)))
                self.train_output_seq = torch.cat((self.train_output_seq, train_output_seq.unsqueeze(0)))
                train = torch.utils.data.TensorDataset(self.train_input_seq[i], self.train_output_seq[i])
                self.train_loader.append(torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=False))
        #make test dataset
        self.test_datasize = test_datasize
        for i in range(0,self.test_datasize):
            filename = "test/test"+str(i)+".csv"
            testdata = pd.read_csv(filename)
            test_full_data = testdata.values
            if i == 0:
                self.test_input = np.expand_dims(test_full_data[:-1],axis = 0) # delete last data
                self.test_output = np.expand_dims(test_full_data[:,1:3],axis = 0)
                test_input_seq, test_output_seq = self.seq_data(self.test_input[i], self.test_output[i], self.sequence_length)
                self.test_input_seq = test_input_seq.unsqueeze(0)
                self.test_output_seq = test_output_seq.unsqueeze(0)
                test = torch.utils.data.TensorDataset(self.test_input_seq[i], self.test_output_seq[i])
                self.test_loader = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False)
                self.test_loader = [self.test_loader]

            else:
                self.test_input = np.append(self.test_input, np.expand_dims(test_full_data[:-1],axis = 0),axis = 0) # delete last data
                self.test_output = np.append(self.test_output,  np.expand_dims(test_full_data[:,1:3],axis = 0),axis = 0)
                test_input_seq, test_output_seq = self.seq_data(self.test_input[i], self.test_output[i], self.sequence_length)
                self.test_input_seq = torch.cat((self.test_input_seq, test_input_seq.unsqueeze(0)))
                self.test_output_seq = torch.cat((self.test_output_seq, test_output_seq.unsqueeze(0)))
                test = torch.utils.data.TensorDataset(self.test_input_seq[i], self.test_output_seq[i])
                self.test_loader.append(torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False))


    def seq_data(self, input, output, sequence_length):

        x_seq = []
        y_seq = []
        
        for i in range(len(input)-sequence_length):
            x_seq.append(input[i:i+sequence_length])
            y_seq.append(output[i+sequence_length])        
        return torch.FloatTensor(x_seq).to(self.device), torch.FloatTensor(y_seq).to(self.device)


