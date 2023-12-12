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
import argparse
from training_database import training_database
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog="train_sparse_feedback",
        description="train",
        epilog="2023_UGRP"
    )
    parser.add_argument("--epoch", type=int, help="epoch num",required=True)
    args = parser.parse_args()


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
    criterion = nn.MSELoss()

    # PATH = "model/train_direct_dict_sl32.pt"
    # model.load_state_dict(torch.load(PATH))

    ## training
    lr = 1e-3
    num_epochs = args.epoch
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_graph = [] # 그래프 그릴 목적인 loss.
    n = len(database.train_loader[0])
    
    training_db = training_database()

    for epoch in range(num_epochs):
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        train_loader = database.train_loader[random.randrange(0,train_datasize)]
        training_db.timestep = 0
        for batch_idx, data in enumerate(train_loader):
            
            seq_batch, target_batch = data #seq, target은 20개, seq는 3차원 배열 20*8*3 , target은 2차원 배열 20*2
            target_batch[:,0] += np.pi/2

            for seq_idx, seq in enumerate(seq_batch):
                
                if training_db.timestep % training_db.true_period == 0:
                    training_db.out = model(seq.unsqueeze(0))
                    training_db.seq_buffer = seq.unsqueeze(0)                 
                    
                else:
                    training_db.seq_buffer = training_db.seq_buffer[:,1:,:] # pop
                    insert_piece = torch.cat( (seq[-1,0].unsqueeze(0), training_db.out.squeeze(0).clone().detach()) ).unsqueeze(0)
                    insert_piece[:,1] -= np.pi/2
                    training_db.seq_buffer = torch.cat( (training_db.seq_buffer, insert_piece.unsqueeze(0)), dim=1 ) # push
                    training_db.out = model(training_db.seq_buffer)      

                if seq_idx == 0:
                    training_db.out_batch = training_db.out
                else:
                    training_db.out_batch = torch.cat( (training_db.out_batch, training_db.out) )

                training_db.timestep += 1
            # loss 1 calculate
            loss1 = criterion(training_db.out_batch, target_batch)
            
            # loss 2 calculate
            dt = 600/30000.
            expected_theta_dot = torch.zeros(training_db.out_batch[:,0].shape[0]).to(device)
            
            bbefore_theta = 0
            before_theta = 0
            for idx, theta in enumerate(training_db.out_batch[:,0]):
                if idx > 1:
                    theta_dot = (theta - bbefore_theta)/dt 
                    expected_theta_dot[idx-1] = theta_dot
                bbefore_theta = before_theta
                before_theta = theta       
            loss2 = criterion(expected_theta_dot[1:-1], target_batch[1:-1,1])

            # loss 3 calculate
            g = 9.8 
            L = 1.5 
            m = 1.0
            M = 2.0 
            k = 0.8 # coefficients c/m
            x_ddot = seq_batch[1:,-1,0]
            true_theta_ddot = -k*target_batch[:-1,1]*torch.cos(target_batch[:-1,0])-(g/L)*torch.sin(target_batch[:-1,0])+(x_ddot/L)*torch.cos(target_batch[:-1,0])
            expected_theta_ddot = torch.zeros(training_db.out_batch[:,0].shape[0]).to(device)
            bbefore_theta_dot = 0
            before_theta_dot = 0
            for idx, theta_dot in enumerate(training_db.out_batch[:,1]):
                if idx > 1:
                    theta_ddot = (theta_dot - bbefore_theta_dot)/dt 
                    expected_theta_ddot[idx-1] = theta_ddot
                bbefore_theta_dot = before_theta_dot
                before_theta_dot = theta_dot       
            loss3 = criterion(expected_theta_ddot[1:-1], true_theta_ddot[1:])

            loss1 = 100*loss1 # 100
            loss2 = 1*loss2 # 0.01
            loss3 = 0.1*loss3 # 0.001
            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss3 += loss3.item()
        running_loss = running_loss1 + running_loss2 + running_loss3
        writer.add_scalar("tot_Loss/epoch", running_loss/n, epoch)
        writer.add_scalar("Loss1/epoch", running_loss1/n, epoch)
        writer.add_scalar("Loss2/epoch", running_loss2/n, epoch)
        writer.add_scalar("Loss3/epoch", running_loss3/n, epoch)
        loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
        
        print('[epoch: %d] loss1: %.4f loss2: %.4f loss3: %.4f'%(epoch, running_loss1 / n, running_loss2 / n, running_loss3 / n))
        if epoch % 50 == 0:
            PATH = "model/checkpoint/train_sparse_feedback_dict_batch"+str(database.batch_size)+"_epoch_" + str(num_epochs)+"\\"+str(epoch)+"_loss123_2.pt"
            torch.save(model.state_dict(), PATH)
    writer.close()
    plt.figure()
    plt.plot(loss_graph)
    plt.show()

    ## model wieght save
    PATH = "model/train_sparse_feedback_dict_batch"+str(database.batch_size)+"_epoch_" + str(num_epochs)+ "_loss123_2.pt"
    torch.save(model.state_dict(), PATH)




