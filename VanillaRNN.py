import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

class VanillaRNN(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers, device):
    super(VanillaRNN, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    # self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 2), nn.Tanh())
    self.fc = nn.Linear(hidden_size, 2)

  def forward(self, x, hn):
    out, hn = self.rnn(x, hn) # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
    out = self.fc(out)
    # out = out.reshape(out.shape[0], -1) # many to many 전략
    # out = self.fc(out)
    return out, hn