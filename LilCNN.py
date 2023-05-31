# %% [markdown]
# **Draft Model for Mice Steep Stage Analysis**

# %%
import random
import numpy as np
import pandas as pd
import os
import re
import copy
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torchvision import transforms, datasets, models

from collections import OrderedDict

import torch.utils.data as utils

# from sklearn.preprocessing import LabelBinarizer

import tkinter as tk #tk for file dialog (requires Jinja2!!!)
from tkinter import filedialog #tkinter for file dialog

import re #regex for parsing
from os.path import exists


# %%
from inputMassager import *
inputHandler = inputMassager()

filepath = inputHandler.askForInput("Blah!")

#Period Size Variable : effects CNN architecture
periodSize = 200

#makePeriodFromTxt(self, filepath, periodSize, maxPeriods=None):
periods = makePeriodFromTxt(filepath, periodSize, 5000)


# target = regan and audreys code ---> This is what will go into the data loader
filepath2 = inputHandler.askForInput("Blah!")
timestamps = find_time_labels(filepath2)
labels, _, _ = label_dataframe_new(periods, timestamps)


# %% [markdown]
# Make a tensor out of eeg data (c1)

# %%
def getChannelTensors():
    #The tensor has to be 3 dimesional with the second dimesion equal to one in order to pass it through a 1d convolutional layer
    c1_tensor = torch.zeros((len(periods.c1.values), 1, periodSize)) 
    c2_tensor = torch.zeros((len(periods.c2.values), 1, periodSize)) 
    for i in range(len(periods.c1.values)):
        c1_tensor[i, 0] = torch.tensor(periods.c1.values[i])
        c2_tensor[i, 0] = torch.tensor(periods.c1.values[i])
    
    return c1_tensor, c2_tensor

eeg_samples, emg_samples = getChannelTensors()

# %% [markdown]
# Make the train and validation data loaders  

# %%
#print(eeg_samples)
print(torch.tensor(labels))

ds = torch.utils.data.TensorDataset(eeg_samples, emg_samples, torch.tensor(labels))
train_dataset, val_dataset = torch.utils.data.random_split(ds, [int(len(ds) *.80), int(len(ds) *.2)])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=True)

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# **Creating class for Model**

# %%
import torch.nn.functional as F
from torch import nn, optim


class Model_3(nn.Module):
    def __init__(self, period_size, input_channels=1):
        super(Model_3, self).__init__()
        self.input_channels = input_channels

        self.eeg_conv1 = nn.Conv1d(in_channels=1, out_channels=period_size, kernel_size=period_size//2)
        self.eeg_conv2 = nn.Conv1d(in_channels=period_size, out_channels=period_size//4, kernel_size=8)
        self.eeg_conv3 = nn.Conv1d(in_channels=period_size // 4, out_channels=256, kernel_size=2)

        self.emg_conv1 = nn.Conv1d(in_channels=1, out_channels=period_size, kernel_size=period_size//2)
        self.emg_conv2 = nn.Conv1d(in_channels=period_size, out_channels=period_size//4, kernel_size=8)
        self.emg_conv3 = nn.Conv1d(in_channels=period_size // 4, out_channels=256, kernel_size=2)



        # out_channels and kernal size are random
        self.fc1 = nn.Linear(2560*2, 20) # 10 is random
        self.fc2 = nn.Linear(10, 5)


    def forward(self, c1, c2):
        #print(x.size())
        c1 = self.eeg_conv1(c1)
        c1 = F.relu(c1)
        # print(x.size())
        c1 = F.max_pool1d(c1, kernel_size=2)
        c1 = self.eeg_conv2(c1)
        c1 = F.max_pool1d(c1, kernel_size=2)
        # print(x.size())
        c1 = self.eeg_conv3(c1)
        c1 = F.max_pool1d(c1, kernel_size=2)
        #print(x.size())

        c2 = self.eeg_conv1(c2)
        c2 = F.relu(c2)
        # print(x.size())
        c2 = F.max_pool1d(c2, kernel_size=2)
        c2 = self.eeg_conv2(c2)
        c2 = F.max_pool1d(c2, kernel_size=2)
        # print(x.size())
        c2 = self.eeg_conv3(c2)
        c2 = F.max_pool1d(c2, kernel_size=2)


        c1c2 = torch.cat((c1, c2), dim=1)

        c1c2 = c1c2.flatten(1)
        #print(x.size())
        c1c2 = self.fc1(c1c2)
        c1c2 = F.max_pool1d(c1c2, kernel_size=2)
        # print(x.size())
        c1c2 = self.fc2(c1c2)
        c1c2 = F.log_softmax(c1c2, dim=1)
        return c1c2

# %%
def train_model(epochs, model):
    model.train() # set model to training mode
    loss_fun = nn.CrossEntropyLoss() #define a loss function object

    for epoch in range(epochs):

      for batch_idx, (channel1, channel2, target) in enumerate(train_loader):
          #print(c)
          # print(target)
          channel1, channel2, target = channel1.to(device), channel2.to(device), target.to(device)
          
          optimizer.zero_grad()
          output = model(channel1, channel2) # guess we have to pass all channels instead of data?
          loss = loss_fun(output,target)
          loss.backward()
          optimizer.step()
          if batch_idx % 100 == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(channel1), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))

# %%
model_step_3 = Model_3(periodSize)
model_step_3.to(device)
  
optimizer = torch.optim.SGD(model_step_3.parameters(), lr=.01, momentum=0.9)
train_model(3, model_step_3)

# %%
def evaluate_model(model, dataloader, is_test=False):
  #Evaluation

  # Set model to evaluation mode
  model.eval()

  with torch.no_grad():
    correct = 0
    loss = 0

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    for channel1, channel2, target in dataloader:
        channel1, channel2, target = channel1.to(device), channel2.to(device), target.to(device)
        outputs = model(channel1, channel2)
        
        loss += torch.sum(criterion(outputs, target)).item()
        
        pred = outputs.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    loss /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        "Test" if is_test else "Validation",
        loss, correct, len(dataloader.dataset),
        accuracy))
  # Set model back to training mode
  model.train()

# %%
evaluate_model(model_step_3, val_loader, is_test=True)


