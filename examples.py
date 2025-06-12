import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.parameter import Parameter
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from torchvision import transforms
from torchvision import datasets
import os
import random
from SpykeTorch.utils import *
from SpykeTorch.functional import *
from SpykeTorch.snn import *
from SpykeTorch.visualization import *
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import json
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import f1_score, mean_squared_error

kernels = [	GaborKernel(window_size = 3, orientation = 45+22.5),
            GaborKernel(3, 90+22.5),
            GaborKernel(3, 135+22.5),
            GaborKernel(3, 180+22.5)]
filt = Filter(kernels, use_abs = True)

def time_dim(input):
    return input.unsqueeze(0)

transform = transforms.Compose(
    [transforms.Grayscale(),
    transforms.ToTensor(),
    time_dim,
    filt,
    sf.pointwise_inhibition,
    Intensity2Latency(number_of_spike_bins = 15, to_spike = True)])

dataset = ImageFolder("Spyketorch_Training", transform)

indices = list(range(len(dataset)))
random.shuffle(indices)
split_point = int(0.50*len(indices))
train_indices = indices[:split_point]
test_indices = indices[split_point:]

print("Size of the training set:", len(train_indices))
print("Size of the  testing set:", len(test_indices))

dataset = CacheDataset(dataset)
train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_indices))
test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_indices))

class SNN(nn.Module):
    def __init__(self, input_channels, features_per_class, number_of_classes):
        super(SNN, self).__init__()
        self.features_per_class = features_per_class
        self.number_of_classes = number_of_classes
        self.number_of_features = features_per_class * number_of_classes
        
        self.pool = Pooling(kernel_size=3, stride=1, padding=1)
        self.conv = Convolution(input_channels, self.number_of_features, 3, 0.8, 0.05)
        self.conv.reset_weight()
        self.stdp = STDP(conv_layer=self.conv, learning_rate=(0.05, -0.015))
        self.anti_stdp = STDP(conv_layer=self.conv, learning_rate=(-0.05, 0.0005))
        self.winner_history = []
        
        
        # internal state of the model
        self.ctx = {"input_spikes": None, "potentials": None, "output_spikes": None, "winners": None}
        
        # map each neuron to the class it represents
        self.decision_map = []
        for i in range(number_of_classes):
            self.decision_map.extend([i] * features_per_class)
        
    def forward(self, x):
        x = self.pool(x)
        p = self.conv(x)
        spikes, potentials = fire(potentials=p, threshold=20, return_thresholded_potentials=True)
        winners = get_k_winners(potentials=p, kwta=1, inhibition_radius=0, spikes=spikes)
        # update the internal state: store updates after applying convolution and pooling
        self.ctx["input_spikes"] = x
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = spikes
        self.ctx["winners"] = winners
        output = -1
        if len(winners) != 0:
            output = self.decision_map[winners[0][0]]
        return output
    
    def reward(self, reward):
        if reward > 0:
            self.stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        else:
            self.anti_stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

def plot_spike_raster(spike_history):
    plt.figure(figsize=(12, 6)) 
    for i, spikes in enumerate(spike_history):
        spiking_times = np.nonzero(spikes)[0]  # Get the indices for spike times
        plt.scatter(spiking_times, np.full_like(spiking_times, i), marker='|', color='black')
    plt.title('Spike Raster Plot')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.yticks(range(len(spike_history)))  # Y-axis for each neuron
    plt.grid()
    plt.show()

spike_history = []


model = SNN(4, 20, 2)
for data, targets in train_loader:
    for x, y in zip(data, targets):

        out = model.forward(x)
        reward = int(out == y.item())
        model.reward(reward)
        spike_history.append(model.ctx["output_spikes"].cpu().numpy())
        #print(spike_history)

        print("action", out)
        if reward == 0:
            print('Decision', False)
        else:
             print('Decision', True)

y_preds = []
y_true = []

for data, targets in test_loader:
    for x, y in zip(data, targets):
        out = model.forward(x)
        y_preds.append(out)
        y_true.append(y.item())
        

F1 = f1_score(y_true, y_preds)
MSE = mean_squared_error(y_true, y_preds)
print('MSE', MSE)
print('F1', F1)
plot_spike_raster(spike_history)