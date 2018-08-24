import torch
import torch.nn as nn
import math

# MLP with two hidden layers, each with 200 neurons, and ReLU activations.

class MLP(nn.Module):
    def __init__(self, input_size=28*28, num_classes=10):
        self.architecture = 'MLP--2x200_neurons--input_size=' + str(input_size) + '--num_classes=' + str(num_classes)
        self.input_size = input_size
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(self.input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
