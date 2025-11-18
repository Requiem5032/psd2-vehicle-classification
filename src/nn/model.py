import torch
import torch.nn as nn

from src.utils import *


class NeuralNetwork(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super(NeuralNetwork, self).__init__()
        self.hidden_activation = nn.ReLU()
        self.final_activation = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(input_channels*WINDOW_SIZE, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.hidden_activation(x)
        x = self.fc2(x)
        x = self.final_activation(x)
        return x


class LogisticRegression(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super(LogisticRegression, self).__init__()
        self.final_activation = nn.Softmax(dim=-1)
        self.linear = nn.Linear(input_channels*WINDOW_SIZE, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.final_activation(x)
        return x
