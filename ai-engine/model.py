import torch
import torch.nn as nn

class SimplePolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimplePolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
