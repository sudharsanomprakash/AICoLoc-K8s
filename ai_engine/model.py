import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim=6):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.fc(x)

