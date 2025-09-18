import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEdgeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # input 3 channels -> 8 -> 16 -> 1
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)  # logits
        return x
