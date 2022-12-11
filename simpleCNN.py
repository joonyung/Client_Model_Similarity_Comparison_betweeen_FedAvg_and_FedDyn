import torch
import torch.nn as nn
import torch.nn.functional as F

class simpleCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size = 5, stride = 1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5, stride = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc3 = nn.Linear(400, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 84)
        self.fc6 = nn.Linear(84, 256)
        self.fc7 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = self.fc7(x)

        return x

def make_simpleCNN():
    return simpleCNN()
    

