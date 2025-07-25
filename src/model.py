import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ConvNet(nn.Module):
    def __init__(self, input_channels=18, output_size=4544):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.fc3 = nn.Linear(64 * 8 * 8, 256)
        self.fc4 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        p = F.relu(self.fc1(x))
        p = self.fc2(p)

        v = F.relu(self.fc3(x))
        v = torch.tanh(self.fc4(v))

        return p, v