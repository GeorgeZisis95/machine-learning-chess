import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self, num_moves=4544):
        super(ChessNet, self).__init__()

        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_moves)  # Output = number of UCI move classes

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B, 32, 8, 8)
        x = F.relu(self.conv2(x))   # (B, 64, 8, 8)
        x = F.relu(self.conv3(x))   # (B, 64, 8, 8)

        x = x.view(x.size(0), -1)   # Flatten
        x = F.relu(self.fc1(x))     # (B, 512)
        x = self.fc2(x)             # (B, num_moves)

        return x
