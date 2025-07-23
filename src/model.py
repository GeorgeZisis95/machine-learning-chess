import torch.nn as nn
import torch.nn.functional as F

class ConvModel(nn.Module):
    def __init__(self, input_channels=12, output_size=4544):
        super(ConvModel, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(filters)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.batch1(self.conv1(x)))
        x = self.batch2(self.conv2(x))
        x = x + residual
        x = F.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, input_channels=18, filters=256, res_blocks=19, output_size=4544):
        super().__init__()
        self.start_conv = nn.Conv2d(input_channels, filters, kernel_size=3, padding=1, bias=False)
        self.start_batch = nn.BatchNorm2d(filters)
        self.res_tower = nn.ModuleList([ResBlock(filters) for _ in range(res_blocks)])
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(filters, 2 * filters)
        self.fc2 = nn.Linear(2 * filters, output_size)

    def forward(self, x):
        x = F.relu(self.start_batch(self.start_conv(x)))
        for res_block in self.res_tower:
            x = res_block(x)
        x = self.global_avg(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x