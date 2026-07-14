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
    
class ResNet(nn.Module):
	def __init__(self, input_channels=18, filters=256, res_blocks=19):
		super().__init__()

		self.start_block = nn.Sequential(nn.Conv2d(input_channels, filters, kernel_size=3, padding=1, bias=False),
										nn.BatchNorm2d(filters),
										nn.ReLU())

		self.res_tower = nn.ModuleList([ResBlock(filters) for _ in range(res_blocks)])

		self.policy_head = nn.Sequential(nn.Conv2d(filters, 2, kernel_size=1, padding=0, bias=True),
										nn.BatchNorm2d(2),
										nn.ReLU(),
										nn.Flatten(),
										nn.Linear(2 * 8 * 8, 4544))

		self.value_head = nn.Sequential(nn.Conv2d(filters, 1, kernel_size=1, padding=0, bias=True),
									   nn.BatchNorm2d(1),
									   nn.ReLU(),
									   nn.Flatten(),
									   nn.Linear(1 * 8 * 8, 1),
									   nn.Tanh())

	def forward(self, x):
		x = self.start_block(x)
		for res_block in self.res_tower:
			x = res_block(x)
		policy = self.policy_head(x)
		value = self.value_head(x)
		return policy, value


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