import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
	def __init__(self, game, device, input_channels=18, filters=256, res_blocks=19):
		super().__init__()
		self.device = device

		self.start_block = nn.Sequential(nn.Conv2d(input_channels, filters, kernel_size=3, padding=1, bias=False),
										nn.BatchNorm2d(filters),
										nn.ReLU())

		self.res_tower = nn.ModuleList([ResBlock(filters) for _ in range(res_blocks)])

		self.policy_head = nn.Sequential(nn.Conv2d(filters, 2, kernel_size=1, padding=0, bias=True),
										nn.BatchNorm2d(2),
										nn.ReLU(),
										nn.Flatten(),
										nn.Linear(2 * game.rows * game.columns, game.action_size))

		self.value_head = nn.Sequential(nn.Conv2d(filters, 1, kernel_size=1, padding=0, bias=True),
									   nn.BatchNorm2d(1),
									   nn.ReLU(),
									   nn.Flatten(),
									   nn.Linear(1 * game.rows * game.columns, 256),
									   nn.ReLU(),
									   nn.Linear(256, 1),
									   nn.Tanh())
		self.to(device)

	def forward(self, state):
		state = self.start_block(state)
		for res_block in self.res_tower:
			state = res_block(state)
		policy = self.policy_head(state)
		value = self.value_head(state)
		return policy, value


class ResBlock(nn.Module):
	def __init__(self, filters):
		super().__init__()
		self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
		self.batch1 = nn.BatchNorm2d(filters)
		self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
		self.batch2 = nn.BatchNorm2d(filters)

	def forward(self, state):
		residual = state
		state = F.relu(self.batch1(self.conv1(state)))
		state = self.batch2(self.conv2(state))
		state = state + residual
		state = F.relu(state)
		return state


def train(self, dataset):
    chunks = (len(dataset) - 1) // self.args.batch_size + 1
    for i in range(chunks-1):
        sample = dataset[i*self.args.batch_size:(i+1)*self.args.batch_size]
        states, policy_targets, value_targets = zip(*sample)

        states = np.array(states, dtype=np.float32) 
        policy_targets = np.array(policy_targets, dtype=np.float32) 
        value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

        states = torch.tensor(states, dtype=torch.float32, device=self.model.device)

        out_policy, out_value = self.model(states)

        policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
        value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

        policy_loss = F.cross_entropy(out_policy, policy_targets)
        value_loss = F.mse_loss(out_value, value_targets)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_dataset(self, dataset):
    print("==> Start training")
    self.model.train()
    for epoch in range(self.args.num_epochs):
        print(f"Epoch number {epoch} ---------------------------------------------------------------------------")
        self.train(dataset)
        if self.scheduler is not None:
            self.scheduler.step()
    print("==> Saving Model")
    if not os.path.isdir("model"):
        os.mkdir("model")
    torch.save(self.model.state_dict(), f"model/{repr(self.game)}{self.args.version}.{self.args.iteration}.pt")
    if not os.path.isdir("optim"):
        os.mkdir("optim")
    torch.save(self.optimizer.state_dict(),  f"optim/{repr(self.game)}{self.args.version}.{self.args.iteration}.pt")