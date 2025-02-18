import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_SIZE = 1968


class ResNet(nn.Module):
    def __init__(self, device, input_channels=18, filters=256, res_blocks=19):
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
                                        nn.Linear(2 * 8 * 8, ACTION_SIZE))

        self.value_head = nn.Sequential(nn.Conv2d(filters, 1, kernel_size=1, padding=0, bias=True),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(),
                                        nn.Flatten(),
                                        nn.Linear(1 * 8 * 8, 256),
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

def train():
    BATCH_SIZE = 256
    NUM_EPOCHS = 10
    FILTERS = 256
    RES_BLOCKS = 19
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001

    states = np.load(f'encoded_data_collection/features.npy', allow_pickle=True)
    actions = np.load(f'encoded_data_collection/labels.npy', allow_pickle=True)
    rewards = np.load(f'encoded_data_collection/rewards.npy', allow_pickle=True)
    
    dataset = list(zip(states, actions, rewards))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(device=device, input_channels=18, filters=FILTERS, res_blocks=RES_BLOCKS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("--> Start training")
    model.train()
    for epoch in range(NUM_EPOCHS):
        print("|---------------------------------------------------------------------------|")
        print(f"    Epoch number {epoch+1}")
        chunks = (len(dataset) - 1) // BATCH_SIZE + 1
        for i in range(chunks-1):
            sample = dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            states, policy_targets, value_targets = zip(*sample)

            states = np.array(states, dtype=np.float32) 
            policy_targets = np.array(policy_targets, dtype=np.float32) 
            value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

            states = torch.tensor(states, dtype=torch.float32, device=model.device)

            out_policy, out_value = model(states)

            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=model.device)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"    Policy Loss: {policy_loss:.3f} Value Loss: {value_loss:.3f}")
    print("|---------------------------------------------------------------------------|")
    print("--> Saving Model")
    if not os.path.isdir("checkpoint_model"):
        os.mkdir("checkpoint_model")
    torch.save(model.state_dict(), f"checkpoint_model/sl_model.pt")
    if not os.path.isdir("checkpoint_optim"):
        os.mkdir("checkpoint_optim")
    torch.save(optimizer.state_dict(),  f"checkpoint_optim/sl_optim.pt")