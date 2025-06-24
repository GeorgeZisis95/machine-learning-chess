import torch
import torch.nn as nn
import torch.optim as optim
from src.model import ChessNet
from src.dataset import ChessDataset
from torch.utils.data import DataLoader

# --- Load data ---
dataset = ChessDataset("data/dataset.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- Initialize model ---
model = ChessNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for boards, labels in loader:
        boards = boards.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(boards)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")
