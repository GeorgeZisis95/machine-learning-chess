import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import ChessNet
from dataclass import ChessDataset
from torch.utils.data import DataLoader

def save_model(current_epoch:int, latest_loss:float, model_name:str):
    torch.save({
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': latest_loss,
    }, f"models/{model_name}.pth")

def load_model(model_name:str):
    checkpoint = torch.load(f"models/{model_name}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")

# --- Load data ---
dataset = ChessDataset("data/csv/datasetA.csv")
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# --- Initialize model ---
model = ChessNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
EPOCHS = 5

for epoch in tqdm.trange(EPOCHS):
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
    save_model(epoch, avg_loss, "test_model")