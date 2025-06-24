from torch.utils.data import DataLoader
from src.dataset import ChessDataset

dataset = ChessDataset("data/dataset.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    boards, labels = batch  # boards: (B, 12, 8, 8), labels: (B,)
    print("Board shape:", boards.shape)
    print("Label shape:", labels.shape)
    break
