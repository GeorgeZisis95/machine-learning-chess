from torch.utils.data import DataLoader
from src.dataset import ChessDataset
from src.encode import uci_to_index
from src.model import ChessNet

dataset = ChessDataset("data/dataset.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    boards, labels = batch  # boards: (B, 12, 8, 8), labels: (B,)
    print("Board shape:", boards.shape)
    print("Label shape:", labels.shape)
    break


model = ChessNet(num_moves=len(uci_to_index))
sample_batch = next(iter(loader))
sample_boards, _ = sample_batch

output = model(sample_boards)

print("Output shape:", output.shape)  # Should be (B, 4672)
