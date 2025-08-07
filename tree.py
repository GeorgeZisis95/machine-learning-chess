import torch
import chess

from src.model import ResNet
from src.dataclass import ChessDataset
from src.search import AlphaTreeSearch

val_dataset = ChessDataset("data/csv/le2017-01.csv")
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(filters=128, res_blocks=6)

checkpoint = torch.load(f"models/model5/model.50.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
print(board.is_game_over())
legal_moves = [element.uci() for element in board.legal_moves]
print(legal_moves)
search = AlphaTreeSearch(model, device)
action = search.search(board, 100)

print(action)