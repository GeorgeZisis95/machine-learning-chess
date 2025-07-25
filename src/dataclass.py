import torch
import pandas as pd

from src.encode import get_canonical_board, uci_to_index

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen = self.data.iloc[idx]["fen"]
        move = self.data.iloc[idx]["move"]
        value = self.data.iloc[idx]["value"]

        board_tensor = get_canonical_board(fen)
        move_index = uci_to_index[move]

        board_tensor = torch.tensor(board_tensor)
        move_index = torch.tensor(move_index)
        value_tensor = torch.tensor(value, dtype=torch.float32)

        return board_tensor, move_index, value_tensor