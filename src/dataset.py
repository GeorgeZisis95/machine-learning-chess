import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from src.encode import fen_to_tensor, uci_to_index

class ChessDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen = self.data.iloc[idx]["fen"]
        move = self.data.iloc[idx]["move"]

        board_tensor = fen_to_tensor(fen)
        move_index = uci_to_index[move]

        board_tensor = torch.tensor(board_tensor)
        move_index = torch.tensor(move_index)

        return board_tensor, move_index