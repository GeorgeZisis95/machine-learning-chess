import torch
import pandas as pd

from src.encode import position_planes, uci_to_index

def preprocess_csv_to_tensors(csv_path, pt_path):
    df = pd.read_csv(csv_path)
    boards = []
    moves = []

    for i in range(len(df)):
        fen = df.iloc[i]['fen']
        move = df.iloc[i]['move']
        
        board = position_planes(fen)
        move_index = uci_to_index[move]

        boards.append(board)
        moves.append(move_index)

    boards = torch.tensor(boards, dtype=torch.float32)
    moves = torch.tensor(moves, dtype=torch.long)

    torch.save(
        {
            "boards": boards,
            "moves": moves,
        },
    pt_path
)

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)

        self.boards = data["boards"]
        self.moves = data["moves"]
    
    def __len__(self):
        return self.boards.shape[0]

    def __getitem__(self, idx):
        return (
            self.boards[idx],
            self.moves[idx],
        )