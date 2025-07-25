import torch
import torch.nn as nn
import torch.nn.functional as F

import chess
import chess.engine

import tqdm
import math
import random

from src.encode import get_canonical_board, index_to_uci

class ConvNet(nn.Module):
    def __init__(self, input_channels=18, output_size=4544):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet()

checkpoint = torch.load(f"models/model.2.40.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

stockfish_path = r"C:\Users\myPC\Downloads\stockfish\stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
engine.configure({"Skill Level": 0})
stockfish_depth = 1

def predict_move(board):
    legal_moves = [element.uci() for element in board.legal_moves]
    model_input = torch.from_numpy(get_canonical_board(board.fen())).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(model_input)
        probs = torch.softmax(logits, dim=1).squeeze()
        top_moves = torch.topk(probs, k=3)
        shuffled_moves = top_moves.indices[torch.randperm(top_moves.indices.nelement())]
        for index in shuffled_moves:
            uci_move = index_to_uci[index.item()]
            if uci_move in legal_moves:
                return chess.Move.from_uci(uci_move)
    return random.choice(list(board.legal_moves))

def play_game(model_white=True):
    board = chess.Board()
    while not board.is_game_over():
        if (board.turn == chess.WHITE and model_white) or (board.turn == chess.BLACK and not model_white):
            move = predict_move(board)
        else:
            result = engine.play(board, chess.engine.Limit(depth=stockfish_depth))
            move = result.move
        board.push(move)
    return board.result()

num_games = 50
results = {"win": 0, "loss": 0, "draw": 0}

for i in tqdm.tqdm(range(num_games)):
    model_white = i % 2 == 0
    result = play_game(model_white=model_white)
    if (result == "1-0" and model_white) or (result == "0-1" and not model_white):
        results["win"] += 1
    elif result == "1/2-1/2":
        results["draw"] += 1
    else:
        results["loss"] += 1

print(results)
engine.quit()

W = results["win"]
L = results["loss"]
D = results["draw"]

if L + 0.5 * D == 0:
    elo_diff = float('inf')
else:
    elo_diff = 400 * math.log10((W + 0.5 * D + 1e-10) / (L + 0.5 * D + 1e-10))

stockfish_elo = 1500
your_model_elo = stockfish_elo + elo_diff
print(f"Estimated Elo: {your_model_elo:.0f}")