from stockfish import Stockfish

stockfish = Stockfish(path="/usr/games/stockfish")

import numpy as np
import random
import chess
import torch

from data_training import ResNet
from data_encoding import get_canonical_board, create_uci_labels
from data_collection import check_finish

FILTERS = 256
RES_BLOCKS = 19
ACTION_SIZE = 1968
ELO_RATING = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(device=device, input_channels=18, filters=FILTERS, res_blocks=RES_BLOCKS)
model.load_state_dict(torch.load(f"checkpoint_model/sl_model.pt"))
model.eval()

stockfish.set_elo_rating(elo_rating=ELO_RATING)

weight_cases = {
    1: [100],
    2: [80, 20],
    3: [80, 15, 5]
}

def get_valid_moves(board:chess.Board, state:str) -> np.array:
    board.set_fen(state)
    curr_actions = list(map(str, list(board.legal_moves)))
    return np.array([1 if obj in curr_actions else 0 for obj in create_uci_labels()])

@torch.no_grad
def get_prediction(board:chess.Board, state:str):
    policy, value = model(torch.tensor(get_canonical_board(state, False), device=device).unsqueeze(0))
    policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
    policy = policy * get_valid_moves(board, state)
    policy = policy / np.sum(policy)
    value = value.item()
    return policy, value

def play_game():
    action_list = []
    player = 1
    board = chess.Board()
    stockfish.set_position([])
    while True:
        if player == 1:
            best_actions = stockfish.get_top_moves(3)
            weights = weight_cases.get(len(best_actions), [])
            action_taken = random.choices(best_actions, weights=weights, k=1)[0]["Move"]
        else:
            fen = board.fen()
            policy, _ = get_prediction(board, fen)
            action_index = np.argmax(policy)
            action_taken = create_uci_labels()[action_index]

        board.push_san(action_taken)

        action_list.append(action_taken)
        stockfish.set_position(action_list)

        if check_finish(board):
                result = board.result()
                if result == "0-1":
                    reward = -1
                elif result == "1-0":
                    reward = 1
                else:
                    reward = 0
                print(reward)
                break

        player = player * -1
        print("------------------------------------")
        print(f"{board}")
        print("------------------------------------")

play_game()