import torch
import tqdm
import chess
import chess.pgn 
import chess.engine

from src.model import ConvNet
from src.predict import get_uci_move
from src.encode import all_uci_moves, uci_to_index, position_planes
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet(input_channels=12, output_size=4544)

checkpoint = torch.load(f"models/model.99.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

stockfish_path = r"C:\Users\myPC\Downloads\stockfish\stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
engine.configure({"Skill Level": 1})
stockfish_depth = 5

def play_game(model_white=True):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    game.headers["White"] = "MyModel" if model_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if model_white else "MyModel"

    while not board.is_game_over():
        if (board.turn == chess.WHITE and model_white) or (board.turn == chess.BLACK and not model_white):
            move = get_uci_move(board, model, device)   
        else:
            result = engine.play(board, chess.engine.Limit(depth=stockfish_depth))
            move = result.move
        node = node.add_variation(move)
        board.push(move)
    game.headers["Result"] = board.result()
    print(game)
    return board.result()


def play_n_games(num_games=50):
    results = {"win": 0, "loss": 0, "draw": 0}

    for i in tqdm.tqdm(range(num_games)):
        model_white = (i % 2 == 0)

        result = play_game(model_white=model_white)

        if result == "1/2-1/2":
            results["draw"] += 1
        elif (result == "1-0" and model_white) or (
            result == "0-1" and not model_white
        ):
            results["win"] += 1
        else:
            results["loss"] += 1

    print(results)
    return results

# print(play_game(model_white=True))
play_n_games(num_games=50)

engine.quit()