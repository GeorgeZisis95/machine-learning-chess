import os
import csv
import tqdm
import chess
import chess.pgn

files = [file for file in os.listdir("data/pgn") if file.endswith(".pgn")]

def load_pgn(file_path:str) -> list:
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

def load_multiple_pgns(num_pgns:int) -> list:
    total_games = []
    for index, file in tqdm.tqdm(enumerate(files), total=num_pgns):
        if index > num_pgns:
            break
        total_games.extend(load_pgn(f"data/pgn/{file}"))
        
    return total_games

def get_state_action_pairs(games:list) -> list:
    state_action_pairs = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            fen = board.fen()
            uci = move.uci()
            state_action_pairs.append([fen, uci])
            board.push(move)
    return state_action_pairs

def create_csv_dataset(games:list, name:str):
    with open(f"data/csv/{name}.csv", 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["fen", "move"])
        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                fen = board.fen()
                uci = move.uci()
                writer.writerow([fen, uci])
                board.push(move)

# games = load_multiple_pgns(27)
# create_csv_dataset(games, "datasetA")

# DatasetA contains the games of the first 27 files 

create_csv_dataset(load_pgn(f"data/pgn/lichess_elite_2016-02.pgn"), "evalA")