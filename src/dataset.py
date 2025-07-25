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

def result_to_value(result_str:str, current_player:chess) -> int:
    if result_str == "1-0":
        return 1 if current_player == chess.WHITE else -1
    elif result_str == "0-1":
        return -1 if current_player == chess.WHITE else 1
    else:
        return 0

def create_value_csv_dataset(games:list, name:str):
    with open(f"data/csv/{name}.csv", 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["fen", "move", "value"])
        for game in games:
            result = game.headers.get("Result")
            board = game.board()
            for move in game.mainline_moves():
                fen = board.fen()
                uci = move.uci()
                value = result_to_value(result, board.turn)
                writer.writerow([fen, uci, value])
                board.push(move)