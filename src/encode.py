import csv
import chess
import chess.pgn
import itertools
import numpy as np

def create_pgn():
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    game.headers["Event"] = "Test Game"
    game.headers["White"] = "MyEngine"
    game.headers["Black"] = "RandomBot"
    game.headers["Result"] = board.result()

    # Sample moves (replace with actual game loop)
    while not board.is_game_over():
        move = list(board.legal_moves)[0]  # play first legal move
        board.push(move)
        node = node.add_variation(move)

    # Save to PGN file
    with open("saved_game.pgn", "w") as f:
        print(game, file=f)

def extract_from_pgn():
    pgn = open("sample_game.pgn")
    game = chess.pgn.read_game(pgn)
    board = game.board()

    for move in game.mainline_moves():
        fen = board.fen()
        uci = move.uci()

        print(f"fen: {fen}")
        print(f"Move: {uci}")
        print("------")

        board.push(move)

def save_csv():
    with open("dataset.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["fen", "move"])

        game = chess.pgn.read_game(open("sample_game.pgn"))
        board = game.board()

        for move in game.mainline_moves():
            fen = board.fen()
            uci = move.uci()
            writer.writerow([fen, uci])
            board.push(move)

piece_to_index = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            idx = piece_to_index[piece.piece_type]
            if piece.color == chess.BLACK:
                idx += 6
            tensor[idx, row, col] = 1

    return tensor

def generate_all_possible_uci_moves():
    all_moves = set()

    for from_rank in range(8):
        for from_file in range(8):
            for to_rank in range(8):
                for to_file in range(8):
                    from_sq = chess.square(from_file, from_rank)
                    to_sq = chess.square(to_file, to_rank)

                    if from_sq == to_sq:
                        continue

                    move = chess.Move(from_sq, to_sq)
                    all_moves.add(move.uci())

                    # Add legal promotions (only from rank 6 to 7 for white, 1 to 0 for black)
                    if (from_rank == 6 and to_rank == 7) or (from_rank == 1 and to_rank == 0):
                        for promo in ['q', 'r', 'b', 'n']:
                            promo_move = chess.Move(from_sq, to_sq, promotion=chess.Piece.from_symbol(promo).piece_type)
                            all_moves.add(promo_move.uci())

    all_moves = sorted(all_moves)
    return all_moves


all_uci_moves = generate_all_possible_uci_moves()
uci_to_index = {uci: idx for idx, uci in enumerate(all_uci_moves)}
index_to_uci = {idx: uci for uci, idx in uci_to_index.items()}