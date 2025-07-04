import chess
import numpy as np

def position_planes(board:str) -> np.ndarray:
    check_location = list(board.split()[0])

    for idx, char in enumerate(check_location):
        check_location[idx] = '1' * int(char) if char.isnumeric() else char
    check_location = "".join(check_location).replace("/","")

    pieces_order = 'KQRBNPkqrbnp'
    pieces_dict = {pieces_order[i]: i for i in range(12)}

    planes = np.zeros((12,8,8), dtype=np.float32)
    for row in range(8):
        for column in range(8):
            piece = check_location[row * 8 + column]
            if piece.isalpha():
                planes[pieces_dict[piece]][row][column] = 1

    assert planes.shape == (12,8,8), f"position_planes shape is: {planes.shape} instead of (12,8,8)"
    return planes

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