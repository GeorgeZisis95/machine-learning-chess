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

def helper_planes(board:str) -> np.ndarray:
    split_string = board.split()
    en_passant = np.zeros((8, 8), dtype=np.float32)
    alg_to_coord = lambda col, row: (8-int(row), ord(col)-ord('a')) 
    if split_string[3] != '-':
        col, row = split_string[3][0], split_string[3][1]
        the_rank, the_file = alg_to_coord(col, row)
        en_passant[the_rank][the_file] = 1

    planes = [
        np.full((8, 8), int('K' in split_string[2]), dtype=np.float32),
        np.full((8, 8), int('Q' in split_string[2]), dtype=np.float32),
        np.full((8, 8), int('k' in split_string[2]), dtype=np.float32),
        np.full((8, 8), int('q' in split_string[2]), dtype=np.float32),
        np.full((8, 8), int(split_string[1] == 'b'), dtype=np.float32),
        en_passant,
    ]
    planes = np.asarray(planes, dtype=np.float32)
    assert planes.shape == (6,8,8), f"helper_planes shape is:{planes.shape} instead of (6,8,8)"
    return planes

def change_perspective(board:str) -> str:
    split_string = board.split()
    if split_string[1] == 'b':
        piece_position = split_string[0].split("/")
        split_string[0] = "/".join([char.swapcase() for char in reversed(piece_position)])
        split_string[1] = 'w'
        split_string[2] = "".join(sorted([char.swapcase() for char in split_string[2]]))
    return " ".join(split_string)

def get_canonical_board(board:str, perpective:bool=False) -> np.ndarray:
    if perpective:
        updated_board = change_perspective(board)
        return np.vstack((position_planes(updated_board), helper_planes(updated_board)))
    return np.vstack((position_planes(board), helper_planes(board)))

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