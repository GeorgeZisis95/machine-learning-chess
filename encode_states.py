import numpy as np

def change_perspective(board:str) -> str:
    split_string = board.split()
    if split_string[1] == 'b':
        piece_position = split_string[0].split("/")
        split_string[0] = "/".join([char.swapcase() for char in reversed(piece_position)])
        split_string[1] = 'w'
        split_string[2] = "".join(sorted([char.swapcase() for char in split_string[2]]))
    return " ".join(split_string)

def helper_planes(board:str) -> np.ndarray:
    split_string = board.split()

    fifty_move_count = np.full((8, 8), int(split_string[4]), dtype=np.float32)
    total_move_count = np.full((8, 8), int(split_string[5]), dtype=np.float32)

    planes = [
        np.full((8, 8), int('K' in split_string[2]), dtype=np.float32),
        np.full((8, 8), int('Q' in split_string[2]), dtype=np.float32),
        np.full((8, 8), int('k' in split_string[2]), dtype=np.float32),
        np.full((8, 8), int('q' in split_string[2]), dtype=np.float32),
        fifty_move_count,
        total_move_count,
    ]
    planes = np.asarray(planes, dtype=np.float32)
    assert planes.shape == (6,8,8), f"helper_planes shape is:{planes.shape} instead of (6,8,8)"
    return planes

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

def get_canonical_board(board:str) -> np.ndarray:
    updated_board = change_perspective(board)
    return np.vstack((position_planes(updated_board), helper_planes(updated_board)))