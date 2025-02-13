import numpy as np
import os

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
    en_passant = np.zeros((8, 8), dtype=np.float32)
    alg_to_coord = lambda col, row: (8-int(row), ord(col)-ord('a')) 
    if split_string[3] != '-':
        col, row = split_string[3][0], split_string[3][1]
        the_rank, the_file = alg_to_coord(col, row)
        en_passant[the_rank][the_file] = 1
    fifty_move_count = np.full((8, 8), int(split_string[4]), dtype=np.float32)

    planes = [
        np.full((8, 8), int('K' in split_string[2]), dtype=np.float32),
        np.full((8, 8), int('Q' in split_string[2]), dtype=np.float32),
        np.full((8, 8), int('k' in split_string[2]), dtype=np.float32),
        np.full((8, 8), int('q' in split_string[2]), dtype=np.float32),
        fifty_move_count,
        en_passant,
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

def create_uci_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                            [(l1, t) for t in range(8)] + \
                            [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                            [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                            [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
    for l1 in range(8):
        l = letters[l1]
        for p in promoted_to:
            labels_array.append(l + '2' + l + '1' + p)
            labels_array.append(l + '7' + l + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(l + '2' + l_l + '1' + p)
                labels_array.append(l + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(l + '2' + l_r + '1' + p)
                labels_array.append(l + '7' + l_r + '8' + p)
    return labels_array

def encode_data():
    files = os.listdir('expert_data_collection')
    for idx, f in enumerate(files):
        actions_and_states = np.load(f'expert_data_collection/{f}', allow_pickle=True)
        labels = np.array(actions_and_states[:,0])
        features = actions_and_states[:,1]

        encoded_features = []
        for i in range(len(labels)):
            encoded_features.append(get_canonical_board(features[i]))
        encoded_data = np.concatenate(labels, np.array(encoded_features))

        if not os.path.isdir('encoded_data_collection'):
            os.mkdir('encoded_data_collection')
        np.save(f"encoded_data_collection/labels_and_features{idx}", encoded_data)