import chess
import numpy as np
from typing import Any

def unpack(move: chess.Move) -> tuple[int, int, int, int]:
    """Converts chess.Move instances into move coordinates."""

    from_rank = chess.square_rank(move.from_square)
    from_file = chess.square_file(move.from_square)

    to_rank = chess.square_rank(move.to_square)
    to_file = chess.square_file(move.to_square)

    return from_rank, from_file, to_rank, to_file

def pack(from_rank: int, from_file: int, to_rank: int, to_file: int) -> chess.Move:
    """Converts move coordinates into a chess.Move instance."""

    from_square = chess.square(from_file, from_rank)
    to_square = chess.square(to_file, to_rank)

    return chess.Move(from_square, to_square)

def rotate(move: chess.Move) -> chess.Move:
    """Flips a move from Black's perspective to White's."""

    def flip_square(square):
        the_rank = chess.square_rank(square)
        the_file = chess.square_file(square)
        flipped_rank = 7 - the_rank  # Reverse the rank
        return chess.square(the_file, flipped_rank)

    flipped_from = flip_square(move.from_square)
    flipped_to = flip_square(move.to_square)

    # Handle promotions by keeping the same promotion piece
    return chess.Move(flipped_from, flipped_to, move.promotion)



class IndexedTuple:
    """A regular tuple with an efficient `index` operation."""

    def __init__(self, *items: Any) -> None:

        #: The items stored in the tuple
        self._items = items

        #: Maps tuple elements to their indices
        self._indices = { item: idx for idx, item in enumerate(items) }

    def __getitem__(self, idx: int) -> Any:
        return self._items[idx]

    def index(self, item: Any) -> int:
        return self._indices[item]

    def __contains__(self, item: Any) -> bool:
        return item in self._items


def encode_knight(move: chess.Move):
    #: Starting point of knight moves in last dimension of 8 x 8 x 73 action array.
    _TYPE_OFFSET: int = 56

    #: Set of possible directions for a knight move, encoded as 
    #: (delta rank, delta square).
    _DIRECTIONS = IndexedTuple(
        (+2, +1),
        (+1, +2),
        (-1, +2),
        (-2, +1),
        (-2, -1),
        (-1, -2),
        (+1, -2),
        (+2, -1),
    )

    from_rank, from_file, to_rank, to_file = unpack(move)

    delta = (to_rank - from_rank, to_file - from_file)
    is_knight_move = delta in _DIRECTIONS

    if not is_knight_move:
        return None

    knight_move_type = _DIRECTIONS.index(delta)
    move_type = _TYPE_OFFSET + knight_move_type

    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    return action

def encode_queen(move: chess.Move):
    _DIRECTIONS = IndexedTuple(
        (+1,  0),
        (+1, +1),
        ( 0, +1),
        (-1, +1),
        (-1,  0),
        (-1, -1),
        ( 0, -1),
        (+1, -1),
    )

    from_rank, from_file, to_rank, to_file = unpack(move)

    delta = (to_rank - from_rank, to_file - from_file)

    is_horizontal = delta[0] == 0
    is_vertical = delta[1] == 0
    is_diagonal = abs(delta[0]) == abs(delta[1])
    is_queen_move_promotion = move.promotion in (chess.QUEEN, None)

    is_queen_move = (
        (is_horizontal or is_vertical or is_diagonal) 
            and is_queen_move_promotion
    )

    if not is_queen_move:
        return None

    direction = tuple(np.sign(delta))
    distance = np.max(np.abs(delta))

    direction_idx = _DIRECTIONS.index(direction)
    distance_idx = distance - 1

    move_type = np.ravel_multi_index(
        multi_index=([direction_idx, distance_idx]),
        dims=(8,7)
    )

    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    return action

def encode_underpromotion(move: chess.Move):
    _TYPE_OFFSET: int = 64
    _DIRECTIONS = IndexedTuple(
        -1,
        0,
        +1,
    )
    _PROMOTIONS = IndexedTuple(
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
    )

    from_rank, from_file, to_rank, to_file = unpack(move)

    is_underpromotion = (
        move.promotion in _PROMOTIONS 
        and from_rank == 6 
        and to_rank == 7
    )

    if not is_underpromotion:
        return None

    delta_file = to_file - from_file

    direction_idx = _DIRECTIONS.index(delta_file)
    promotion_idx = _PROMOTIONS.index(move.promotion)

    underpromotion_type = np.ravel_multi_index(
        multi_index=([direction_idx, promotion_idx]),
        dims=(3,3)
    )

    move_type = _TYPE_OFFSET + underpromotion_type

    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    return action

def encode_move(move: str, board) -> int:
    move = chess.Move.from_uci(move)

    split_string = board.split()
    if split_string[1] == 'b':
        move = rotate(move)

    action = encode_queen(move)

    if action is None:
        action = encode_knight(move)

    if action is None:
        action = encode_underpromotion(move)

    if action is None:
        raise ValueError(f"{move} is not a valid move")

    return action