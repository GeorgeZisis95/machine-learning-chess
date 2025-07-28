import chess 

class ChessWrapper:
    def __init__(self):
        self.board = chess.Board()
    
    def get_initial_state(self):
        self.board.reset()
    
    def get_legal_actions(self, board):
        return list(board.legal_moves)

    def get_next_state(self, board, move):
        new_board = board.copy()
        new_board.push(move)
        return new_board
    
    def get_terminated(self, board):
        return board.is_game_over()
    
    def get_result(self, board):
        if board.is_checkmate():
            return 1 if board.turn == chess.BLACK else -1
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            return 0
        return None