import chess
import chess.pgn

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