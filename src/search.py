import chess
import torch
import math

from src.predict import get_policy_value
from src.encode import all_uci_moves

class AlphaNode:
    def __init__(self, board, prior, value, player, parent=None):
        self.board = board
        self.prior = prior
        self.value = value
        self.player = player
        self.parent = parent

        self.children = {}
        self.visit_count = 0
        self.total_value = 0
    
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def uct_score(self, child, c_puct=2):
        if child.visit_count == 0:
            return float('inf')
        return - child.total_value / child.visit_count + \
            c_puct * math.sqrt(math.log(self.visit_count) / child.visit_count) * child.prior
    
    def select_child(self):
        best_child = None
        best_score = float('-inf')

        for child in self.children.values():
            score = self.uct_score(child)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    
class AlphaTreeSearch: 
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def search(self, board, num_searches):
        player = 1 if board.turn == chess.WHITE else -1
        root = AlphaNode(board, 0, 0, player, parent=None)
        for _ in range(num_searches):
            node = self.select(root)
            value = self.expand(node)
            self.backpropagate(node, value)
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]
    
    def select(self, node):
        while not node.board.is_game_over():
            if not node.is_fully_expanded():
                return node
            node = node.select_child()
        
    def expand(self, node):
        policy, value = get_policy_value(node.board, self.model, self.device)
        for index, prior in enumerate(policy):
            if prior > 0:
                # Get the move for current index
                uci_move = all_uci_moves[index]
                move = chess.Move.from_uci(uci_move)
                # Make the move
                new_board = node.board.copy()
                new_board.push(move)
                # Get the player
                player = 1 if new_board.turn == chess.WHITE else -1
                # Create the corresponding child and add it to children
                child_node = AlphaNode(new_board, prior, 0, player, parent=node)
                node.children[move] = child_node
        return value
    
    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            if node.player == 1:
                node.total_value += value
            elif node.player == -1:
                node.total_value += -value
            node = node.parent