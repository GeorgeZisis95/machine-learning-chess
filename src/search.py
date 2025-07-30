import numpy as np
import torch
import math

class AlphaNode:
    def __init__(self, game, board, prior=0, value=0, player=1,  parent=None):
        self.game = game
        self.board = board
        self.prior = prior
        self.value = value
        self.player = player
        self.parent = parent

        self.children = {}
        self.visit_count = 0
        self.total_value = 0
    
    def is_fully_expanded(self):
        return len(self.children) == len(self.game.get_legal_actions(self.board))
    
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
    def __init__(self, game, model):
        self.game = game
        self.model = model

    @torch.no_grad()
    def search(self, board, player, num_searches):
        root = AlphaNode(self.game, board, 0, 0, player, parent=None)
        for _ in range(num_searches):
            node = root
            if node.is_fully_expanded():
                node = node.select_child()
            if not self.game.get_terminated(node.board, node.player):
                policy, value = self.get_policy_value(node.board)
                node.value = value
                self.expand(node, policy)
            self.backpropagate(node)
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def expand(self, node, policy):
        for action, prior in enumerate(policy):
            if prior > 0:
                expanded_board = np.copy(self.board)
                expanded_board = self.game.get_next_state(expanded_board, node.player, action)
                child_node = AlphaNode(self.game, expanded_board, prior, 0, node.player * -1, parent=node)
                node.children[action] = child_node
            else:
                continue

    def backpropagate(self, node):
        while node is not None:
            node.visit_count += 1
            node.total_value += node.value
            node = node.parent
    
    def get_policy_value(self, board):
        # The board needs to be reshaped into a 4D tensor with the shape(batch_size, channels, height, width)
        # We use float32 because neural networks typically expect floating-point numbers for input
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        policy, value = self.model(board_tensor)
        value = value.item()
        policy = policy.squeeze(0).detach().cpu().numpy()
        actions = self.game.get_legal_actions(board)
        total_actions = [0] * self.game.columns
        for action in actions:
            total_actions[action] = 1
        policy = policy * total_actions
        policy = policy / np.sum(policy)
        return policy, value