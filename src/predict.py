import torch
import chess
import numpy as np

from src.encode import uci_to_index, all_uci_moves, get_canonical_board

ACTION_SIZE = 4544

def mask_actions(indices):
    total_actions = [0] * ACTION_SIZE
    for index in indices:
        total_actions[index] = 1
    return total_actions

def get_model_output(board, model, device):
    model_input = torch.from_numpy(get_canonical_board(board.fen())).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(model_input)
        probs = torch.softmax(logits, dim=1).squeeze()
    return probs.squeeze(0).detach().cpu().numpy()

def filter_policy(probs, mask):
    probs = probs * mask
    probs = probs / np.sum(probs)
    return probs

def get_policy(board, model, device, threshold=0.05):
    legal_moves = [element.uci() for element in board.legal_moves]
    legal_indices = []
    for legal_move in legal_moves:
        legal_indices.append(uci_to_index[legal_move])
    total_actions = mask_actions(legal_indices)
    policy = get_model_output(board, model, device)
    policy = filter_policy(policy, total_actions)
    best_indices = np.where(policy > threshold)[0]
    total_actions = mask_actions(best_indices)
    policy = filter_policy(policy, total_actions)
    return policy

def get_uci_move(board, model, device):
    probs = get_policy(board, model, device)
    uci_move = np.random.choice(all_uci_moves, p=probs)
    return chess.Move.from_uci(uci_move)