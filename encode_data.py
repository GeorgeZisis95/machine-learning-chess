import numpy as np
import os 

from encode_actions import encode_move
from encode_states import get_canonical_board

def encode_data():
    files = os.listdir('expert_data_collection')
    for idx, f in enumerate(files):
        actions_and_states = np.load(f'expert_data_collection/{f}', allow_pickle=True)
        actions = actions_and_states[:,0]
        states = actions_and_states[:,1]
        encoded_actions, encoded_states = [], []

        for i in range(len(actions)):
            encoded_states.append(get_canonical_board(states[i]))
            encoded_actions.append(encode_move(actions[i], states[i]))
            
        if not os.path.isdir('encoded_data_collection'):
            os.mkdir('encoded_data_collection')
        np.save(f"encoded_data_collection/actions{idx}", np.array(encoded_actions))
        np.save(f"encoded_data_collection/states{idx}", np.array(encoded_states))