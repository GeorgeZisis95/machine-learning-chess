import numpy as np
import os

from collections import defaultdict
from data_encoding import create_uci_labels, get_canonical_board

np.set_printoptions(threshold=np.inf)

total_states, total_actions = [], []
files = os.listdir('expert_data_collection')
for idx, f in enumerate(files):
    states_actions = np.load(f'expert_data_collection/{f}', allow_pickle=True)
    states = states_actions[:,0]
    actions = states_actions[:,1]

    total_states.extend(states)
    total_actions.extend(actions)

occurences_dict = defaultdict(list)
for i,item in enumerate(total_states):
    occurences_dict[item].append(i)
occurences_dict = {k:v for k,v in occurences_dict.items() if len(v) >= 1}

state_prob_dict = {}
for current_state, indeces in occurences_dict.items():
    actions = [total_actions[i] for i in indeces]
    total_repetitions = len(indeces)
    counts = {}
    for n in actions:
        counts[n] = counts.get(n, 0) + 1
    probs = np.zeros((len(create_uci_labels())))
    for action, count in counts.items():
        get_index = create_uci_labels().index(action)
        correct_probability = count / total_repetitions
        probs[get_index] = correct_probability
    state_prob_dict[current_state] = probs 

encoded_states, encoded_actions = [], []
for state, action in state_prob_dict.items():
    encoded_states.append(get_canonical_board(state))
    encoded_actions.append(action)

if not os.path.isdir('encoded_data_collection'):
    os.mkdir('encoded_data_collection')
np.save(f"encoded_data_collection/features", encoded_states)
np.save(f"encoded_data_collection/labels", encoded_actions)