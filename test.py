import numpy as np
import os

from data_encoding import create_uci_labels

np.set_printoptions(threshold=np.inf)

#TODO: Crash all data from the different games into one file
#TODO: Check for duplicate states 
#TODO: Get the probability table for those states since the action might be different each time
#TODO: Transform action from uci to table of possibilities

total_states, total_actions = [], []
files = os.listdir('expert_data_collection')
for idx, f in enumerate(files):
    states_actions = np.load(f'expert_data_collection/{f}', allow_pickle=True)
    states = states_actions[:,0]
    actions = states_actions[:,1]

    total_states.extend(states)
    total_actions.extend(actions)

from collections import defaultdict

state_action_dict = defaultdict(list)
for i,item in enumerate(state_action_dict):
    state_action_dict[item].append(i)
state_action_dict = {k:v for k,v in state_action_dict.items() if len(v) >= 1}

newdict = {}
for key, value in state_action_dict.items():
    actions = [total_actions[i] for i in value]
    total_repetitions = len(value)
    counts = {}
    for n in actions:
        counts[n] = counts.get(n, 0) + 1
    probs = np.zeros((len(create_uci_labels())))
    for action, count in counts.items():
        get_index = create_uci_labels().index(action)
        correct_probability = count / total_repetitions
        probs[get_index] = correct_probability
    newdict[key] = probs 

for key, value in newdict.items():
    print(value)
    print(key)
    break