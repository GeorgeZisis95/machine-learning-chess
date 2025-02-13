import numpy as np
import os

#TODO: Crash all data from the different games into one file

total_states, total_actions = [], []
files = os.listdir('expert_data_collection')
for idx, f in enumerate(files):
    states_actions = np.load(f'expert_data_collection/{f}', allow_pickle=True)
    states = states_actions[:,0]
    actions = states_actions[:,1]
    total_states.extend(states)
    total_actions.extend(actions)

from collections import defaultdict
duplicates = defaultdict(list)
for i,item in enumerate(total_states):
    duplicates[item].append(i)
duplicates = {k:v for k,v in duplicates.items() if len(v)>1}

for key, value in duplicates.items():
    print(key, value)

for i in [0, 101, 198, 299, 400, 488, 589, 690]:
    print(total_actions[i])
