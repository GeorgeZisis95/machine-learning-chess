import argparse
import multiprocessing
import concurrent.futures
import numpy as np

from data_collection import generate_games
from data_encoding import encode_data
from data_training import train

np.set_printoptions(threshold = np.inf)

parser = argparse.ArgumentParser(description='Chess Engine with Machine Learning')
parser.add_argument("--get_expert_data", "-ged", action="store_true", help="Begin the process of getting and saving expert data")
parser.add_argument("--encode_data", "-ed", action="store_true", help="Encode existing data for training")
parser.add_argument("--start_training", "-t", action="store_true", help="Train the model")
args = parser.parse_args()

NUM_GAMES = 1000
NUM_WORKERS = 8

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    if args.get_expert_data:
        for _ in range(NUM_GAMES//8):
            with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = [executor.submit(generate_games, parallel_tasks) for parallel_tasks in range(1, NUM_WORKERS+1)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
    if args.encode_data:
        encode_data()
    if args.start_training:
        train()