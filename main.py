import argparse
import multiprocessing
import concurrent.futures

from data_collection import generate_games
from data_encoding import encode_data

parser = argparse.ArgumentParser(description='Chess Engine with Machine Learning')
parser.add_argument("--get_expert_data", "-ged", action="store_true", help="Begin the process of getting and saving expert data")
args = parser.parse_args()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    if args.get_expert_data:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(generate_games, parallel_tasks) for parallel_tasks in range(1, 8+1)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

    encode_data()