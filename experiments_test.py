import sys
sys.path.insert(0, 'evoman')

import os
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import evo_utils
from simple_controller import player_controller
from EA_demo import MyDemoEAInstance, evaluation_wrapper

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

SAVEPATH = 'results'
REPEATS = 5

if __name__ == '__main__':
    # Read the results from the specified path
    if len(sys.argv) != 2:
        print("\nPlease provide the path to the JSON files as an argument")
        sys.exit()

    # Make sure savepath exists
    if not os.path.exists(SAVEPATH):
        os.makedirs(SAVEPATH)

    # Read in the dictionary of best performers
    read_path = sys.argv[1]
    with open(read_path, 'r') as file:
        best_performers = json.load(file)

    # Dictionary to save results (later becomes dataframe)
    results = {'ea_instance': [], 'enemy': [], 'individual': [], 'score': []}

    print(f"\nRunning evaluations...\n")

    for ea_instance in best_performers.keys():
        print(f"\nInstance: {ea_instance}")
        for enemy in best_performers[ea_instance].keys():
            print(f"\nEnemy: {enemy}")
            top_ten = best_performers[ea_instance][enemy]
            # TODO could parallelise this part if needed
            for i, individual in enumerate(tqdm(top_ten, desc='individuals')):
                for repeat in range(REPEATS):

                    # Run simulation
                    score = evaluation_wrapper(
                        individual,
                        # TODO remove hardcoding
                        player_controller=player_controller(10))[0]

                    # Save results to dictionary
                    results['ea_instance'].append(ea_instance)
                    results['enemy'].append(enemy)
                    results['individual'].append(i)
                    results['score'].append(score)

# Turn results into a dataframe
df_results = pd.DataFrame(results)

# Save the dataframe
now = datetime.now().strftime("%m-%d-%H_%M_%S")  # Timestamp
f_name = f"{SAVEPATH}/{now}_offline_results.csv"
df_results.to_csv(f_name)
print(f"\nResults saved to {f_name}")


print("\n\nResult summary:\n")
print(df_results.drop('individual', axis=1).groupby(
    ['ea_instance', 'enemy']).mean())
