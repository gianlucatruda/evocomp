import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, 'evoman')


OBSERVE = False  # Whether to visualise the output in realtime
SAVEPATH = 'results'
REPEATS = 5

if not OBSERVE:
    # Enable fast mode on some *NIX systems
    os.putenv("SDL_VIDEODRIVER", "fbcon")
    os.environ["SDL_VIDEODRIVER"] = 'dummy'
    # Disable pygame load message
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from EA_adaptive import CustomEASimple
from EA_base import BaselineEAInstance
import evo_utils
from simple_controller import player_controller

INSTANCES = [BaselineEAInstance]


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
    results = {'ea_instance': [], 'enemies': [], 'individual': [], 'gain': []}

    print(f"\nRunning evaluations...\n")

    for ea_instance in INSTANCES:
        print(f"\nInstance: {ea_instance}")
        for enemies in best_performers[str(ea_instance)].keys():

            # Convert from string form to list or int (safer then `eval`)
            _enemies = json.decoder.JSONDecoder().decode(enemies)

            # Determine if specialist or generalist based on enemies
            if isinstance(_enemies, int):
                multi="no"
                _enemies = [_enemies]
            else:
                multi="yes"

            print(f"\nEnemies: {enemies}")
            top_ten = best_performers[str(ea_instance)][enemies]

            for i, individual in enumerate(tqdm(top_ten, desc='individuals')):
                if not OBSERVE:
                    for repeat in range(REPEATS):
                        # Run simulation to compute gain sore
                        gain = ea_instance(enemies=_enemies).evaluate(
                            individual,
                            metric='gain',
                            multiplemode=multi)[0]

                        # Save results to dictionary
                        results['ea_instance'].append(str(ea_instance))
                        results['enemies'].append(enemies)
                        results['individual'].append(i)
                        results['gain'].append(gain)
                else:
                    # Visually inspect performance
                    gain = ea_instance(enemies=_enemies).evaluate(
                        individual,
                        metric='gain',
                        speed="normal",
                        multiplemode=multi)[0]
                    print("Gain:", gain)

# Turn results into a dataframe
df_results = pd.DataFrame(results)

# Save the dataframe
if not OBSERVE:
    now = datetime.now().strftime("%m-%d-%H_%M_%S")  # Timestamp
    f_name = f"{SAVEPATH}/{now}_offline_results.csv"
    df_results.to_csv(f_name)
    print(f"\nResults saved to {f_name}")


print("\n\nResult summary:\n")
print(df_results.drop('individual', axis=1).groupby(
    ['ea_instance', 'enemies']).mean())
