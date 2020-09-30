import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Enable fast mode on some *NIX systems
os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'
# Disable pygame load message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import evo_utils
from EA_adaptive import CustomEASimple
from EA_base import BaselineEAInstance

sys.path.insert(0, 'evoman')

SAVEPATH = 'results'
ENEMIES = [[1, 3], [2, 4]]
REPEATS = 10
VERBOSE = False
INSTANCES = [BaselineEAInstance]

# Automatically infers specialist or generalist from ENEMIES nesting
if any(isinstance(i, list) for i in ENEMIES):
    multi = "yes"
else:
    multi = "no"

# Make sure savepath exists
if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH)

# Initialise a dictionary for saving best genomes
best_performers = {}
# Initialise list for saving statistic logs
results = []


for ea_instance in INSTANCES:
    print(f"\nInstance: {ea_instance}")
    # Instantiate nested dictionary for this instance
    best_performers[str(ea_instance)] = {str(e): [] for e in ENEMIES}
    for enemies in ENEMIES:
        _enemies = [enemies]
        if multi == "yes":
            _enemies = enemies
        print(f"\nEnemies: {_enemies}")
        for repeat in tqdm(range(REPEATS), desc='Repeats'):
            pop, stats, bests = ea_instance(
                enemies=_enemies,
                multiplemode=multi).evolve(verbose=VERBOSE)

            # Save the stats (fitness and genome)
            stats['enemies'] = str(enemies)
            stats['ea_instance'] = str(ea_instance)

            # Store the best genome
            best_fitness = np.max(list(bests.keys()))
            top_genome = bests[best_fitness]

            best_performers[str(ea_instance)][str(enemies)].append(top_genome)
            results.append(stats)


now = datetime.now().strftime("%m-%d-%H_%M_%S")  # Timestamp

# Combine the overall statistics
df_results = pd.concat(results)
# Save ALL the results to CSV
f_name = f"{SAVEPATH}/{now}_online_results.csv"
df_results.to_csv(f_name)
print(f"\nResults saved to {f_name}")

# Save the best performers to JSON
with open(f"{SAVEPATH}/{now}_best_genomes.json", 'w') as file:
    json.dump(best_performers, file)


def evolve_island(ea_instance, enemy):
    raise DeprecationWarning("Parallelisation is unreliable.")
