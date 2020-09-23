import sys
sys.path.insert(0, 'evoman')

import os
# Enable fast mode on some *NIX systems
os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'
# Disable pygame load message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import json
import random
from datetime import datetime

import numpy as np
import pandas as pd

import evo_utils
from EA_base import BaselineEAInstance
from EA_adaptive import CustomEASimple

from joblib import Parallel, delayed
import multiprocessing


SAVEPATH = 'results'
ENEMIES = [1, 3, 5]
REPEATS = 10
JOBS = 1  # Leave this as 1 unless you want a world of pain
VERBOSE = True


def evolve_island(ea_instance, enemy):
    pop, stats, bests = ea_instance(enemies=[enemy]).evolve(verbose=VERBOSE)

    # Save the stats (fitness and genome)
    stats['enemy'] = enemy
    stats['ea_instance'] = str(ea_instance)

    # Store the best genome
    best_fitness = np.max(list(bests.keys()))
    top_genome = bests[best_fitness]

    return stats, best_fitness, top_genome


# Make sure savepath exists
if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH)

# Initialise a dictionary for saving best genomes
best_performers = {}
# Initialise list for saving statistic logs
results = []


for ea_instance in [BaselineEAInstance, CustomEASimple]:
    print(f"\nInstance: {ea_instance}")
    # Instantiate nested dictionary for this instance
    best_performers[str(ea_instance)] = {e: [] for e in ENEMIES}
    for enemy in ENEMIES:
        print(f"\nEnemy: {enemy}")
        # Parallel execution of `evolve_island` function for N repeats
        island_results = Parallel(n_jobs=JOBS)(
            delayed(evolve_island)(ea_instance, enemy) for repeat in range(REPEATS))

        for res in island_results:
            stats, best_fitness, top_genome = res
            best_performers[str(ea_instance)][enemy].append(top_genome)
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
