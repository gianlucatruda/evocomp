import sys
sys.path.insert(0, 'evoman')

import os
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd

import evo_utils
from EA_demo import MyDemoEAInstance

from joblib import Parallel, delayed
import multiprocessing

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

SAVEPATH = 'results'
ENEMIES = [1, 3, 5]
REPEATS = 10
JOBS = 1  # Leave this as 1 unless you want a world of pain


def evolve_island(ea_instance, enemy):
    pop, stats, bests = ea_instance(enemies=[enemy]).evolve(verbose=False)

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


for ea_instance in [MyDemoEAInstance, ]:
    # Instantiate nested dictionary for this instance
    best_performers[str(ea_instance)] = {e: [] for e in ENEMIES}
    for enemy in ENEMIES:

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
