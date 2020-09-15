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

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

SAVEPATH = 'experiment_run'
ENEMIES = [1, 3, 5]
REPEATS = 10

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
        for repeat in range(REPEATS):
            pop, stats, bests = ea_instance(enemies=[enemy]).evolve()

            # Save the stats (fitness and genome)
            stats['enemy'] = enemy
            stats['ea_instance'] = str(ea_instance)
            results.append(stats)

            # Store the best genome
            best_fitness = np.max(list(bests.keys()))
            top_genome = bests[best_fitness]
            best_performers[str(ea_instance)][enemy].append(top_genome)

now = datetime.now().strftime("%m-%d-%H_%M_%S")  # Timestamp

# Combine the overall statistics
df_results = pd.concat(results)
# Save ALL the results to CSV
df_results.to_csv(f"{SAVEPATH}/{now}_all_results.csv")

# Save the best performers to JSON
with open(f"{SAVEPATH}/{now}_best_genomes.json", 'w') as file:
    json.dump(best_performers, file)
