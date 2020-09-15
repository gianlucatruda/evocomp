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

SAVEPATH = 'saved_instances'
ENEMIES = [1, 3, 5]
REPEATS = 10  # TODO make 10

# Make sure savepath exists
if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH)

# Initialise a dictionary for saving best genomes
best_performers = {}

for ea_instance in [MyDemoEAInstance, ]:
    # Instantiate nested dictionary for this instance
    best_performers[str(ea_instance)] = {e: [] for e in ENEMIES}
    for enemy in ENEMIES:
        for repeat in range(REPEATS):
            pop, logs, bests = ea_instance(enemies=[enemy]).evolve()
            best_fitness = np.max(list(bests.keys()))
            top_genome = bests[best_fitness]
            # Store best genome from run
            best_performers[str(ea_instance)][enemy].append(top_genome)

# Save the best performers to JSON
now = datetime.now().strftime("%m-%d-%H_%M_%S")  # Timestamp
with open(f"{SAVEPATH}/{now}_best_genomes.json", 'w') as file:
    json.dump(best_performers, file)
