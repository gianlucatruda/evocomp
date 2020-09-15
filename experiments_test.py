import sys
sys.path.insert(0, 'evoman')

import os
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd

import evo_utils
from simple_controller import player_controller
from EA_demo import MyDemoEAInstance, evaluation_wrapper

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

REPEATS = 5

if __name__ == '__main__':
    # Read the results from the specified path
    if len(sys.argv) != 2:
        print("\nPlease provide the path to the JSON files as an argument")
        sys.exit()

    # Read in the dictionary of best performers
    read_path = sys.argv[1]
    with open(read_path, 'r') as file:
        best_performers = json.load(file)

    for ea_instance in best_performers.keys():
        print(f"Instance: {ea_instance}")
        for enemy in best_performers[ea_instance].keys():
            print(f"Enemy: {enemy}")
            for repeat in range(REPEATS):
                # TODO I really can't make sense of the task description and what she wants yet
                genome = best_performers[ea_instance][enemy][0] # Pick just first genome
                fitness = evaluation_wrapper(
                    genome,
                    player_controller=player_controller(10),  # TODO remove hardcoding
                )[0]
                print(fitness)
