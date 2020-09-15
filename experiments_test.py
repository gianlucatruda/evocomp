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

REPEATS = 5

if __name__ == '__main__':
    # Read the results from the specified path
    if len(sys.argv) != 2:
        print("Please provide the path to the JSON files as an argument")
        sys.exit()

    # Read in the dictionary of best performers
    read_path = sys.argv[1]
    with open(read_path, 'r') as file:
        best_performers = json.load(file)

    for ea_instance in best_performers.keys():
        for enemy in best_performers[ea_instance].keys():
            for repeat in range(REPEATS):

                # TODO evaluate best_performers[ea_instance][enemy]
                pass
