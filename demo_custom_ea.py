import sys
sys.path.insert(0, 'evoman')

import os
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd

import evo_utils
from EA_adaptive import CustomEASimple
from EA_base import BaselineEAInstance
from visualiser import inspect_evolution_stats

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

SAVEPATH = 'results'
ENEMIES = [1, 3, 5]
VERBOSE = True

# Testing baseline
base = BaselineEAInstance(enemies=ENEMIES)
final_pop_base, stats_base, best_base = base.evolve(verbose=VERBOSE)

# Testing the self-adaptive EA
ea = CustomEASimple(enemies=ENEMIES)
final_pop_adapt, stats_adapt, best_adapt = ea.evolve(verbose=VERBOSE)

# Visualise
inspect_evolution_stats(stats_base)
inspect_evolution_stats(stats_adapt)
