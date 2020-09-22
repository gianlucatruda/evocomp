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
from EA_demo import BaselineEAInstance


os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

SAVEPATH = 'results'
ENEMIES = [1, 3, 5]
VERBOSE = True
"""
# Testing baseline
base = BaselineEAInstance(enemies=ENEMIES)
base.evolve(verbose=VERBOSE)
"""
# Testing the custom simpleEA class
ea = CustomEASimple(enemies=ENEMIES)
ea.evolve(verbose=VERBOSE)
