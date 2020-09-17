import sys
sys.path.insert(0, 'evoman')

import os
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd

import evo_utils
from auxiliary_funcs import CustomEASimple


os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

SAVEPATH = 'results'
ENEMIES = [1, 3, 5]
VERBOSE = True

ea = CustomEASimple(enemies=ENEMIES)
