import sys
sys.path.insert(0, 'evoman')

import os
import numpy as np
from EA_base import BaselineEAInstance
from deap import tools

from plyer import notification
from datetime import datetime
import json

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

SAVEPATH = 'results'
ENEMIES = [3]
VERBOSE = True
start_time = datetime.now()


base = BaselineEAInstance(enemies=ENEMIES, NGEN=5, POPSIZE=10, CXPB=0.5)
final_population, stats, best = base.evolve(verbose=VERBOSE)

# with open('experiments/tmp/specialist2345.json') as f:
#     data = json.load(f)

for fitness, individual in best.items():
    print(fitness)
    base = BaselineEAInstance(enemies=ENEMIES, NGEN=5, POPSIZE=10, CXPB=0.5)
    f = base.evaluate(individual, speed='normal')
    print(f)


final_uptime = datetime.now() - start_time

notification.notify(
    title='Experiments finished',
    message=f'It took {final_uptime.seconds} s',
    #app_name='Here is the application name',
    #app_icon='path/to/the/icon.' + ('ico' if platform == 'win' else 'png')
)