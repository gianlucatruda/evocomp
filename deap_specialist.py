
import sys
sys.path.insert(0, 'evoman')
import os
import numpy as np
import random
from deap import base, creator, tools, algorithms

from simple_controller import player_controller
from evoman.environment import Environment

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

"""
Reference: https://deap.readthedocs.io/en/master/overview.htm
"""

# Set directory for saving logs and experiment states
EXPERIMENT_DIRECTORY = 'experiments/tmp'
if not os.path.exists(EXPERIMENT_DIRECTORY):
    os.makedirs(EXPERIMENT_DIRECTORY)

N_HIDDEN_NEURONS = 10  # how many neurons in the hidden layer of the NN
# number of weights for multilayer with 10 hidden neurons (assuming 20 sensors)
IND_SIZE = (21) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS+1)*5
# Initialise the controller (neural network) for our AI player
nn_controller = player_controller(N_HIDDEN_NEURONS)

# Create a fitness maximises (single attribute)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Create an individual class (our solutions) using our fitness function
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create a new "toolbox"
toolbox = base.Toolbox()
# Attributes for our individuals are initialised as random floats
toolbox.register("attribute", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
# Define a population of these individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# We set our operators (out-of-the-box ones, for now)
toolbox.register("mate", tools.cxUniform, indpb=0.3)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


# We define our evaluation function
def evaluate(individual: list) -> list:
    """Custom evaluation function based on evoman specialist demo (NN)

    Parameters
    ----------
    individual : list
        The genotype of the individual

    Returns
    -------
    list
        The fitness score(s)
    """

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=EXPERIMENT_DIRECTORY,
        multiplemode="no",
        enemies=[2],                 # 1 to 8
        playermode="ai",
                      player_controller=nn_controller,
        enemymode="static",
        player_controller=nn_controller,
                      level=2,
        speed="fastest",
        inputscoded="no",            # yes or no
        randomini="no",              # yes or no
        sound="off",                 # on or off
        logs="off",                   # on or off
        savelogs="yes",              # yes or no
        clockprec="low",
        timeexpire=3000,             # integer
        overturetime=100,            # integer
        solutions=None,              # any
        fullscreen=False,            # True or False
    )

    # Run the simulation (score fitness)
    fitness, player_life, enemy_life, sim_time = env.play(
        pcont=np.array(individual))

    return [fitness]


# We then add the evaluation function to our toolbox
toolbox.register("evaluate", evaluate)

# Initialise our population
pop = toolbox.population(n=100)

# Define some NB parameters for our EA
CXPB = 0.5  # Probability of mating two individuals
MUTPB = 0.2  # Probability of mutating an individual
NGEN = 10  # The number of generations

print("Running Simple EA...")
# We will use one of DEAP's provided evolutionary algorithms for now
final_population = algorithms.eaSimple(
    pop, toolbox, CXPB, MUTPB, NGEN, verbose=True)
