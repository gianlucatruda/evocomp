import sys
sys.path.insert(0, 'evoman')

import os
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools

import evo_utils
from evoman.environment import Environment
from simple_controller import player_controller

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

# Set directory for saving logs and experiment states
EXPERIMENT_DIRECTORY = 'experiments/tmp'
if not os.path.exists(EXPERIMENT_DIRECTORY):
    os.makedirs(EXPERIMENT_DIRECTORY)

N_HIDDEN_NEURONS = 10  # how many neurons in the hidden layer of the NN
# number of weights for multilayer with 10 hidden neurons (assuming 20 sensors)
IND_SIZE = (21) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS+1)*5

# Define some NB parameters for our EA
CXPB = 0.5  # Probability of mating two individuals
MUTPB = 0.2  # Probability of mutating an individual
NGEN = 5  # The number of generations
POPSIZE = 10  # Number of individuals per generation (population size)
HOFSIZE = 5  # Maximum size of hall of fame (best genomes)


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


# We then add our custom evaluation function to our toolbox
toolbox.register("evaluate", evo_utils.evaluate,
                 player_controller=nn_controller,
                 experiment_name=EXPERIMENT_DIRECTORY)

# Initialise our population
pop = toolbox.population(n=POPSIZE)

# Create Hall of Fame (keeps N best individuals over all history)
hall_of_fame = tools.HallOfFame(maxsize=HOFSIZE)

# Make a single statistics object to monitor fitness and genome stats
stats = evo_utils.make_custom_statistics()

print(f"\nRunning EA for {NGEN} generations...\n")
# We will use one of DEAP's provided evolutionary algorithms for now
final_population, logbook = algorithms.eaSimple(
    pop, toolbox, CXPB, MUTPB, NGEN,
    halloffame=hall_of_fame, stats=stats, verbose=True)

# Get dataframe of stats
df_stats = evo_utils.compile_stats(logbook)
print('\nFinal results:\n')
print(df_stats)

# Get dictionary of best_individuals -> {fitness: [genome], ...}
best_individuals = evo_utils.compile_best_individuals(hall_of_fame)
top_scores = sorted(list(best_individuals.keys()), reverse=True)
print(f"\nTop scores: {top_scores}")

# Save the statistics to CSV and the best individuals to JSON
now = datetime.now().strftime("%m-%d-%H_%M_%S")  # Timestamp
df_stats.to_csv(f"{EXPERIMENT_DIRECTORY}/{now}_logbook.csv")
with open(f"{EXPERIMENT_DIRECTORY}/{now}_best_individuals.json", 'w') as file:
    json.dump(best_individuals, file)
