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
from auxiliary_funcs import self_adaptive_mutation
from objproxies import CallbackProxy
from simple_controller import player_controller

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

# TODO this feels naughty
global curr_gen
curr_gen = 1

# Set directory for saving logs and experiment states
EXPERIMENT_DIRECTORY = 'experiments/tmp'
if not os.path.exists(EXPERIMENT_DIRECTORY):
    os.makedirs(EXPERIMENT_DIRECTORY)

N_HIDDEN_NEURONS = 10  # how many neurons in the hidden layer of the NN
# Genotype size
IND_SIZE = ((21) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS+1)*5)

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

# We set our operators
toolbox.register("mate", tools.cxUniform, indpb=0.3)
toolbox.register("mutate", self_adaptive_mutation,
                 step=CallbackProxy(lambda: curr_gen))
toolbox.register("select", tools.selTournament, tournsize=3)


def evaluation_wrapper(individual: [float]) -> [float]:
    """Custom fitness function wrapper for the
    adaptive mutation technique.
    """

    # Only first half of genome is evaluated (control weights)
    control_weights = individual[0:int(len(individual)/2)]

    # Use the evoman evaluation function we've defined
    fitness = evo_utils.evaluate(control_weights,
                                 player_controller=nn_controller,
                                 experiment_name=EXPERIMENT_DIRECTORY)
    return fitness


# We then add our custom evaluation function to our toolbox
toolbox.register("evaluate", evaluation_wrapper)

# Initialise our population
pop = toolbox.population(n=POPSIZE)

# Create Hall of Fame (keeps N best individuals over all history)
hall_of_fame = tools.HallOfFame(maxsize=HOFSIZE)

# Make a single statistics object to monitor fitness and genome stats
stats = evo_utils.make_custom_statistics()

now = datetime.now().strftime("%m-%d-%H_%M_%S")  # Timestamp
print(f"\n{now}: Running EA for {NGEN} generations...\n")
try:
    final_population, logbook = algorithms.eaSimple(
        pop, toolbox, CXPB, MUTPB, NGEN,
        halloffame=hall_of_fame, stats=stats, verbose=True)
except KeyboardInterrupt as e:
    print(f"\nINTERRUPT: Stopping EA early. Stats will be lost.\n")
else:  # Only executed if NO exceptions
    # Get dataframe of stats
    df_stats = evo_utils.compile_stats(logbook)
    print('\nFinal results:\n')
    print(df_stats, end='\n\n')
    # Save the statistics to CSV
    file_name = f"{EXPERIMENT_DIRECTORY}/{now}_logbook.csv"
    df_stats.to_csv(file_name)
    print(f"Statistics saved to '{file_name}'.")
finally:  # Always executed
    # Get dictionary of best_individuals -> {fitness: [genome], ...}
    best_individuals = evo_utils.compile_best_individuals(hall_of_fame)
    top_scores = sorted(list(best_individuals.keys()), reverse=True)
    if len(top_scores) > 0:
        print(f"\nTop scores: {top_scores}")
        # Save the best individuals to JSON file
        file_name = f"{EXPERIMENT_DIRECTORY}/{now}_best_individuals.json"
        with open(file_name, 'w') as file:
            json.dump(best_individuals, file)
        print(f"Best individuals saved to '{file_name}'.")
