import sys
sys.path.insert(0, 'evoman')

import os
import json
import random
from datetime import datetime

from deap import algorithms, base, creator, tools

import evo_utils
from evo_utils import BaseEAInstance
from simple_controller import player_controller

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

N_HIDDEN_NEURONS = 10  # how many neurons in the hidden layer of the NN
# Genotype size
IND_SIZE = (21) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS+1)*5

# Define some NB parameters for our EA
CXPB = 0.5  # Probability of mating two individuals
MUTPB = 0.2  # Probability of mutating an individual
NGEN = 5  # The number of generations
POPSIZE = 10  # Number of individuals per generation (population size)
HOFSIZE = 5  # Maximum size of hall of fame (best genomes)




class BaselineEAInstance(BaseEAInstance):
    def __init__(self, experiment_directory='experiments/tmp', enemies=[2]):
        self.experiment_directory = experiment_directory
        self.enemies = enemies

        # Set directory for saving logs and experiment states
        if not os.path.exists(self.experiment_directory):
            os.makedirs(self.experiment_directory)

        # Initialise the controller (neural network) for our AI player
        self.player_controller = player_controller(N_HIDDEN_NEURONS)

        # Create a fitness maximises (single attribute)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # Create an individual class (our solutions) using our fitness function
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create a new "toolbox"
        self.toolbox = base.Toolbox()
        # Attributes for our individuals are initialised as random floats
        self.toolbox.register("attribute", random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attribute, n=IND_SIZE)
        # Define a population of these individuals
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual)

        # We set our operators
        self.toolbox.register("mate", tools.cxUniform, indpb=CXPB)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=MUTPB)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # We then add our custom evaluation function to our toolbox
        self.toolbox.register("evaluate",
                              evo_utils.evaluate,
                              experiment_name=self.experiment_directory,
                              player_controller=self.player_controller,
                              enemies=self.enemies,
                              )

        # Initialise our population
        self.population = self.toolbox.population(n=POPSIZE)

        # Create Hall of Fame (keeps N best individuals over all history)
        self.hall_of_fame = tools.HallOfFame(maxsize=HOFSIZE)

        # Make a single statistics object to monitor fitness and genome stats
        self.stats = evo_utils.make_custom_statistics()

    def evolve(self, verbose=True):

        if verbose:
            print(f"\nRunning EA for {NGEN} generations...\n")

        self.final_population, self.logbook = algorithms.eaSimple(
            self.population, self.toolbox, CXPB, MUTPB, NGEN,
            halloffame=self.hall_of_fame, stats=self.stats, verbose=verbose)

        # Get dataframe of stats
        self.stats = evo_utils.compile_stats(self.logbook)

        # Get dictionary of best_individuals -> {fitness: [genome], ...}
        self.best_individuals = evo_utils.compile_best_individuals(
            self.hall_of_fame)
        self.top_scores = sorted(
            list(self.best_individuals.keys()), reverse=True)

        # Save the statistics to CSV and the best individuals to JSON
        now = datetime.now().strftime("%m-%d-%H_%M_%S")  # Timestamp
        self.stats.to_csv(f"{self.experiment_directory}/{now}_logbook.csv")
        with open(f"{self.experiment_directory}/{now}_best_individuals.json", 'w') as file:
            json.dump(self.best_individuals, file)

        return self.final_population, self.stats, self.best_individuals

