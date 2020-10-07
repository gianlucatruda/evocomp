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


N_HIDDEN_NEURONS = 10  # how many neurons in the hidden layer of the NN
# Genotype size
IND_SIZE = 21 * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5


class CommaEAInstance(BaseEAInstance):
    def __init__(self,
                 experiment_directory='experiments/tmp',
                 enemies=[2],
                 CXPB=0.5,
                 MUTPB=0.3,
                 NGEN=20,
                 POPSIZE=20,
                 HOFSIZE=5,
                 LAMBDA=None,
                 multiplemode="no"):
        self.experiment_directory = experiment_directory
        self.enemies = enemies
        self.multiplemode = multiplemode

        # Define some NB parameters for our EA
        self.CXPB = CXPB  # Probability of mating two individuals
        self.MUTPB = MUTPB  # Probability of mutating an individual
        self.NGEN = NGEN  # The number of generations
        # Number of individuals per generation (population size)
        self.POPSIZE = POPSIZE
        self.HOFSIZE = HOFSIZE  # Maximum size of hall of fame (best genomes)

        self.MU = self.POPSIZE # Number of individuals to select for next generation
        self.LAMBDA = LAMBDA # Number of children to make each generation

        if self.LAMBDA is None:
            self.LAMBDA = int(2 * self.MU)

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
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", tools.mutGaussian,
                              mu=0, sigma=1, indpb=MUTPB)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # We then add our custom evaluation function to our toolbox
        self.toolbox.register("evaluate",
                              evo_utils.evaluate,
                              experiment_name=self.experiment_directory,
                              player_controller=self.player_controller,
                              enemies=self.enemies,
                              multiplemode=self.multiplemode,
                              )

        # Initialise our population
        self.population = self.toolbox.population(n=POPSIZE)

        # Create Hall of Fame (keeps N best individuals over all history)
        self.hall_of_fame = tools.HallOfFame(maxsize=HOFSIZE)

        # Make a single statistics object to monitor fitness and genome stats
        self.stats = evo_utils.make_custom_statistics()

    def evolve(self, verbose=True):

        if verbose:
            print(f"\nRunning EA for {self.NGEN} generations...\n")

        self.final_population, self.logbook = algorithms.eaMuCommaLambda(
            self.population, self.toolbox,
            self.MU, self.LAMBDA,
            self.CXPB, self.MUTPB, self.NGEN,
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

    def evaluate(self, individual: [float], *args, **kwargs) -> [float]:
        """Runs the evaluation process for this EA on the specified individual.

        Parameters
        ----------
        individual : list
            The genome of the individual to evaluate.

        Returns
        -------
        [float]
            The fitness score (or gain score) wrapped in a list.
        """

        return evo_utils.evaluate(
            individual,
            experiment_name=self.experiment_directory,
            player_controller=self.player_controller,
            enemies=self.enemies,
            *args,
            **kwargs,
        )
