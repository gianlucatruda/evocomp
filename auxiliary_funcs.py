import sys
sys.path.insert(0, 'evoman')
import os
import json
import numpy as np
from math import exp, sqrt


import random
from datetime import datetime
from deap import algorithms, base, creator, tools

import evo_utils
from objproxies import CallbackProxy
from simple_controller import player_controller

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = 'dummy'

N_HIDDEN_NEURONS = 10  # how many neurons in the hidden layer of the NN
# Genotype size
IND_SIZE = ((21) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS+1)*5) * 2

# Define some NB parameters for our EA
CXPB = 0.5  # Probability of mating two individuals
MUTPB = 0.2  # Probability of mutating an individual
NGEN = 5  # The number of generations
POPSIZE = 10  # Number of individuals per generation (population size)
HOFSIZE = 5  # Maximum size of hall of fame (best genomes)


def evaluation_wrapper(individual: [float], *args, **kwargs) -> [float]:
    """Custom fitness function wrapper for the
    adaptive mutation technique.
    """

    # Only first half of genome is evaluated (control weights)
    control_weights = individual[0:int(len(individual)/2)]

    # Use the evoman evaluation function we've defined
    fitness = evo_utils.evaluate(control_weights, *args, **kwargs)

    return fitness


def self_adaptive_mutation(individual: list, step: int) -> list:
    """Custom mutation function based on "Multi-step Self-Adaptation".

    Using A.E. Eiben's paper as reference for the concrete implementation
    http://www.few.vu.nl/~gks290/papers/SigmaAdaptationComparison.pdf

    Parameters
    ----------
    individual : list
        The genotype of the individual
    step: int
        The current generation number
    Returns
    -------
    list
        The mutated genotype of the individual
    """

    idxSigma1 = int(len(individual)/2)
    # REMARK - didn't saw a better way to initialize the sigma's to 0.8 with the underlying structure in the deap_specialist file
    if step == 1:
        for i in range(idxSigma1, len(individual)):
            individual[i] = 0.8

    for i in range(idxSigma1, len(individual)):
        # Updating first the sigma
        # sometimes it comes negative from the recombination operation
        individual[i] = abs(individual[i])
        individual[i] = individual[i] * exp(np.random.normal(
            0, 1/(sqrt(2*step))) + np.random.normal(0, 1/(sqrt(2*sqrt(step)))))

        # Updating secondly the x
        individual[i-idxSigma1] = individual[i-idxSigma1] + \
            np.random.normal(0, individual[i])

    return individual,


class CustomEASimple(evo_utils.BaseEAInstance):
    def __init__(self, enemies, experiment_directory='experiments/customEA_adaptive'):
        self.experiment_directory = experiment_directory
        self.enemies = enemies

        # TODO this feels naughty
        global curr_gen
        curr_gen = 1

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
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # We set our operators
        self.toolbox.register("mate", tools.cxUniform, indpb=0.3)
        self.toolbox.register("mutate", self_adaptive_mutation,
                        step=CallbackProxy(lambda: curr_gen))
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # We then add our custom evaluation function to our toolbox
        self.toolbox.register("evaluate",
                              evaluation_wrapper,
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

    def evolve(self, verbose):
        if verbose:
            print(f"\nRunning EA for {NGEN} generations...\n")

        self.final_population, self.logbook = eaSimple(
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

        return evaluation_wrapper(
            individual,
            experiment_name=self.experiment_directory,
            player_controller=self.player_controller,
            enemies=self.enemies,
            *args,
            **kwargs,
        )


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    Retrieved from Retrieved from https://github.com/DEAP/deap/blob/38083e5923274dfe5ecc0586eb295228f8c99fc4/deap/algorithms.py
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook
