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


def is_same_species(ind1, ind2, threshold):
    """Returns `True` if individuals' genomes are within the same-species threshold
    """
    return evo_utils.diversity_L1([ind1, ind2]) < threshold


def speciation_crossover(ind1, ind2, indpb, threshold):
    """If same species, executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped according to the
    *indpb* probability.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    """
    if is_same_species(ind1, ind2, threshold):
        size = min(len(ind1), len(ind2))
        for i in range(size):
            if random.random() < indpb:
                ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


class SpeciationEA(BaseEAInstance):
    def __init__(self,
                 experiment_directory='experiments/tmp',
                 enemies=[2],
                 CXPB=0.5,
                 MUTPB=0.3,
                 NGEN=15,
                 POPSIZE=30,
                 HOFSIZE=5,
                 SPECIES_THRESH=0.6,
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
        # L1 diversity threshold to consider same species
        self.SPECIES_THRESH = SPECIES_THRESH

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

        # We set our custom mating/crossover operator
        self.toolbox.register("mate", speciation_crossover,
                              indpb=0.5, threshold=self.SPECIES_THRESH)
        # self.toolbox.register("mate", tools.cxUniform, indpb=0.5)

        # We set our operators
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

        self.final_population, self.logbook = self.speciationEA(
            verbose=verbose)

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

    def speciationEA(self, verbose=True):
        """This algorithm is a modified version of the simplest evolutionary algorithm as
        presented in chapter 7 of [Back2000]_.
        Retrieved from
        https://github.com/DEAP/deap/blob/38083e5923274dfe5ecc0586eb295228f8c99fc4/deap/algorithms.py
        """

        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + \
            (self.stats.fields if self.stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if self.hall_of_fame is not None:
            self.hall_of_fame.update(self.population)

        record = self.stats.compile(self.population) if self.stats else {}
        self.logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(self.logbook.stream)

        # Begin the generational process
        for gen in range(1, self.NGEN + 1):
            # Select the next generation individuals
            offspring = self.toolbox.select(
                self.population, len(self.population))
            offspring = [self.toolbox.clone(ind) for ind in self.population]

            # Apply crossover on the offspring
            for i in range(1, len(offspring), 2):
                if random.random() < self.CXPB:
                    offspring[i - 1], offspring[i] = self.toolbox.mate(
                        offspring[i - 1], offspring[i])
                    del offspring[i -
                                  1].fitness.values, offspring[i].fitness.values

            # Apply mutation on the offspring
            for i in range(len(offspring)):
                if random.random() < self.MUTPB:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if self.hall_of_fame is not None:
                self.hall_of_fame.update(offspring)

            # Replace the current population by the offspring
            self.population[:] = offspring

            # Append the current generation statistics to the logbook
            record = self.stats.compile(self.population) if self.stats else {}
            self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(self.logbook.stream)

        return self.population, self.logbook


if __name__ == "__main__":
    os.putenv("SDL_VIDEODRIVER", "fbcon")
    os.environ["SDL_VIDEODRIVER"] = 'dummy'
    ea_instance = SpeciationEA(experiment_directory='experiments/tmp',
                               enemies=[2], CXPB=0.5,
                               MUTPB=0.3, NGEN=10, POPSIZE=20,
                               HOFSIZE=5, multiplemode="no")
    final_population, stats, best = ea_instance.evolve(verbose=True)
