import sys
sys.path.insert(0, 'evoman')

import os
import json
import random
from datetime import datetime

from deap import algorithms, base, creator, tools
import numpy as np

import evo_utils
from evo_utils import BaseEAInstance
from simple_controller import player_controller


N_HIDDEN_NEURONS = 10  # how many neurons in the hidden layer of the NN
# Genotype size
IND_SIZE = 21 * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5


class DynamicEAInstance(BaseEAInstance):
    def __init__(self,
                 experiment_directory='experiments/tmp',
                 enemies=[2],
                 CXPB=1.0,
                 MUTPB=0.8,
                 NGEN=15,
                 POPSIZE=30,
                 HOFSIZE=5,
                 multiplemode="no",
                 seeding_path=None,
                 ):
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
        if seeding_path:
            self.toolbox.register("seeded_population",
                evo_utils.init_population, list, creator.Individual,seeding_path)

        # Attributes for our individuals are initialised as random floats
        self.toolbox.register("attribute", random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attribute, n=IND_SIZE)
        # Define a population of these individuals
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual)

        # We set our custom mating/crossover operator
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)

        # We set our operators
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=MUTPB)
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
        seeded_population = []
        if seeding_path:
            seeded_population = self.toolbox.seeded_population()
        if POPSIZE - len(seeded_population) < 0:
            print(f'Seeded population exceeds the given population size, corrected to {len(seeded_population)}')
        self.population = self.toolbox.population(
            n=max(0, POPSIZE - len(seeded_population))) + seeded_population

        # Create Hall of Fame (keeps N best individuals over all history)
        self.hall_of_fame = tools.HallOfFame(maxsize=HOFSIZE)

        # Make a single statistics object to monitor fitness and genome stats
        self.stats = evo_utils.make_custom_statistics()

    def evolve(self, verbose=True):

        if verbose:
            print(f"\nRunning EA for {self.NGEN} generations on enemies {self.enemies}...\n")

        self.final_population, self.logbook = self.custom_ea(
            verbose=verbose)

        # Get dataframe of stats
        self.stats = evo_utils.compile_stats(self.logbook)

        # Get dictionary of best_individuals -> {fitness: [genome], ...}
        self.best_individuals = evo_utils.compile_best_individuals(self.hall_of_fame)
        self.top_scores = sorted(list(self.best_individuals.keys()), reverse=True)

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

    def custom_ea(self, verbose=True):
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

            # Custom code to dynamically update parameters (annealling-like)
            temperature = (self.NGEN - gen)/self.NGEN

            # Probability of mating two individuals
            cxpb = max(temperature * self.CXPB, 0.4)

            # Probability of mutating an individual
            mutpb = max(temperature * self.MUTPB, 0.1)

            if gen % 5 == 1 and verbose:
                print(f"CXPB: {cxpb}\tMUTPB: {mutpb}")


            # Select the next generation individuals
            offspring = self.toolbox.select(self.population, len(self.population) - 2)
            offspring = [self.toolbox.clone(ind) for ind in self.population]
            # Custom code to include 2 of hall-of-fame in offspring
            offspring.extend([self.toolbox.clone(ind) for ind in random.choices(self.hall_of_fame.items, k=2)])

            # Apply crossover on the offspring
            for i in range(1, len(offspring), 2):
                if random.random() < cxpb:
                    offspring[i - 1], offspring[i] = self.toolbox.mate(
                        offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

            # Apply mutation on the offspring
            for i in range(len(offspring)):
                if random.random() < mutpb:
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

            # Custom code to purge bad individuals when temp is low
            if temperature < 0.4:
                # Purging bottom 5 individuals and seeding from hall-of-fame
                offspring = sorted(offspring, key=lambda x: x.fitness.values[0])[5:]
                offspring.extend([self.toolbox.clone(
                    ind) for ind in random.choices(self.hall_of_fame.items, k=5)])

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
    ea_instance = DynamicEAInstance(
        NGEN=50,
        enemies=[1, 2, 3, 4, 5, 6, 7],
        multiplemode="yes",
        seeding_path="experiments/tmp/10-07-16_52_16_best_individuals.json",
        )
    final_population, stats, best = ea_instance.evolve(verbose=True)
