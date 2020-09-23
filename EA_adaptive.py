import sys
sys.path.insert(0, 'evoman')
import os
import json

from math import ceil
import numpy as np
from operator import attrgetter
import random
from datetime import datetime
from deap import algorithms, base, creator, tools

from auxiliary_funcs import self_adaptive_mutation
import evo_utils
from objproxies import CallbackProxy
from simple_controller import player_controller


N_HIDDEN_NEURONS = 10  # how many neurons in the hidden layer of the NN
# Genotype size
IND_SIZE = ((21) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS+1)*5) * 2


def evaluation_wrapper(individual: [float], *args, **kwargs) -> [float]:
    """Custom fitness function wrapper for the
    adaptive mutation technique.
    """

    # Only first half of genome is evaluated (control weights)
    control_weights = individual[0:int(len(individual)/2)]

    # Use the evoman evaluation function we've defined
    fitness = evo_utils.evaluate(control_weights, *args, **kwargs)

    return fitness


class CustomEASimple(evo_utils.BaseEAInstance):
    def __init__(self, enemies=[2], experiment_directory='experiments/customEA_adaptive'):
        self.experiment_directory = experiment_directory
        self.enemies = enemies

        # Initializing the variable to use in the self adaptive mutation
        self.current_gen = 0

        # Define some NB parameters for our EA
        self.CXPB = 0.7  # Probability of mating two individuals
        self.MUTPB = 0.3  # Probability of mutating an individual
        self.NGEN = 10  # The number of generations
        self.POPSIZE = 30  # Number of individuals per generation (population size)
        self.HOFSIZE = 5  # Maximum size of hall of fame (best genomes)

        self.parentSelCons = 0.6 # The best 60 % of individuals is selected to mate
        self.survivorSelCons = 0.05 # The worst 5 % of individuals to replace for the next generation

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
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", self_adaptive_mutation,
                        step=CallbackProxy(lambda: self.current_gen))
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # We then add our custom evaluation function to our toolbox
        self.toolbox.register("evaluate",
                              evaluation_wrapper,
                              experiment_name=self.experiment_directory,
                              player_controller=self.player_controller,
                              enemies=self.enemies,
                              )

        # Initialise our population
        self.population = self.toolbox.population(n=self.POPSIZE)

        # Create Hall of Fame (keeps N best individuals over all history)
        self.hall_of_fame = tools.HallOfFame(maxsize=self.HOFSIZE)

        # Make a single statistics object to monitor fitness and genome stats
        self.stats = evo_utils.make_custom_statistics()

    def evolve(self, verbose=True):
        if verbose:
            print(f"\nRunning EA for {self.NGEN} generations...\n")

        self.eaSimple(verbose=verbose)

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

    def eaSimple(self, verbose=__debug__):
        """This algorithm reproduce the simplest evolutionary algorithm as
        presented in chapter 7 of [Back2000]_.
        Retrieved from
        https://github.com/DEAP/deap/blob/38083e5923274dfe5ecc0586eb295228f8c99fc4/deap/algorithms.py
        """
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + (self.stats.fields if self.stats else [])

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
            self.current_gen = gen
            # Select the next generation individuals
            offspring = self.toolbox.select(self.population, self.POPSIZE)

            # Simple Parent Selection Mechanism
            parents = tools.selBest(offspring, k=int(len(offspring)*self.parentSelCons))
            #parents2 = self.selBest(offspring, k=int(len(offspring)*self.parentSelCons), fit_attr=fitnesses)

            # Vary the pool of individuals
            offspring = algorithms.varAnd(parents, self.toolbox, self.CXPB, self.MUTPB)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Simple Survivor Selection Mechanism
            if (self.POPSIZE + len(offspring)) > int(self.POPSIZE*(1 + self.survivorSelCons)):
                worst = tools.selWorst(offspring, k=ceil((len(offspring) * self.survivorSelCons)))
                for w in worst:
                    # print(w.fitness.values)
                    offspring.remove(w)

            # Update the hall of fame with the generated individuals
            if self.hall_of_fame is not None:
                self.hall_of_fame.update(offspring)

            # Replace the current population by the offspring
            self.population[len(self.population):] = offspring

            # Append the current generation statistics to the logbook
            record = self.stats.compile(self.population) if self.stats else {}
            self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(self.logbook.stream)

        self.final_population = self.population

    def selBest(self, individuals, k, fit_attr):
        """ Optimized version of
        https://github.com/DEAP/deap/blob/38083e5923274dfe5ecc0586eb295228f8c99fc4/deap/tools/selection.py#L27
        """
        indivs = np.array(individuals)
        print(type(fit_attr), fit_attr)
        """
        # Evaluate the individuals with an invalid fitness
        #invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = np.array([0 for i in range(indivs.shape[0])])
        for ind, fit in zip(fitnesses, fit_attr):
            print(ind, fit)
            fitnesses[ind] = fit
        #fitnesses = [ind=fit for ind, fit in map(fitnesses, fit_attr)]
        print(type(fitnesses), fitnesses)
        ind = np.argsort(fitnesses)
        print(indivs.shape, ind.shape)
        indivs = np.take_along_axis(indivs, ind, axis=0)  # same as np.sort(x, axis=0)
        best = indivs[:k]
        return best.tolist()
        """
