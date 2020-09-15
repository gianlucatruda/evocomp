import sys
sys.path.insert(0, 'evoman')
import os
from abc import ABC

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from scipy.spatial import distance

from evoman.controller import Controller
from evoman.environment import Environment
from simple_controller import player_controller


def evaluate(individual: list,
             player_controller: Controller,
             experiment_name='experiments/tmp',
             enemies=[2],) -> list:
    """Custom evaluation function based on evoman specialist (NN)

    Parameters
    ----------
    individual : list
        The genome of the individual.
    player_controller : Controller
        The evoman controller instance to use for evaluation.
    experiment_name : str
        The name of the experiment (usually a path like 'experiments/tmp').
        Used for saving state and evoman logs.
    enemies : list
        Which enemy/enemies to fight against, default [2]


    Returns
    -------
    list
        The fitness score(s) for the individual
    """

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=experiment_name,
        multiplemode="no",
        enemies=enemies,
        playermode="ai",
        enemymode="static",
        player_controller=player_controller,
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


def diversity_L1(pop: list) -> float:
    """Calculate mean Manhattan (L1) distance between every pair
    of genomes in the population.

    Parameters
    ----------
    pop : list
        The population. A list of individual genomes.

    Returns
    -------
    float
        The mean mutual Manhattan distance
    """
    dists = [distance.cityblock(i, j) for i in pop for j in pop if i != j]
    return np.mean(dists)


def make_custom_statistics() -> tools.MultiStatistics:
    """Makes custom fitness and genome statistics using DEAP tools.
    Combines them into a single MultiStatistics object and returns.

    Returns
    -------
    tools.MultiStatistics
        Full complement of fitness and genome statistics, ready for use
        with DEAP framework.
    """

    # Create a statistics object to log stats about `fitness` to our logbook
    fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    # Add some keys and functions to the statistics object
    fitness_stats.register("min_fitness", np.min)
    fitness_stats.register("mean_fitness", np.mean)
    fitness_stats.register("std_fitness", np.std)
    fitness_stats.register("max_fitness", np.max)

    # Create a statistics object to log stats about `genome` to our logbook
    genome_stats = tools.Statistics(key=lambda ind: ind)
    # Add some keys and functions to the statistics object
    genome_stats.register("diversity", diversity_L1)
    genome_stats.register(
        "mean_genome_size", lambda pop: np.mean([len(i) for i in pop]))

    # Make a single statistics object from both our stats objects
    stats = tools.MultiStatistics(fitness=fitness_stats, genome=genome_stats)

    return stats


def compile_stats(logbook: tools.Logbook) -> pd.DataFrame:
    """Compile DEAP logs into a dataframe of statistics about fitness
    and genome values over generations.

    Parameters
    ----------
    logbook : tools.Logbook
        The logbook object generated by DEAP during the EA run.

    Returns
    -------
    pd.DataFrame
        A dataframe of fitness and genome statistics indexed on
        `gen` (generations).
    """

    # Build dataframe of results for each key
    df_fitness = pd.DataFrame(logbook.chapters['fitness'])
    df_genome = pd.DataFrame(logbook.chapters['genome'])
    # Combine the dataframes on `gen` and set index
    df_stats = df_fitness.merge(df_genome, on=['gen', 'nevals'])
    df_stats.set_index('gen', inplace=True)

    return df_stats


def compile_best_individuals(hall_of_fame: tools.HallOfFame) -> dict:
    """Makes a dictionary of {<fitness>: [<genome>], ...} items
    representing the top N individuals from all generations.

    Parameters
    ----------
    hall_of_fame : tools.HallOfFame
        DEAPs hall of fame object generated during EA run.

    Returns
    -------
    dict
        {<fitness>: [<genome>], ...}
        Keys are fitness scores. Values are the corresponding
        genome (list of floats).
    """

    best_individuals = {}
    for key, item in zip(hall_of_fame.keys, hall_of_fame.items):
        best_individuals[key.values[0]] = item

    return best_individuals


class BaseEAInstance(ABC):
    """Base class for EA instances.
    """

    def __init__(self, experiment_directory='experiments/tmp'):
        self.experiment_directory = experiment_directory
        self.enemies = None
        self.player_controller = None
        self.toolbox = None
        self.population = None
        self.hall_of_fame = None
        self.stats = None
        self.final_population = None
        self.logbook = None
        self.best_individuals = None
        self.top_scores = None

        # Set directory for saving logs and experiment states
        if not os.path.exists(self.experiment_directory):
            os.makedirs(self.experiment_directory)

    def evolve(self):
        raise NotImplementedError()

    def __repr__(self):
        params = {
            'enemies': self.enemies,
            'toolbox': self.toolbox.__repr__(),
            'pop_size': len(self.population),
            'top_fitness': np.max(self.top_scores),
        }
        return f"EA instance: {params}"
