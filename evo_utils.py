import sys
sys.path.insert(0, 'evoman')

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from scipy.spatial import distance

from simple_controller import player_controller
from evoman.environment import Environment
from evoman.controller import Controller


def evaluate(individual: list,
             player_controller: Controller,
             experiment_name: str) -> list:
    """Custom evaluation function based on evoman specialist (NN)

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
        experiment_name=experiment_name,
        multiplemode="no",
        enemies=[2],                 # 1 to 8
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


def diversity_L1(pop):
    """ Calculate mean Manhattan (L1) distance between every pair
        of genomes in the population.
    """
    dists = [distance.cityblock(i, j) for i in pop for j in pop if i != j]
    return np.mean(dists)


def make_custom_statistics():

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
        "genome_size", lambda pop: np.sum([len(i) for i in pop]))

    # Make a single statistics object from both our stats objects
    stats = tools.MultiStatistics(fitness=fitness_stats, genome=genome_stats)

    return stats


def compile_stats(logbook: tools.Logbook) -> pd.DataFrame:

    # Build dataframe of results for each key
    df_fitness = pd.DataFrame(logbook.chapters['fitness'])
    df_genome = pd.DataFrame(logbook.chapters['genome'])
    # Combine the dataframes on `gen` and set index
    df_stats = df_fitness.merge(df_genome, on=['gen', 'nevals'])
    df_stats.set_index('gen', inplace=True)

    return df_stats


def compile_best_individuals(hall_of_fame: tools.HallOfFame) -> dict:

    best_individuals = {}
    for key, item in zip(hall_of_fame.keys, hall_of_fame.items):
        best_individuals[key.values[0]] = item

    return best_individuals

    # Save logbook data to timestamped CSV file
