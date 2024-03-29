import sys
sys.path.insert(0, 'evoman')
import os
from abc import ABC
import json
from pathlib import Path

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from scipy.spatial import distance

from evoman.controller import Controller
from evoman.environment import Environment
from simple_controller import player_controller


def genome_to_txt(genome, filepath, strict=True):
    """Saves `genome` to `filepath` in competition format.
    """

    # Make sure genome is numpy array
    _genome = np.array(genome)

    if strict and _genome.shape != (265, ):
        raise ValueError(
            f"Genome must be of shape (265,), but is of shape {_genome.shape}")

    # Ensure path exists
    path = Path(filepath)
    if not os.path.exists(str(path.parent)):
        os.makedirs(str(path.parent))

    # Save to txt file
    np.savetxt(str(path), _genome)


def init_population(pop_dtype, ind_dtype, filenames):
    if type(filenames) != list:
        filenames = [filenames]
    inds = []
    for filename in filenames:
        if filename.split('.')[-1] == 'json':
            print('Processing json file')
            with open(filename, "r") as f:
                inds += json.load(f).values()
        elif filename.split('.')[-1] == 'txt':
            print('Processing txt file')
            inds += [np.loadtxt(filename)]
    return pop_dtype(ind_dtype(ind) for ind in inds)


def evaluate(individual: list,
             player_controller: Controller,
             experiment_name='experiments/tmp',
             enemies=[2],
             metric='fitness',
             speed='fastest',
             multiplemode="no") -> list:
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
    metrix : str
        Whether to return 'fitness' score or 'gain' score, default 'fitness'


    Returns
    -------
    list
        The fitness score(s) for the individual (if metric='fitness').
        Or, the gain score for the individual (if metric='gain').
    """

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=experiment_name,
        multiplemode=multiplemode,
        enemies=enemies,
        playermode="ai",
        enemymode="static",
        player_controller=player_controller,
        level=2,
        speed=speed,  # "normal",
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

    if metric == 'gain':
        return [player_life - enemy_life]

    return [fitness]


def diversity_L1(pop: list) -> float:
    """Calculate mean Manhattan (L1) distance between every pair
    of genomes in the population, divided by mean genome size.

    Parameters
    ----------
    pop : list
        The population. A list of individual genomes.

    Returns
    -------
    float
        The mean mutual Manhattan distance (scaled by genome size)
    """
    dists = [distance.cityblock(i, j) for i in pop for j in pop if i != j]
    mean_genome_size = np.mean([len(i) for i in pop])

    return np.mean(dists) / mean_genome_size


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


def diversity_comparison(genomes_path: str) -> pd.DataFrame:
    """Generates table comparing diversity amongst best genomes
    for each enemy and EA instance.

    Parameters
    ----------
    genomes_path : str
        Path to JSON file containing results from experiment run.

    Returns
    -------
    pd.DataFrame
        Pivoted results comparing diversity, with one EA per column.
    """
    with open(genomes_path, 'r') as file:
        genomes = json.load(file)

    res = {'Instance': [], 'Enemy': [], 'Diversity': []}

    for instance in genomes.keys():
        inst = genomes[instance]
        for enemy in inst.keys():
            top10 = inst[enemy]
            res['Instance'].append(instance)
            res['Enemy'].append(enemy)
            res['Diversity'].append(diversity_L1(top10))

    df = pd.DataFrame(res)
    df = df.pivot(index='Enemy', columns='Instance', values='Diversity')

    return df


class BaseEAInstance(ABC):
    """Base class for EA instances.
    """

    def __init__(self, experiment_directory='experiments/tmp', enemies=[2], multiplemode="no"):
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

    def evolve(self, verbose=True) -> (list, pd.DataFrame, list):
        """Evolve population under specified parameters.

        Parameters
        ----------
        verbose : bool
            Whether or not to print progress information, default True

        Returns
        -------
        (pop, stats, bests) : (list, pd.DataFrame, list)
            The list of genomes from final generation,
            the stats as a compiled dataframe,
            a list of top N genomes.
        """
        raise NotImplementedError()

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

        raise NotImplementedError()

    def __repr__(self):
        """Make the class print useful info if print() called on it.
        """
        params = {
            'enemies': self.enemies,
            'toolbox': self.toolbox.__repr__(),
            'pop_size': len(self.population),
            'top_fitness': np.max(self.top_scores),
        }
        return f"EA instance: {params}"
