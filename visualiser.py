import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

from evo_utils import diversity_comparison

SAVE_DPI = 300
COLOURS = ['gray', 'blue', 'red']

if not os.path.exists('figs'):
    os.makedirs('figs')

# Set sensible defaults
sns.set()
sns.set_style("ticks")
sns.set_context('paper')


def tidy_instance_name(x):
    return x.split('.')[-1].split("'")[0].split('EA')[0]+' EA'


def format_online_results(df: pd.DataFrame):
    """ Formats the raw online results by calculating
    statistics (mean, std) over all iterations.
    """
    # Make a copy of raw data
    _df = df.copy()

    # Calculate statistics over all iterations
    _df = _df.groupby(['ea_instance', 'enemies', 'gen']).agg(['mean', 'std'])
    # Tidy up the column names and collapse multi-index
    _df.columns = _df.columns.to_series().str.join('_')
    _df.reset_index(inplace=True)

    # Rename instances nicely
    _df['ea_instance'] = _df['ea_instance'].apply(tidy_instance_name)

    return _df


def format_offline_results(df: pd.DataFrame):
    """ Formats the raw offline results by calculating
    statistics (mean) over all iterations.
    """
    # Make a copy of raw data
    _df = df.copy()

    # Calculate statistics over all iterations
    _df = _df.groupby(['ea_instance', 'enemies', 'individual'],
                      as_index=False).mean()
    _df.reset_index(inplace=True)

    # Rename instances nicely
    _df['ea_instance'] = _df['ea_instance'].apply(tidy_instance_name)

    return _df


def specialist_lineplots(df: pd.DataFrame, save_path=None):
    """
    Compare your algorithms by enemy, making a line-plot across the generations, with the average/std (for the mean and the maximum) of the fitness.
    Note that you need to calculate the average (over the 10 runs) of the mean and maximum (over the population in each generation).
    Do one plot by enemy, thus, separately.
    """

    enemies = df['enemies'].unique()
    n_enemies = len(enemies)
    # Get neat versions of the instance names
    instances = df['ea_instance'].unique()

    fig, ax = plt.subplots(n_enemies, 1)

    for i, enemy in enumerate(enemies):
        for j, instance in enumerate(instances):
            colour = COLOURS[j]
            _df = df[(df['enemies'] == enemy) & (
                df['ea_instance'] == instance)]
            _df.plot.line(x='gen', y='mean_fitness_mean', yerr='mean_fitness_std',
                          ax=ax[i], c=colour, label=f'{instance} mean fitness', alpha=0.7, markersize=5, capsize=2)
            _df.plot.line(x='gen', y='max_fitness_mean', yerr='max_fitness_std',
                          ax=ax[i], c=colour, linestyle='--', label=f'{instance} max fitness', alpha=0.7, markersize=5, capsize=2)
            ax[i].set_title(f'Training enemies: {enemy}')
            ax[i].set_xlabel('')
            handles, labels = ax[i].get_legend_handles_labels()
            ax[i].get_legend().remove()

    fig.legend(handles, labels,
        framealpha=1.0,
        loc='lower center',
        fancybox=False,
        shadow=False,
        ncol=3)

    plt.xlabel('Generation')
    plt.ylabel('Fitness score')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)
    plt.show()


def specialist_boxplots(df: pd.DataFrame, save_path=None):
    """
    Compare your algorithms by enemy, testing 5 times your final best solution for each of the 10 independent runs, and present the individual gain in box-plots.
    Note that you need to calculate the means of the 5 times for each solution of the algorithm for the enemy, and these means are the values that will be points in the box-plot. In summary, it is a total of 3 pairs of box-plots (so 6 boxes), being one pair per enemy.
    """

    enemies = df['enemies'].unique()
    n_enemies = len(enemies)

    fig, ax = plt.subplots(1, n_enemies)
    for i, enemy in enumerate(enemies):
        _df = df[df['enemies'] == enemy]
        _df.boxplot(column='gain', by='ea_instance', grid=False, ax=ax[i])
        ax[i].set_title(f'Training enemies: {enemy}')
        ax[i].set_ylabel('Gain score')
        ax[i].set_xlabel('')
        ax[i].tick_params(axis='x', labelrotation=90)
    fig.suptitle('')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)
    plt.show()


def diversity_compare(df: pd.DataFrame, save_path=None):
    """ Compares relative diversity measures (online stats) for multiple EAs
    across enemies.
    """

    enemies = df['enemies'].unique()
    n_enemies = len(enemies)
    # Get neat versions of the instance names
    instances = df['ea_instance'].unique()

    fig, ax = plt.subplots(n_enemies, 1)

    for i, enemy in enumerate(enemies):
        for j, instance in enumerate(instances):
            colour = COLOURS[j]
            _df = df[(df['enemies'] == enemy) & (
                df['ea_instance'] == instance)]
            _df.plot.line(x='gen', y='diversity_mean', yerr='diversity_std', fmt='.-',
                          ax=ax[i], c=colour, label=f'{instance} diversity', alpha=0.7, markersize=5, capsize=2)
            ax[i].set_title(f'Training enemies: {enemy}')
            ax[i].set_xlabel('')
            handles, labels = ax[i].get_legend_handles_labels()
            ax[i].get_legend().remove()

    fig.legend(handles, labels,
                framealpha=1.0,
                loc='lower center',
                fancybox=False,
                shadow=False,
                ncol=3)

    plt.xlabel('Generation')
    plt.ylabel('Relative population diversity')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)
    plt.show()


def inspect_evolution_stats(df: pd.DataFrame, save_path=None):
    """Produces a plot of all the stats (logbook) returned from .evolve()
    functions (or the versions saved to .csv files).

    Parameters
    ----------
    df : pd.DataFrame
        Logbook statistics (output of evo_utils.compile_stats).
    """

    # Copy so that original is untouched
    _df = df.copy()

    # Make sure index has been reset
    if 'gen' not in _df.columns:
        _df = _df.reset_index()

    fig, ax = plt.subplots(3, 1)
    _df.plot.line(x='gen', y='mean_fitness', yerr='std_fitness', ax=ax[0])
    _df.plot.line(x='gen', y=['min_fitness', 'max_fitness'], ax=ax[1])
    _df.plot.line(x='gen', y='diversity', ax=ax[2])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)
    plt.show()


def stat_test_t(df):

    # Load names of EA instances
    instances = df['ea_instance'].unique()

    baseline = instances[0]

    for ea in instances[1:]:

        stats_results = {'enemies': [], 'statistic': [], 'p_value': []}

        for enemy in df['enemies'].unique():
            gains1 = df[(df['ea_instance'] == baseline) & (
                df['enemies'] == enemy)]['gain'].values
            gains2 = df[(df['ea_instance'] == ea) & (
                df['enemies'] == enemy)]['gain'].values

            # Perform independent t-test for that enemy
            w, p = stats.ttest_ind(gains1, gains2)

            # Save results to dictionary of lists
            stats_results['enemies'].append(enemy)
            stats_results['statistic'].append(w)
            stats_results['p_value'].append(p)
            print(f"p-val: {instances[0]} vs {ea}: {p}")

    # Make dataframe of significance results
    df = pd.DataFrame(stats_results)
    return df


if __name__ == "__main__":

    # Quick inspection of results
    # df = pd.read_csv('experiments/tmp/10-07-13_47_16_logbook.csv')
    # inspect_evolution_stats(df)

    if len(sys.argv) < 3:
        sys.exit("Need python3 visualiser.py <online_results> <offline_results> [optional genomes]")

    online_results = pd.read_csv(sys.argv[1])
    offline_results = pd.read_csv(sys.argv[2])

    # Format the data and calculate statistics
    online_summary = format_online_results(online_results)
    print('\nOnline results', online_summary.groupby(
        ['ea_instance', 'enemies']).mean(), sep='\n', end='\n\n')

    offline_summary = format_offline_results(offline_results)

    # Specialist lineplots
    specialist_lineplots(
        online_summary, save_path='figs/fitness_lineplots.png')
    diversity_compare(online_summary, save_path='figs/diversity.png')

    # Specialist boxplots
    specialist_boxplots(offline_summary, save_path='figs/gain_boxplots.png')

    # Significance tests
    df_sig = stat_test_t(offline_summary)

    # Combine offline with significance
    df_offline = offline_summary[['ea_instance', 'enemies', 'gain']].groupby(
        ['ea_instance', 'enemies']).mean().reset_index()
    df_offline = df_offline.pivot(
        index='enemies', columns='ea_instance', values='gain')

    # df_offline = df_offline.merge(df_sig, on='enemies')
    df_offline = df_offline.round(4)
    print("\nOffline results", df_offline, sep='\n', end='\n\n')
    print(df_offline.to_latex(index=True))

    # Diversity comparisons
    if len(sys.argv) == 4:
        df_div = diversity_comparison(sys.argv[3])
        # Clean up names and ordering in dataframe
        df_div.columns = [tidy_instance_name(x) for x in df_div.columns]
        # df_div = df_div[sorted([c for c in df_div.columns])]
        df_div.reset_index(inplace=True)
        print("\nDiversity results", df_div, sep='\n', end='\n\n\n')
        print(df_div.to_latex(index=False))
