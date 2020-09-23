import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

DPI = 200

# Set sensible defaults
sns.set()
sns.set_style("ticks")
sns.set_context('paper')


def tidy_instance_name(x):
    return x.split('.')[-1].split("'")[0]


def format_online_results(df: pd.DataFrame):
    """ Formats the raw online results by calculating
    statistics (mean, std) over all iterations.
    """
    # Make a copy of raw data
    _df = df.copy()

    # Calculate statistics over all iterations
    _df = _df.groupby(['ea_instance', 'enemy', 'gen']).agg(['mean', 'std'])
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
    _df = _df.groupby(['ea_instance', 'enemy', 'individual'],
                      as_index=False).mean()
    _df.reset_index(inplace=True)

    # Rename instances nicely
    _df['ea_instance'] = _df['ea_instance'].apply(tidy_instance_name)

    return _df


def specialist_lineplots(df: pd.DataFrame):
    """
    Compare your algorithms by enemy, making a line-plot across the generations, with the average/std (for the mean and the maximum) of the fitness.
    Note that you need to calculate the average (over the 10 runs) of the mean and maximum (over the population in each generation).
    Do one plot by enemy, thus, separately.
    """

    enemies = df['enemy'].unique()
    n_enemies = len(enemies)
    # Get neat versions of the instance names
    instances = df['ea_instance'].unique()

    fig, ax = plt.subplots(n_enemies, 1, dpi=DPI)
    colours = ['blue', 'red']

    for i, enemy in enumerate(enemies):
        for j, instance in enumerate(instances):
            colour = colours[j]
            _df = df[(df['enemy'] == enemy) & (df['ea_instance'] == instance)]
            _df.plot.line(x='gen', y='mean_fitness_mean', yerr='mean_fitness_std',
                          ax=ax[i], c=colour, label=f'{instance} mean fitness')
            _df.plot.line(x='gen', y='max_fitness_mean', yerr='max_fitness_std',
                          ax=ax[i], c=colour, linestyle='--', label=f'{instance} max fitness')
            ax[i].set_title(f'Enemy: {enemy}')
            ax[i].set_xlabel('')
            handles, labels = ax[i].get_legend_handles_labels()
            ax[i].get_legend().remove()

    fig.legend(handles, labels, loc='lower right')

    plt.xlabel('Generation')
    plt.ylabel('Fitness score')
    plt.tight_layout()
    plt.show()


def specialist_boxplots(df: pd.DataFrame):
    """
    Compare your algorithms by enemy, testing 5 times your final best solution for each of the 10 independent runs, and present the individual gain in box-plots.
    Note that you need to calculate the means of the 5 times for each solution of the algorithm for the enemy, and these means are the values that will be points in the box-plot. In summary, it is a total of 3 pairs of box-plots (so 6 boxes), being one pair per enemy.
    """

    enemies = df['enemy'].unique()
    n_enemies = len(enemies)

    fig, ax = plt.subplots(1, n_enemies, dpi=DPI)
    for i, enemy in enumerate(enemies):
        _df = df[df['enemy'] == enemy]
        _df.boxplot(column='gain', by='ea_instance', grid=False, ax=ax[i])
        ax[i].set_title(f'Enemy: {enemy}')
        ax[i].set_ylabel('Gain score')
        ax[i].set_xlabel('')
        ax[i].tick_params(axis='x', labelrotation=90)
    fig.suptitle('')
    plt.tight_layout()
    plt.show()


def stat_test_t(df):

    # Load names of EA instances
    instances = df['ea_instance'].unique()
    if len(instances) != 2:
        raise ValueError(
            "Need exactly 2 EA instances to do significance tests")
    ea1, ea2 = instances[0], instances[1]

    stats_results = {'enemy': [], 'statistic': [], 'p_value': []}

    for enemy in df['enemy'].unique():
        gains1 = df[(df['ea_instance'] == ea1) & (
            df['enemy'] == enemy)]['gain'].values
        gains2 = df[(df['ea_instance'] == ea2) & (
            df['enemy'] == enemy)]['gain'].values

        # Perform independent t-test for that enemy
        w, p = stats.ttest_ind(gains1, gains2)

        # Save results to dictionary of lists
        stats_results['enemy'].append(enemy)
        stats_results['statistic'].append(w)
        stats_results['p_value'].append(p)

    # Make dataframe of significance results
    df = pd.DataFrame(stats_results)
    return df


def inspect_evolution_stats(df: pd.DataFrame):
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

    fig, ax = plt.subplots(3, 1, dpi=DPI)
    _df.plot.line(x='gen', y='mean_fitness', yerr='std_fitness', ax=ax[0])
    _df.plot.line(x='gen', y=['min_fitness', 'max_fitness'], ax=ax[1])
    _df.plot.line(x='gen', y='diversity', ax=ax[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    online_results = pd.read_csv(
        'results/09-22-21_51_13_online_results.csv')
    offline_results = pd.read_csv(
        'results/09-23-12_01_38_offline_results.csv')

    # Format the data and calculate statistics
    online_summary = format_online_results(online_results)
    offline_summary = format_offline_results(offline_results)

    # Specialist lineplots
    specialist_lineplots(online_summary)

    # Specialist boxplots
    specialist_boxplots(offline_summary)

    # Significance tests
    df_sig = stat_test_t(offline_summary)

    print(df_sig, end='\n\n\n')
    print(df_sig.to_latex(index=False))

    # Quick inspection of results
    df = pd.read_csv('experiments/tmp/09-22-18_58_02_logbook.csv')
    inspect_evolution_stats(df)
