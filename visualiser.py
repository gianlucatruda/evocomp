import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats


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

    fig, ax = plt.subplots(n_enemies, 1,)

    for i, enemy in enumerate(enemies):
        for instance in instances:
            _df = df[df['enemy'] == enemy]
            _df.plot.line(x='gen', y='mean_fitness_mean', yerr='mean_fitness_std',
                          ax=ax[i], label=f'{instance} mean fitness')
            _df.plot.line(x='gen', y='max_fitness_mean', yerr='max_fitness_std',
                          ax=ax[i], label=f'{instance} max fitness')
            ax[i].set_title(f'Enemy: {enemy}')
            ax[i].set_xlabel('')

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

    fig, ax = plt.subplots(1, n_enemies)
    for i, enemy in enumerate(enemies):
        _df = df[df['enemy'] == enemy]
        _df.boxplot(column='gain', by='ea_instance', grid=False, ax=ax[i])
        ax[i].set_title(f'Enemy: {enemy}')
        ax[i].set_xlabel('')
        ax[i].tick_params(axis='x', labelrotation=90)

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

        # Perform wilcoxon for that enemy
        w, p = stats.wilcoxon(x=gains1, y=gains2, zero_method='wilcox',
                              correction=False, alternative='two-sided')

        # Save results to dictionary of lists
        stats_results['enemy'].append(enemy)
        stats_results['statistic'].append(w)
        stats_results['p_value'].append(p)

    # Make dataframe of significance results
    df = pd.DataFrame(stats_results)
    return df


if __name__ == "__main__":
    online_results = pd.read_csv(
        'results/09-16-19_53_05_online_results.csv')
    offline_results = pd.read_csv('dummy_offline_results.csv')

    # Format the data and calculate statistics
    online_summary = format_online_results(online_results)
    offline_summary = format_offline_results(offline_results)

    # Specialist lineplots
    specialist_lineplots(online_summary)

    # Specialist boxplots
    specialist_boxplots(offline_summary)

    # Significance tests
    df_sig = stat_test_t(offline_summary)

    print(df_sig)
