import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats


# Set sensible defaults
sns.set()
sns.set_style("ticks")
sns.set_context('paper')

# Set name of algorithms
first_ea = "<class 'EA_demo.MyDemoEAInstance'>"
second_ea = "<class 'EA_dummy.DummyEAInstance'>"


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
    instances = [x.split('.')[-1].split("'")[0]
                 for x in df['ea_instance'].unique()]

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


def specialist_boxplots(df_res):
    """
    Compare your algorithms by enemy, testing 5 times your final best solution for each of the 10 independent runs, and present the individual gain in box-plots.
    Note that you need to calculate the means of the 5 times for each solution of the algorithm for the enemy, and these means are the values that will be points in the box-plot. In summary, it is a total of 3 pairs of box-plots (so 6 boxes), being one pair per enemy.
    """

    # Get
    df_avg = df_res.groupby(['ea_instance', 'enemy', 'individual'], as_index=False).mean()
    df_avg.boxplot('gain', by=['enemy', 'ea_instance'])
    degrees = 70
    plt.xticks(rotation=degrees)

    print(df_avg)
    plt.xticks([1, 2, 3, 4, 5, 6], [str(1) + ' ea_1', str(1) + ' ea_2', str(3) + ' ea_1',
                                    str(3) + ' ea_2', str(5) + ' ea_1', str(5) + ' ea_2'])


    # Perform statistical tests
    stat_test_t(df_avg)
    plt.savefig('boxplots.jpg', dpi=300)
    plt.show()


def stat_test_t(df_avg):
    enemy_1 = df_avg.loc[df_avg['enemy'] == 1]
    enemy_3 = df_avg.loc[df_avg['enemy'] == 3]
    enemy_5 = df_avg.loc[df_avg['enemy'] == 5]
    enemies = [enemy_1, enemy_3, enemy_5]
    counter = 1

    for enemy in enemies:
        g1 = enemy.loc[enemy['ea_instance'] == first_ea]
        g2 = enemy.loc[enemy['ea_instance'] == second_ea]

        # Perform t-test on gain
        print("t-test")
        t, p = stats.ttest_ind(g1['gain'], g2['gain'])
        print(str(counter) + " t value is " + str(t))
        print(str(counter) + " p value is " + str(p))

        # Perform Wilcoxon test on gain
        print("Wilcoxon test")
        w, p2 = stats.wilcoxon(x=g1['gain'], y=g2['gain'], zero_method='wilcox', correction=False,
                              alternative='two-sided')
        print(str(counter) + " w value is " + str(w))
        print(str(counter) + " p value is " + str(p2))

        counter += 2


if __name__ == "__main__":
    # online_results = pd.read_csv(
    #     'experiment_run/09-15-20_15_36_all_results.csv')

    # Format the data and calculate statistics
    # df_summary = format_online_results(online_results)

    # Specialist lineplots
    # specialist_lineplots(df_summary)

    # boxplots
    df_results = pd.read_csv('dummy_offline_results.csv')
    specialist_boxplots(df_results)
