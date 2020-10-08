import sys
import os
import numpy as np
import time
from pathlib import Path

import json
from datetime import datetime
import pandas as pd

sys.path.insert(0, 'evoman')

from simple_controller import player_controller
from evoman.environment import Environment

from evo_utils import genome_to_txt


def best_indiv_to_txt(path):
    """
        Reads the best individuals from the json file and
        converts them to the txt format
    """
    # Read in the dictionary of best performers
    read_path = path
    with open(read_path, 'r') as file:
        best_performers = json.load(file)

    for best in best_performers.keys():
        print(best)
        individual = best_performers[best]
        # print(individual)
        now = datetime.now().strftime("%m-%d-%H_%M_%S")  # Timestamp
        genome_to_txt(individual, f"results/best_{now}.txt")
        time.sleep(1)


def simulation(env, pcont):
    """
    Retrieved from specialist_demo

    Run the simulation
    env : environment object
    pcont: player controller object
    """
    fitness, player_life, enemy_life, sim_time = env.play(pcont=pcont)
    return fitness, player_life, enemy_life, sim_time


def rankI_evaluation(results):
    print("\n@@@@@@@@@ RANK I @@@@@@@@@@@@\n")
    print("The number of defeated enenies is {}/8.".format(sum(results.get("defeated"))))
    print("In case of Ties:")
    print("The accumulated evoman's life is {:.3f}/800.".format(sum(results.get("pl_life"))))
    print("The total matches time is {}/10000.".format(sum(results.get("time"))))
    print("Additional info:")
    print("The accumulated enemies life is {:.3f}/800.".format(sum(results.get("en_life"))))
    pass


def rankII_evaluation(gains):
    print("\n@@@@@@@@@ RANK II @@@@@@@@@@@@\n")
    print("The overall gain across the 8 enemies is:\n")
    for g in gains:
        print("{}".format(g))


def main(path):
    """
        Retrieved from the specialist_demo
        It creates the environment as the competition requires.

        By last it computes the evaluation for Rank I and Rank II as
        specified in the standard_assignment_taskII
    """
    # Set directory for saving logs and experiment states
    EXPERIMENT_DIRECTORY = 'experiments/refactored_specialist_demo'
    if not os.path.exists(EXPERIMENT_DIRECTORY):
        os.makedirs(EXPERIMENT_DIRECTORY)

    N_HIDDEN_NEURONS = 10  # how many neurons in the hidden layer of the NN
    # Initialise the controller (neural network) for our AI player
    nn_controller = player_controller(N_HIDDEN_NEURONS)

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=EXPERIMENT_DIRECTORY,
                      enemies=[2],
                      playermode="ai",
                      player_controller=nn_controller,
                      enemymode="static",
                      level=2,
                      speed="fastest")

    env.state_to_log()  # checks environment state and logs

    # loads file with the best solution for testing
    bsol = np.loadtxt(path)
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'fastest')

    enemies = [i for i in range(1, 9)]
    results = {"enemy": [], "defeated": [], "pl_life": [], "en_life": [], "time": [], "gain": []}
    for en in enemies:
        env.update_parameter('enemies', [en])
        res = np.array(list(map(lambda y: simulation(env, y), [bsol])))
        res = res[0]
        results['enemy'].append(en)
        if res[2] == 0:
            results["defeated"].append(1)
        else:
            results["defeated"].append(0)
        results["pl_life"].append(res[1])
        results["en_life"].append(res[2])
        results["time"].append(res[3])

        results["gain"].append(res[1] - res[2])

    rankI_evaluation(results)
    rankII_evaluation(results["gain"])

    # Turn results into a dataframe
    df_results = pd.DataFrame(results)

    # Save the dataframe
    now = datetime.now().strftime("%m-%d-%H_%M_%S")  # Timestamp
    f_name = f"results/{now}_competition_results.csv"
    df_results.to_csv(f_name, index=False)
    print(f"\nResults saved to {f_name}")


if __name__ == "__main__":
    # Converting best individuals from json to txt
    # best_indiv_to_txt("experiments/tmp/10-07-23_53_48_best_individuals.json")

    if len(sys.argv) < 2:
        print("Requires the path to the best genome .txt file ...")
        exit(0)

    # Run single file if passed, else run all files in directory given
    path = sys.argv[1]
    if path.endswith(".txt"):
        main(path)
    else:
        for file in Path(path).glob("*.txt"):
            print(f"Testing: {file}")
            main(file)