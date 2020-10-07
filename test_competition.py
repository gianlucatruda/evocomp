import sys
import os
import numpy as np
import time

sys.path.insert(0, 'evoman')

from simple_controller import player_controller
from evoman.environment import Environment


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
    print(results)
    pass


def rankII_evaluation():
    pass


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

    RUN_MODE = 'train'  # train or test mode

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
    results = {"defeated": 0, "pl_life": 0, "en_life": 0, "time": 0}
    for en in enemies:
        env.update_parameter('enemies', [en])
        res = np.array(list(map(lambda y: simulation(env, y), [bsol])))
        res = res[0]
        if res[2] == 0:
            results["defeated"] = results.get("defeated") + 1
        results["pl_life"] = results.get("pl_life") + res[1]
        results["en_life"] = results.get("en_life") + res[2]
        results["time"] = results.get("time") + res[3]

    rankI_evaluation(results)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Requires the path to the best genome .txt file ...")
        exit(0)
    main(sys.argv[1])
