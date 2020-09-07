# Refactored version of optimization_specialist_demo

# imports framework
import sys
sys.path.insert(0, 'evoman')

from simple_controller import player_controller
from evoman.environment import Environment

# imports other libs
import time
import numpy as np
from math import fabs, sqrt
import glob
import os


def simulation(env, pcont):
    """Run the simulation
    env : environment object
    pcont: player controller object
    """
    fitness, player_life, enemy_life, sim_time = env.play(pcont=pcont)
    return fitness


def norm(x, pfit_pop):
    """Normalises ??? TODO
    """

    if (max(pfit_pop) - min(pfit_pop)) > 0:
        x_norm = (x - min(pfit_pop))/(max(pfit_pop) - min(pfit_pop))
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


def evaluate(x):
    """Evaluate a solution (individual) over all y? TODO
    """
    return np.array(list(map(lambda y: simulation(env, y), x)))


def tournament(pop):
    """Tournament selection ??? TODO
    """
    c1 = np.random.randint(0, pop.shape[0], 1)
    c2 = np.random.randint(0, pop.shape[0], 1)

    if fit_pop[c1] > fit_pop[c2]:
        return pop[c1][0]
    else:
        return pop[c2][0]


def limits(x, dom_u, dom_l):
    """What does this limit? Number of offspring? Population size? TODO
    """
    if x > dom_u:
        return dom_u
    elif x < dom_l:
        return dom_l
    else:
        return x


def crossover(pop, dom_u, dom_l, mutation):
    """
    Performs crossover on population members
    """

    total_offspring = np.zeros((0, n_vars))

    for p in range(0, pop.shape[0], 2):
        p1 = tournament(pop)
        p2 = tournament(pop)

        n_offspring = np.random.randint(1, 3+1, 1)[0]
        offspring = np.zeros((n_offspring, n_vars))

        for f in range(0, n_offspring):

            cross_prop = np.random.uniform(0, 1)
            offspring[f] = p1*cross_prop+p2*(1-cross_prop)

            # mutation
            for i in range(0, len(offspring[f])):
                if np.random.uniform(0, 1) <= mutation:
                    offspring[f][i] = offspring[f][i]+np.random.normal(0, 1)

            offspring[f] = np.array(
                list(map(lambda y: limits(y, dom_u, dom_l), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring


def doomsday(pop, fit_pop, npop, n_vars, dom_u, dom_l):
    """ kills the worst genomes, and replace with new best/random solutions
    """

    worst = int(npop/4)  # a quarter of the population
    order = np.argsort(fit_pop)
    orderasc = order[0:worst]

    for o in orderasc:
        for j in range(0, n_vars):
            pro = np.random.uniform(0, 1)
            if np.random.uniform(0, 1) <= pro:
                # random dna, uniform dist.
                pop[o][j] = np.random.uniform(dom_l, dom_u)
            else:
                pop[o][j] = pop[order[-1:]][0][j]  # dna from best

        fit_pop[o] = evaluate([pop[o]])

    return pop, fit_pop


if __name__ == "__main__":
    # Set directory for saving logs and experiment states
    EXPERIMENT_DIRECTORY = 'experiments/refactored_specialist_demo'
    if not os.path.exists(EXPERIMENT_DIRECTORY):
        os.makedirs(EXPERIMENT_DIRECTORY)

    N_HIDDEN_NEURONS = 10  # how many neurons in the hidden layer of the NN
    # Initialise the controller (neural network) for our AI player
    nn_controller = player_controller(N_HIDDEN_NEURONS)

    # Evolutionary Algorithm parameters
    DOM_U = 1
    DOM_L = -1
    NPOP = 100
    GENS = 30
    MUTATION = 0.2
    LAST_BEST = 0

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
    ini = time.time()  # sets time marker
    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1) * \
        N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS+1)*5

    if RUN_MODE == 'test':
        # loads file with the best solution for testing
        bsol = np.loadtxt(EXPERIMENT_DIRECTORY+'/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed', 'normal')
        evaluate([bsol])
        sys.exit(0)

    if not os.path.exists(EXPERIMENT_DIRECTORY+'/evoman_solstate'):
        # initializes population loading old solutions or generating new ones
        print('\nNEW EVOLUTION\n')
        pop = np.random.uniform(DOM_L, DOM_U, (NPOP, n_vars))
        fit_pop = evaluate(pop)
        best = np.argmax(fit_pop)
        mean = np.mean(fit_pop)
        std = np.std(fit_pop)
        ini_g = 0
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
    else:
        print('\nCONTINUING EVOLUTION\n')
        env.load_state()
        pop = env.solutions[0]
        fit_pop = env.solutions[1]
        best = np.argmax(fit_pop)
        mean = np.mean(fit_pop)
        std = np.std(fit_pop)
        # finds last generation number
        file_aux = open(EXPERIMENT_DIRECTORY+'/gen.txt', 'r')
        ini_g = int(file_aux.readline())
        file_aux.close()

    # saves results for first pop
    file_aux = open(EXPERIMENT_DIRECTORY+'/results.txt', 'a')
    file_aux.write('\n\ngen best mean std')
    print('\n GENERATION '+str(ini_g)+' ' +
          str(round(fit_pop[best], 6))+' '+str(round(mean, 6))+' '+str(round(std, 6)))
    file_aux.write('\n'+str(ini_g)+' ' +
                   str(round(fit_pop[best], 6))+' '+str(round(mean, 6))+' '+str(round(std, 6)))
    file_aux.close()

    # Run the evolutionary loop through generations
    last_sol = fit_pop[best]
    notimproved = 0

    for i in range(ini_g+1, GENS):

        offspring = crossover(pop, DOM_U, DOM_L, MUTATION)  # crossover
        fit_offspring = evaluate(offspring)   # evaluation
        pop = np.vstack((pop, offspring))
        fit_pop = np.append(fit_pop, fit_offspring)

        best = np.argmax(fit_pop)  # best solution in generation
        # repeats best eval, for stability issues
        fit_pop[best] = float(evaluate(np.array([pop[best]]))[0])
        best_sol = fit_pop[best]

        # selection
        fit_pop_cp = fit_pop
        # avoiding negative probabilities, as fitness is ranges from negative numbers
        fit_pop_norm = np.array(
            list(map(lambda y: norm(y, fit_pop_cp), fit_pop)))
        probs = (fit_pop_norm)/(fit_pop_norm).sum()
        chosen = np.random.choice(pop.shape[0], NPOP, p=probs, replace=False)
        chosen = np.append(chosen[1:], best)
        pop = pop[chosen]
        fit_pop = fit_pop[chosen]

        # searching new areas
        if best_sol <= last_sol:
            notimproved += 1
        else:
            last_sol = best_sol
            notimproved = 0
        if notimproved >= 15:
            file_aux = open(EXPERIMENT_DIRECTORY+'/results.txt', 'a')
            file_aux.write('\ndoomsday')
            file_aux.close()
            pop, fit_pop = doomsday(pop, fit_pop, NPOP, n_vars, DOM_U, DOM_L)
            notimproved = 0

        best = np.argmax(fit_pop)
        std = np.std(fit_pop)
        mean = np.mean(fit_pop)

        # saves results
        file_aux = open(EXPERIMENT_DIRECTORY+'/results.txt', 'a')
        print('\n GENERATION '+str(i)+' ' +
              str(round(fit_pop[best], 6))+' '+str(round(mean, 6))+' '+str(round(std, 6)))
        file_aux.write(
            '\n'+str(i)+' '+str(round(fit_pop[best], 6))+' '+str(round(mean, 6))+' '+str(round(std, 6)))
        file_aux.close()

        # saves generation number
        file_aux = open(EXPERIMENT_DIRECTORY+'/gen.txt', 'w')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(EXPERIMENT_DIRECTORY+'/best.txt', pop[best])

        # saves simulation state
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        env.save_state()

    fim = time.time()  # prints total execution time for experiment
    print('\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')

    # saves control (simulation has ended) file for bash loop file
    file = open(EXPERIMENT_DIRECTORY+'/neuroended', 'w')
    file.close()

    env.state_to_log()  # checks environment state
