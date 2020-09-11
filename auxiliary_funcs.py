import sys
sys.path.insert(0, 'evoman')
import numpy as np
from math import exp, sqrt


# Using A.E. Eiben's paper as reference for the concrete implementation - http://www.few.vu.nl/~gks290/papers/SigmaAdaptationComparison.pdf
def selfAdaptiveMutation(individual: list, step: int) -> list:
    """Custom function based on "Multi-step Self-Adaptation" from the paper above
    Parameters
    ----------
    individual : list
        The genotype of the individual
    step: int
        The current generation number
    Returns
    -------
    list
        The mutated genotype of the individual
    """

    print("The current generation is {}".format(step))
    idxSigma1 = int(len(individual)/2)
    # REMARK - didn't saw a better way to initialize the sigma's to 0.8 with the underlying structure in the deap_specialist file
    if step == 1:
        for i in range(idxSigma1, len(individual)):
            individual[i] = 0.8

    for i in range(idxSigma1, len(individual)):
        # Updating first the sigma
        # print(i)
        individual[i] = individual[i] * exp(np.random.normal(0, 1/(sqrt(2*step))) + np.random.normal(0, 1/(sqrt(2*sqrt(step)))))

        # Updating secondly the x
        # print(i-idxSigma1, individual[i])
        individual[i-idxSigma1] = individual[i-idxSigma1] + np.random.normal(0, individual[i])

    return individual,
