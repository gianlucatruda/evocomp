# Off-the-shelf player controller (based on NN) provided with the framework

from evoman.controller import Controller
import numpy as np


def sigmoid_activation(x):
    return 1./(1.+np.exp(-x))

# implements controller structure for player


class player_controller(Controller):
    def __init__(self, _n_hidden):
        # Number of hidden neurons
        self.n_hidden = [_n_hidden]

    def control(self, inputs, genotype_weights):
        # Normalises the input using min-max scaling
        inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

        if self.n_hidden[0] > 0:
            # Preparing the weights and biases from the genotype_weights of layer 1

            # Biases for the n hidden neurons
            bias1 = genotype_weights[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
            # Weights for the connections from the inputs to the hidden nodes
            weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0]
            weights1 = genotype_weights[self.n_hidden[0]:weights1_slice].reshape(
                (len(inputs), self.n_hidden[0]))

            # Outputs activation first layer.
            output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

            # Preparing the weights and biases from the genotype_weights of layer 2
            bias2 = genotype_weights[weights1_slice:weights1_slice + 5].reshape(1, 5)
            weights2 = genotype_weights[weights1_slice + 5:].reshape((self.n_hidden[0], 5))

            # Outputting activated second layer. Each entry in the output is an action
            output = sigmoid_activation(output1.dot(weights2) + bias2)[0]
        else:
            bias = genotype_weights[:5].reshape(1, 5)
            weights = genotype_weights[5:].reshape((len(inputs), 5))

            output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        # takes decisions about sprite actions
        if output[0] > 0.5:
            left = 1
        else:
            left = 0

        if output[1] > 0.5:
            right = 1
        else:
            right = 0

        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0

        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0

        if output[4] > 0.5:
            release = 1
        else:
            release = 0

        return [left, right, jump, shoot, release]
