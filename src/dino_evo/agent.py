"""
Module defining the agent controller using a simple neural network abstraction.
"""

from typing import Callable
from dino_evo.env import DinoAction
import numpy as np


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


class Layer:
    def __init__(self, input_size: int, output_size: int, activation_fn: Callable = relu):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)
        self.activation_fn = activation_fn

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        z = np.dot(inputs, self.weights) + self.biases
        return self.activation_fn(z)


class AgentBrain:
    """
    A simple neural network acting as the controller for the dino.
    """

    def __init__(self, input_size: int, layer_configs: list):
        self.layers = []

        current_input_size = input_size

        for output_size, activation_fn in layer_configs:
            layer = Layer(
                input_size=current_input_size,
                output_size=output_size,
                activation_fn=activation_fn
            )
            self.layers.append(layer)
            current_input_size = output_size

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs a full forward pass through all layers of the network.
        """
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def get_flattened_weights(self) -> np.ndarray:
        """
        Flattens all weights and biases from all layers into a single 1D array.
        """
        flat_params = []
        for layer in self.layers:
            flat_params.append(layer.weights.flatten())
            flat_params.append(layer.biases.flatten())
        return np.concatenate(flat_params)

    def set_flattened_weights(self, flat_weights: np.ndarray):
        """
        Sets the flattened weights as the weights for the brain network layers
        """
        current_pos = 0
        for layer in self.layers:
            # calc the number of elements for the weights matrix
            w_size = layer.weights.size
            # slice and assign the weights
            layer.weights = flat_weights[current_pos: current_pos +
                                         w_size].reshape(layer.weights.shape)
            current_pos += w_size

            b_size = layer.biases.size
            layer.biases = flat_weights[current_pos: current_pos + b_size]
            current_pos += b_size


def create_brain(n_inputs: int) -> AgentBrain:
    """Creates a brain network to control a dino."""
    return AgentBrain(n_inputs, layer_configs=[
        (8, relu),
        (6, relu),
        (len(DinoAction), tanh)
    ])
