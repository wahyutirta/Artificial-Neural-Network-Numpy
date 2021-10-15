# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np


class Layer_Dense:

    # Layer initialization
    def __init__(
        self,
        n_inputs,
        n_neurons,
        weight_regularizer_l1=0,
        weight_regularizer_l2=0,
        bias_regularizer_l1=0,
        bias_regularizer_l2=0,
    ):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

        # dot product matrix operation
        # inputs = [p1, p2, 3]
        # weights = [w1, w2, w3]
        # [w1, w2, w3]
        # [w1, w2, w3]

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        # Calculating the gradients with respect to weights
        # in this case, we’re going to be using gradients to update the weights,
        # so we need to match the shape of weights, not inputs.
        # Since the derivative with respect to the weights equals inputs,
        # weights are transposed,
        # so we need to transpose inputs to receive the derivative of the neuron with respect to weights.
        # drelu_dw0 = drelu_dxw0 * dmul_dw0
        self.dweights = np.dot(self.inputs.T, dvalues)
        # to get derivative of weight each neuron -
        # we need to multiply input with dvalue(derivative of next layer or activation layer)
        # given input, weight, activation output
        # input  [[-0.,  0.],   weight [[0., 0.], to activation [[0., 0.,]
        #       [-0., -0.],           [0., 0.]]         output [0., 0.,]
        #       [ 0.,  0.]]                                    [0., 0.,]]
        # to get derivative
        # input  [[-0.,  0.],  to activation [[0., 0.],
        #       [-0., -0.],                 [0., 0.],
        #       [ 0.,  0.]]          output [0., 0.,]
        # so we need to tranpose the inputs matrix
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
            # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        # this is the order
        # drelu_dx0 = drelu_dz * dsum_dxw0 * w[0] --> dmul_dx0 = w[0]