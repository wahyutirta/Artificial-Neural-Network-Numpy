# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 00:54:54 2020

@author: ASUS
"""


import numpy as np

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# sum weights of given input
# and multiply by the passed in gradient for this neuron
# change this index to array operation into dot product by seeing sum and multiplycation pattern
# dx0 = sum(weights[0]*dvalues[0])
# dx1 = sum(weights[1]*dvalues[0])
# dx2 = sum(weights[2]*dvalues[0])
# dx3 = sum(weights[3]*dvalues[0])

# dinputs = np.array([dx0, dx1, dx2, dx3])
# print(dinputs)

# sum weights of given input
# and multiply by the passed in gradient for this neuron
dinputs = np.dot(dvalues[0], weights.T)

print(dinputs)
