# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:55:41 2020

@author: ASUS
"""

import numpy as np
from Activation_Softmax import Activation_Softmax


class Loss:

    # Regularization loss calculation
    def regularization_loss(self, layer):
        # 0 by default
        regularization_loss = 0
        # L1 regularization - weights
        # calculate only when factor greater than 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(
                np.abs(layer.weights)
            )
        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(
                layer.weights * layer.weights
            )
        # L1 regularization - biases
        # calculate only when factor greater than 0
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(
                np.abs(layer.biases)
            )
        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(
                layer.biases * layer.biases
            )
        return regularization_loss

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        # to prevent  log 0, clipp y_pred to the lowest value
        # range of pred expected to be around 0 to 1
        # so adding lowest value e.g, 1e-7 to 0
        # and substract 1 with 1e-7
        # -np.log(1+1e-7) = -9.999999505838704e-08 --> error become negative
        # -np.log(1-1e-7) = 1.0000000494736474e-07 --> positive error
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels --> [0,1,2,3] --> 4 samples
        # e.g, softmax_outputs :
        # [[0.7, 0.1, 0.2], --> sample 1
        # [0.1, 0.5, 0.4], --> sample 2
        # [0.02, 0.9, 0.08]] --> sample 3
        # 0 = dog, 1 = cat
        # class_targets = [0, 1, 1]  # dog, cat, cat
        # sample 1 expected to be dog
        # sample 2 expected to be cat
        # sample 3 expected to be cat
        if len(y_true.shape) == 1:
            # print( y_true.shape, y_pred_clipped.shape)

            correct_confidences = y_pred_clipped[range(samples), y_true]
            # array slicing
            # softmax_outputs is a numpy array and you can index it like plain Python lists
            # softmax_outputs[5:10] or multi-dimensional NumPy arrays
            # softmax_outputs[5:10, 2:3]
            # where comma separates dimensions. The same is a case here but instead of fixed values,
            # or ranges (like 5:10) we're using a variable containing a list both
            # this range(len()) expression and class_targets are lists.
            # and it might look like class_targets[[5,6,7,8,9,10],[2,3]]
            # take range(samples) number of row
            # the value of each row come from index is pointed by y_pred

            # using loop
            # for targ_indx, distribution in zip(class_targets, softmax_outputs):
            # samples = len(softmax)
            # softmax = [[0.7, 0.1, 0.2], distribution 0
            # [0.1, 0.5, 0.4], distribution 1
            # [0.02, 0.9, 0.08]] distribution 2
            # class_targets or y_true = np.array([0, 1, 1])
            # print(distribution[targ_idx])
            # [0.7, 0.1, 0.2] -> [0] =  0.7 and so on
            # array slicing
            # softmax_outputs is a numpy array and you can index it like plain Python lists
            # softmax_outputs[5:10] or multi-dimensional NumPy arrays
            # softmax_outputs[5:10, 2:3]
            # where comma separates dimensions. The same is a case here but instead of fixed values,
            # or ranges (like 5:10) we're using a variable containing a list both
            # this range(len()) expression and class_targets are lists.
            # and it might look like class_targets[[5,6,7,8,9,10],[2,3]]

        # Mask values - only for one-hot encoded labels
        # one hot label e.g.
        # [[1,0,0], dog
        # [0,1,0], cat
        # [0,0,1]] cat
        elif len(y_true.shape) == 2:
            multiply = np.multiply(y_pred_clipped, y_true)
            # e.g
            # [[0.7, 0.1, 0.2], * [[1,0,0],
            # [0.1, 0.5, 0.4], *   [0,1,0],
            # [0.02, 0.9, 0.08]] * [0,0,1]]
            # to this ----
            # [[0.7  0.   0.  ]
            # [0.   0.5  0.  ]
            # [0.   0.   0.08]]
            correct_confidences = np.sum(multiply, axis=1)
            # sum in row wise but keepdims = false so it becomes
            # [0.7  0.5  0.08]

        # Losses
        negative_log_likelihoods = 1 * (-np.log(correct_confidences))
        # we only perform probabilities on true class(true class has probability of 1) because the other has 0 value
        # multiply anything with 0 only give us zero value in return
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            #

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        # y_pred probabilities
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# softmax_outputs = np.array([[0.7, 0.1, 0.2],[0.1, 0.5, 0.4],[0.02, 0.9, 0.08]])
# class_targets = np.array([1, 2, 1])
# class_targets = np.array([[1, 0, 0],[0, 1, 0],[0,0,1]])
# loss_v = Loss_CategoricalCrossentropy()
# loss_v.forward(softmax_outputs, class_targets)
