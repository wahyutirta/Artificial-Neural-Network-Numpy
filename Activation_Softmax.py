import numpy as np

# softmax activation function basiccaly turn this
#  
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        
        # Get unnormalized probabilities
        # np.exp return e power given input
        
        # unnormalized value
        # inputs [4.8, 1.21, 2.385]
        # exponentiated valued 
        # [121.51041751893969, 3.3534846525504487, 10.85906266492961]
        
        # for the biggest number turn zero.. anything power 0 is 1
        # e^0 (e power 0) = 1
        # normalized value to avoid big number
        # substract input with its biggest value
        # [ 0. -3.59  -2.415]
        # [1. 0.02759833 0.08936734]
        
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # what is keepdims = 1, axis = 1
        # [[4.8, 1.21, 2.385] --first sample,         
        # [8.9, -1.81, 0.2] --second sample,
        # [1.41, 1.051, 0.026]] --third sample
        #np.sum(axis=1,keepdims=True) axis = 1 means sums row direction --->>
        # [[8.395] sum first sample
        # [7.29 ] sum second sample
        # [2.487]] sum third sample
        
        # Normalize them for each sample
        # turn this :  [121.51041751893969, 3.3534846525504487, 10.85906266492961]
        # to this : [0.89528266 0.02470831 0.08000903] --> sum = 1
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

