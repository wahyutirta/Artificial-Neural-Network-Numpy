
import numpy as np

# ReLU activation
class Activation_ReLu:

    # Forward pass
    def forward(self, inputs):

        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        # relu will return non negative input as output, other wise will return 0 as output
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # dvalue is derivatve value from next layer
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        # we just copy value from next layer because 
        # derivative of relu if z > 0 = 1, else 0
        # in chain rule we will multiply dvalue by one 
        # we dont have to do the multiply, just copy the dvalue
        # so we just have to make sure to change negative value to 0
        # drelu_dz = dvalue * (1. if z > 0 else 0.)
        self.dinputs[self.inputs <= 0] = 0 