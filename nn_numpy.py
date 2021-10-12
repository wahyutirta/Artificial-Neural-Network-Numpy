import numpy as np

from Layer_Dense import Layer_Dense
from Activation_ReLu import Activation_ReLu
from Activation_Softmax import Activation_Softmax
from loss import *
from Optimizer import *
from data import *

import matplotlib.pyplot as plt



# Create dataset
X, y = vertical.create_data(samples=10, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg') 
plt.show()

# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 10)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLu()

dense2 = Layer_Dense(10, 10)
activation2 = Activation_ReLu()


# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense3 = Layer_Dense(10, 3)
# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer, choose one
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)
#optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)
#optimizer = Optimizer_SGD(decay=8e-8, momentum=0.9)
#optimizer = Optimizer_Adagrad(decay=1e-4)



# Train in loop
for epoch in range(100):

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)
    
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense3.forward(activation2.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense3.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    # np.argmax return array of index refering to position of maximun value along axis 1
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
        # np.argmax return indexs of max value each row (axis 1)
        # np.argmax return array of index refering to position of maximun value along axis 1
    accuracy = np.mean(predictions==y)
    
    # if not -> to inverse boolan output
    # true -> false
    # false -> true
    # mod 100 = 0 -> false, if not -> true
    if not epoch % 10:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    # using derivative chain rule from error -> Dloss_function output ->
    # -> Dcategorical_function output an so on
    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases 
    # initialize preupdate params
    optimizer.pre_update_params()
    
    #the order can be changed
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.update_params(dense1)
    
    optimizer.post_update_params()
