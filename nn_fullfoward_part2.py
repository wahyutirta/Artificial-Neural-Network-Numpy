import numpy as np 
import matplotlib.pyplot as plt

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X, y = spiral_data(points=1000, classes=3)
#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
#plt.show()
#plot sebaran data


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        #[[4.8, 1.21, 2.385],         
        #[8.9, -1.81, 0.2],
        #[1.41, 1.051, 0.026]]
        #np.sum(axis=1,keepdims=True)
        #[[8.395]
         #[7.29 ]
         #[2.487]]

        self.output = probabilities

# Loss_CategoricalCrossentropy inherist the loss class
#
#loss_function = Loss_CategoricalCrossentropy()
#loss = loss_function.calculate(softmax_outputs, class_targets)
#print(loss)
class Loss:

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


layer1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(3,3)
activation2 = Activation_ReLU()

activation3 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

layer1.forward(X)
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output[:5])

layer2.forward(activation1.output)
#print(layer1.output)
activation2.forward(layer2.output)
print(activation2.output[:5])

activation3.forward(layer2.output)
print(activation3.output[:5])

loss = loss_function.calculate(activation3.output, y)

# Print loss value
print('loss:', loss)
