# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 18:42:39 2020

@author: ASUS
"""


import numpy as np 

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

class softmax():
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        self.input = x
        print("softmax in X\n", x)
        print("softmax in X .T\n", x.T)
        xT = x.T
        output = np.zeros(x.shape)
        d, n = output.shape
        for i in range(n):
            b = xT[i].max()
            exp = np.exp(xT[i] - b)
            sum = np.sum(exp)
            for j in range(d):
                output[j][i] = exp[j]/sum
        self.output = output
        print("softmax out\n", self.output.shape)
        
        
        return output

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward():
        pass


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    def backward():
        pass

class CrossEntropy():
    def __init__(self):
        self.output = None
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        d, n = x.shape
        print("CE input\n", x)
        print("CE target\n", y)
        print("d = " + str(d) + "n = " + str(n))
        output = np.zeros((1, n))
        for i in range(n):
            temp = 0
            for j in range(d):
                temp = temp + y[j][i] * np.log(x[j][i])
            output[0][i] = -temp
        # print('forward', output.shape)
        self.output = output
        return self.output

#X, y = spiral_data(2, 2) # n set fitur, 1 set ada 2 fitur, n kelas
Y = [[0, 1],[0, 1],[1, 0]]
X = [[1, 2],[3, 4],[5, 6]]
X = np.array(X)
Y = np.array(Y)
layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(5,2)
activation2 = Activation_ReLU()

sf = softmax()
ce = CrossEntropy()

print("\ninput",X)
print("\nlayer1",layer1.weights)
print("\nlayer2",layer2.weights)

for item, y in zip(X,Y) :
    
    layer1.forward(X)
    #print(layer1.output)
    activation1.forward(layer1.output)
    print("\n",activation1.output)
    
    layer2.forward(activation1.output)
    #print(layer1.output)
    activation2.forward(layer2.output)
    print("layer 2 out\n",activation2.output)
    
    sf.forward(activation2.output.T)
    print("softmax out\n", sf.output)
    ce.forward(sf.output, Y.T)
    print(ce.output)
    break