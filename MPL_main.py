import numpy as np

from Layer_Dense import Layer_Dense
from Activation_ReLu import Activation_ReLu
from Activation_Softmax import Activation_Softmax
from loss import *
from Optimizer import *
from data import *

import matplotlib.pyplot as plt

class mlp:
    def __init__(self, optimizer = "adam",learning_rate = 0.005, decay = 5e-7, epochs = 100):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epochs = epochs
        dense1 = Layer_Dense(2, 10)
        relu1 = Activation_ReLu()
        
        dense2 = Layer_Dense(10, 10)
        relu2 = Activation_ReLu()
        
        dense3 = Layer_Dense(10, 3)
        loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
        self.layers = [dense1, relu1 ,dense2, relu2, dense3, loss_activation]
        
        if optimizer == "adam":
            self.optimizer = Optimizer_Adam(learning_rate=self.learning_rate, decay=self.decay)
        elif optimizer == "rmsprop":
            self.optimizer = Optimizer_RMSprop(learning_rate=self.learning_rate, decay=self.decay, rho=0.999)
        elif optimizer == "sgd":
            self.optimizer = Optimizer_SGD(learning_rate=self.learning_rate, decay=self.decay, momentum=0.9)
        elif optimizer == "adagrad":
            self.optimizer = Optimizer_Adagrad(learning_rate=self.learning_rate, decay=self.decay)
            
    
    def feedforward(self, x, y):
        inp = x
        for layer in self.layers:
            if isinstance(layer, Activation_Softmax_Loss_CategoricalCrossentropy):
                loss = layer.forward(inp, y)
                output = layer.output
                pass
            else:
                layer.forward(inp)
                inp = layer.output
                pass
                
        return output, loss
    
    def backward(self, y):
        delta = y
        gradient = None
        for layer in self.layers[::-1]:
            if isinstance(layer, Activation_Softmax_Loss_CategoricalCrossentropy):
                
                layer.backward(layer.output, y)
                gradient = layer.dinputs
                pass
            else:
                layer.backward(gradient)
                gradient = layer.dinputs
                pass
            
    
    def updateParams(self):
        self.optimizer.pre_update_params()
        
        for layer in self.layers:
            if isinstance(layer, Layer_Dense):
                
                self.optimizer.update_params(layer)
        
        self.optimizer.post_update_params()
        
    def train(self, x, y):
        for ep in range(self.epochs):
            output, loss = self.feedforward(x, y)
            self.history(output, y, loss, ep)
            
            self.backward(y)
            self.updateParams()
            
            
    def predict(self, x, y):
        output, loss = self.feedforward(x, y)
        predictions = np.argmax(output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
            # np.argmax return indexs of max value each row (axis 1)
            # np.argmax return array of index refering to position of maximun value along axis 1
        accuracy = np.mean(predictions==y)
        print(f'prediction scores -- ' + f'acc: {accuracy:.3f}' + f':: loss: {loss:.3f}')
    
    def history(self, output, y, loss, epoch):
        predictions = np.argmax(output, axis=1)
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
              f'lr: {self.optimizer.current_learning_rate}')

# Create dataset
X, y = vertical.create_data(samples=500, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg') 
plt.show()

optimizer = "adam"
learning_rate = 0.005
epochs = 150

mlpModel = mlp(optimizer = optimizer, learning_rate = learning_rate, epochs=epochs)
mlpModel.train(X, y)

x_test, y_test = vertical.create_data(samples=100, classes=3)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='brg') 
plt.show()
mlpModel.predict(x_test, y_test)


