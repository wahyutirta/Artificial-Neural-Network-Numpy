import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x*(1 - x)

T_input = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])

T_output = np.array([[0,1,1,0]]).T

np.random.seed(1)

weights = 2*np.random.random((3, 1)) - 1

print('Random start weight')
print(weights)
for i in range(1):
    input_layer = T_input
    temp = np.dot(input_layer, weights)
    print('dot product')
    print(temp)
    
    outputs = sigmoid(temp)
    print('output sigmoid')
    print(outputs)
    
    error = T_output - outputs
    print('error')
    print(error)
    
    error_sig = sigmoid_derivative(outputs)
    print('error_sig')
    print(error_sig)
    adjust = error * error_sig
    print('adjust')
    print(adjust)
    weights += np.dot(input_layer.T, adjust)
    #print('adjust')
    #print(adjust)
    
    print('weight setelah traning')
    print(weights)

print('output setelah traning :')
print(outputs)


