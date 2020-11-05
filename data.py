import numpy as np


# Sine sample dataset
class sine():
    def create_data(samples=1000):
    
        X = np.arange(samples).reshape(-1, 1) / samples
        y = np.sin(2 * np.pi * X).reshape(-1, 1)
    
        return X, y

class spiral():
    def create_data(samples, classes):
        X = np.zeros((samples*classes, 2))
        y = np.zeros(samples*classes, dtype='uint8')
        for class_number in range(classes):
            ix = range(samples*class_number, samples*(class_number+1))
            r = np.linspace(0.0, 1, samples)
            t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
            X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
            y[ix] = class_number
            return X, y

class vertical():
    def create_data(samples, classes):
        X = np.zeros((samples*classes, 2))
        y = np.zeros(samples*classes, dtype='uint8')
        for class_number in range(classes):
            ix = range(samples*class_number, samples*(class_number+1))
            X[ix] = np.c_[np.random.randn(samples)*.1 + (class_number)/3, np.random.randn(samples)*.1 + 0.5]
            y[ix] = class_number
        return X, y
