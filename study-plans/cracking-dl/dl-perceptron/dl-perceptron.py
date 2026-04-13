import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """

    X = np.array(X)  
    y = np.array(y)  
    
    n , d = X.shape
    weights = np.zeros(d)
    bias = 0

    for epoch in range(epochs):
        
        for idx, x_i in enumerate(X):
            linear_output = np.dot(x_i,weights) + bias
            y_pred = 1 if linear_output >=0 else 0

            error = lr * (y[idx]-y_pred)
            weights += error * x_i
            bias += error
            
    return weights, bias