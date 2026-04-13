import numpy as np



def activation_functions(x, activation):
    """
    Returns: list
    """
    x = round(float(x),4) 
     
    def sigmoid(x):
        return 1/(1+np.exp(-x)) 
    
    def relu(x):
        return np.maximum(0, x) , round(float(1 if x>0 else 0),4)
    
    def leaky_relu(x):
        alpha = 0.01
        return x if x>0 else alpha*x , 1.0 if x>0 else alpha 
    
    def swish(x):
        return x * sigmoid(x) , sigmoid(x) + (x * sigmoid(x)) * (1-sigmoid(x))
    
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def gelu_derivative(x):
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        inner = sqrt_2_over_pi * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)
    
        term1 = 0.5 * (1 + tanh_inner)
        term2 = 0.5 * x * (1 - tanh_inner**2) * sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)
    
        return term1 + term2    
    
    def tanh(x):
        return np.tanh(x), 1 - np.tanh(x)**2
    
    if activation == "relu":
        return relu(x)
    elif activation == "leaky_relu":
        return leaky_relu(x)
    elif activation == "sigmoid":
        return sigmoid(x), sigmoid(x) * (1-sigmoid(x))
    elif activation == "swish":
        return swish(x)
    elif activation == "gelu":
        return gelu(x), gelu_derivative(x)
    elif activation == "tanh":
        return tanh(x)
    return x