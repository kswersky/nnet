import numpy as np
from logsumexp import logsumexp

def softmax(x):
    #Numerically stable softmax
    return np.exp(x - logsumexp(x,1)[:,None])

def sigmoid(x):
    #Numerically stable sigmoid
    return np.exp(log_sigmoid(x))

def log_sigmoid(x):
    m = np.maximum(-x,0)
    return -(np.log(np.exp(-m) + np.exp(-x-m)) + m)