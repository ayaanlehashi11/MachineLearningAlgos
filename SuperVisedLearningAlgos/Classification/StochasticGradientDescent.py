import random , itertools , statistics
import numpy as np

def gradient_descent(gradient , x , y , start , learn_rate = 0.1 , n_iter = 50,
                     tolerance = 1e-06 , dtype = "float64"):
    if not callable(gradient):
        raise TypeError("Gradient must be callable")
    dtype = np.dtype(dtype)
    x , y = np.array(x , dtype = dtype) , np.array(y , dtype = dtype)
    if x.shape[0] != y.shape[0]:
        raise ValueError("'x' and 'y' lengths do not match")
    vector = np.array(start , dtype = dtype)
    learning_rate = np.array(learn_rate , dtype = dtype)
    if np.any(learning_rate <= 0):
        raise ValueError("'learn rate must be greater than zero")
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter a.k.a number of iteration must not equal to zero'")
    tolerance = np.array(tolerance , dtype = dtype)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than zero")
    #Performing the gradient descent algorithm
    for _ in range(n_iter):
        diff = -learn_rate * np.array(gradient(x , y , vector) , dtype)

        #Checking if the absolute difference is small enough
        if np.all(np.abs(diff) <= tolerance):
            break
        #Updating the value of the variables
        vector += diff
    return vector if vector.shape else vector.item()
