import numpy as np
import pandas as pd

def linearRegression(X: np.array, Y: np.array, lr: float, lambda_: float):
    """
    Parameters:
    - X: Input feature matrix (NumPy array)
    - Y: Target vector (NumPy array)
    - lr: Learning rate (float)
    - lambda_: L1 regularization coefficient (float)

    Returns:
    - weights: Learned model parameters
    """
    
    X = np.c_[np.ones(X.shape[0]), X] # Add bias term (column of 1s) to X. Now shape becomes (n_samples, n_features + 1)
    weights = np.zeros(X.shape[1]) # Initialize weights (including bias term)
    n_iterations = 1000 # Number of training iterations
    # Gradient Descent
    for _ in range(n_iterations):
        predictions = X.dot(weights)
        error = predictions - Y
        gradient = (1 / X.shape[0]) * X.T.dot(error)
        gradient += lambda_ * np.sign(weights) # Add L1 regularization gradient (subgradient for L1)
        weights -= lr * gradient

    return weights
