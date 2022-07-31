import numpy as np

def rmse(x1, x2):
    """
    Assumes x1 and x2 are (n_dim x n_time_steps)
    """
    rmse = np.sqrt(np.mean((x1-x2)**2, axis=1))
    return rmse

