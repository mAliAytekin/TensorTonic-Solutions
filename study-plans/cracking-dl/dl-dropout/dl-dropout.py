import numpy as np

def dropout(X, mask, drop_prob, mode):
    """
    Returns: 2D list with values rounded to 4 decimal places.
    """
    X = np.array(X, dtype=float)
    mask = np.array(mask)

    if mode == "train":
        result = (X * mask) / (1 - drop_prob)

    else :
        result =  X

    return np.round(result,4)

    