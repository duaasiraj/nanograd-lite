import numpy as np
def binary_accuracy(pred, target):
    return np.mean((pred.data >= 0.5).astype(float) == target.data)