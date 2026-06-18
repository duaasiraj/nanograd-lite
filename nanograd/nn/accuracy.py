import numpy as np
def binary_accuracy(pred, target):
    return np.mean((pred.data >= 0.5).astype(float) == target.data)


def  multiclass_Accuracy(pred, target):
    y_pred = np.argmax(pred.data, axis=1)
    accuracy = np.mean(y_pred==target)
    return accuracy