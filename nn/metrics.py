import numpy as np

def multiclass_precision(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1)
    precision_list = []
    for c in np.unique(y_true):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        precision = tp / (tp + fp + 1e-8)
        precision_list.append(precision)
    return np.mean(precision_list)


def multiclass_recall(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1)
    recall_list = []
    for c in np.unique(y_true):
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        recall = tp / (tp + fn + 1e-8)
        recall_list.append(recall)
    return np.mean(recall_list)

def f1score(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1)
    recall_list = []
    precision_list = []
    for c in np.unique(y_true):
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        precision = tp / (tp + fp + 1e-8)
        precision_list.append(precision)
        recall = tp / (tp + fn + 1e-8)
        recall_list.append(recall)
    r=np.mean(recall_list)
    p=np.mean(precision_list)
    return (2*p*r)/(p+r+1e-8)

