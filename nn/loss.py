from engine.tensor import Tensor
import numpy as np
class MSELoss:
    def __call__(self, pred, target):
        x=((pred-target)**2)
        return x.mean()
    
class CrossEntropyLoss:
    def __call__(self, pred, target):
        probs=pred.softmax()
        correct_prob=Tensor(np.clip(probs.data[np.arange(len(target)), target], 1e-7, 1.0))
        loss=-correct_prob.log()
        return loss.mean()

