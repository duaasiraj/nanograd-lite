from nanograd.engine.tensor import Tensor
import numpy as np
class MSELoss:
    def __call__(self, pred, target):
        x=((pred-target)**2)
        return x.mean()
    
class CrossEntropyLoss:
    def __call__(self, pred, target):
        probs = pred.softmax()
        target=np.array(target)
        batch_size = target.shape[0]
        num_classes = pred.data.shape[1]
        target_new = np.zeros((batch_size, num_classes))
        target_new[np.arange(batch_size), target] = 1
        target_tensor = Tensor(target_new)
        log_probs = probs.log()
        loss_matrix = -(log_probs * target_tensor)
        loss_per_sample = loss_matrix.sum(axis=1)
        return loss_per_sample.mean()