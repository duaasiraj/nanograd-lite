from engine.tensor import Tensor
class MSELoss:
    def __call__(self, pred, target):
        x=((pred-target)**2)
        return x.mean()