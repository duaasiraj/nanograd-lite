from nanograd.engine.tensor import Tensor
import numpy as np
class SGD:
    def __init__(self, parameters, lr):
        self.p=parameters
        self.learningrate=lr
    
    def step(self):
        for param in self.p:
            param.data-=self.learningrate*param.grad
    
    def zero_grad(self):
        for param in self.p:
            param.grad=np.zeros_like(param.data)