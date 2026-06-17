from engine.tensor import Tensor
from nn.modules import Module
import numpy as np
import random

class Linear(Module):
    def __init__(self, in_features, out_features):
        inf=in_features
        outf=out_features
        lim=np.sqrt(6/(inf+outf))
        self.w=Tensor(np.random.uniform(-lim,lim,(inf,outf)))
        self.b=Tensor(np.zeros(out_features))
        
    
    def parameters(self):
        return [self.w,self.b]
    
    def forward(self, x):
        return  x @ self.w + self.b

class ReLU(Module):
    def parameters(self):
        return []
    
    def forward(self, x):
        return x.relu()