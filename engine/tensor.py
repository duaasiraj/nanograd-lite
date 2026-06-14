import numpy as np
import math
import matplotlib.pyplot as plt
from engine.utils import topo_sort

class Tensor:
    def __init__(self, data,_children=(),_op=''):
        self.data=data
        self.grad=0
        self._prev=set(_children)#stores how to get back to this node 
        self._op=_op #stores what operation produced this node
        self._backward=lambda:None # does nothing by default, overridden by each op

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out=Tensor(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad+=out.grad*other.data
            other.grad+=out.grad*self.data
        out._backward=_backward
        return out
    
    def __rmul__(self, other):
        return self*other
    
    def exp(self):
        x=self.data
        out = Tensor(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad+=out.data+out.grad
        out._backward=_backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other): 
        return self * other

    def __truediv__(self, other): 
        return self * other**-1

    def __neg__(self): 
        return self * -1

    def __sub__(self, other): 
        return self + (-other)

    def __radd__(self, other): 
        return self + other
    
    def backward(self):
        topo = topo_sort(self)         
        self.grad = 1.0      
        for node in reversed(topo):  
            node._backward()

 