import numpy as np
import math
import matplotlib.pyplot as plt
from engine.utils import topo_sort,_unbroadcast

class Tensor:
    def __init__(self, data,_children=(),_op=''):
        self.data=np.array(data, dtype=float)
        self.grad=np.zeros_like(self.data)
        self._prev=set(_children)#stores how to get back to this node 
        self._op=_op #stores what operation produced this node
        self._backward=lambda:None # does nothing by default, overridden by each op

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += _unbroadcast(out.grad,self.data.shape)
            other.grad += _unbroadcast(out.grad,other.data.shape)
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out=Tensor(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad+=_unbroadcast(out.grad*other.data,self.data.shape)
            other.grad+=_unbroadcast(out.grad*self.data,other.data.shape)
        out._backward=_backward
        return out
    
    def __rmul__(self, other):
        return self*other
    
    def sum(self,axis=None):
        out = Tensor(self.data.sum(axis=axis,keepdims=True), (self,), 'sum')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x=self.data
        out = Tensor(np.exp(self.data), (self, ), 'exp')
        def _backward():
            self.grad+=out.data*out.grad
        out._backward=_backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

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
        self.grad = np.ones_like(self.data)    
        for node in reversed(topo):  
            node._backward()

    def mean(self):
        out = Tensor(np.mean(self.data), (self,), 'mean')
        def _backward():
            self.grad += (np.ones_like(self.data) * out.grad)/self.data.size
        out._backward = _backward
        return out

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), (self,), 'reshape')
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self):
        out=Tensor(np.swapaxes(np.data, -1, -2),(self,),'Transpose')
        def _backward():
            self.grad+=np.swapaxes(out.grad, -1, -2)
        out._backward=_backward
        return out

    def __matmul__(self,other):
        out=Tensor(self.data@other.data,(self,other),"MatMul")
        def _backward():
            grad_self = out.grad @ np.swapaxes(other.data, -1, -2)
            grad_other = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad  += _unbroadcast(grad_self, self.data.shape)
            other.grad += _unbroadcast(grad_other, other.data.shape)
        out._backward=_backward
        return out 
    
    