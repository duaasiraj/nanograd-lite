import numpy as np
import math
import matplotlib.pyplot as plt

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
       x=Tensor(self.data+other.data,(self,other),'+')
       return x
    
    def __mul__(self, other):
        x=Tensor(self.data*other.data,(self,other),'*')
        return x
