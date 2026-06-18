import numpy as np
from nanograd.engine.tensor import Tensor
from nanograd.nn.modules import Module
from nanograd.nn.sequential import Sequential
from nanograd.nn.layers import Linear
from nanograd.nn.layers import ReLU
from nanograd.nn.loss import MSELoss, CrossEntropyLoss

model = Sequential(Linear(2, 4),ReLU(),Linear(4, 1))

x = Tensor(np.random.randn(3, 2))
out = model.forward(x)
print("Output shape:", out.data.shape)
print("Number of parameters:", len(model.parameters()))
loss = MSELoss()
target = Tensor(np.zeros((3, 1)))
l = loss(model(x), target)
l.backward()
print(model.parameters()[0].grad)

#cross entropy sanity check 
pred = [[2.0, 1.0, 0.1],   
        [0.5, 2.0, 0.3]]
#manual calculation:
x = np.array([2.0, 1.0, 0.1])
e = np.exp(x)
print(e / e.sum())
a=-np.log(0.65900114)
print(a)


b = np.array([0.5, 2.0, 0.3])
c = np.exp(b)
print(c / c.sum())
d=-np.log(0.71133182)
print(d)

print((0.4170300145873477 + 0.3406162632509799) / 2 )
print()
#Tensor calculation 
target = [0, 1]
p=Tensor(pred)
loss=CrossEntropyLoss()
print(loss(pred=p,target=target))