import numpy as np
from nanograd.engine.tensor import Tensor
from nanograd.nn.modules import Module
from nanograd.nn.sequential import Sequential
from nanograd.nn.layers import Linear
from nanograd.nn.layers import ReLU
from nanograd.nn.loss import MSELoss
from nanograd.nn.optim import SGD
from nanograd.nn.accuracy import binary_accuracy

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
np.random.seed(42)
inp=Tensor(X)
target=Tensor(y)

model=Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
loss = MSELoss()
sgd=SGD(model.parameters(),0.1)
for i in range(1000):
    out=model.forward(inp)
    l=loss(out,target)
    if(i%100==0):
        print(f'Iteration {i}: Loss {l.data:.4f}')
        acc=binary_accuracy(out,target)
        print(f'Accuracy={acc}')
        print()
    sgd.zero_grad()
    l.backward()
    sgd.step()

    
