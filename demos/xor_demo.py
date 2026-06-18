import numpy as np
from engine.tensor import Tensor
from nn.modules import Module
from nn.sequential import Sequential
from nn.layers import Linear
from nn.layers import ReLU
from nn.loss import MSELoss
from nn.optim import SGD
from nn.accuracy import binary_accuracy

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

    
