import numpy as np
from engine.tensor import Tensor
from nn.modules import Module
from nn.sequential import Sequential
from nn.layers import Linear
from nn.layers import ReLU
from nn.loss import MSELoss

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