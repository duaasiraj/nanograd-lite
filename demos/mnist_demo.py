from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
from engine.tensor import Tensor
from nn.modules import Module
from nn.sequential import Sequential
from nn.layers import Linear
from nn.layers import ReLU
from nn.loss import CrossEntropyLoss
from nn.optim import SGD
from nn.accuracy import multiclass_Accuracy

X,y=fetch_openml("mnist_784",version=1,return_X_y=True,as_frame=False)
print(X.shape)
print(y.shape)
print(X[0].shape)
X = X.astype("float32") / 255.0
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)

model=Sequential(Linear(784,128), ReLU(),Linear(128,64), ReLU(),Linear(64,10))
Loss=CrossEntropyLoss()
sgd=SGD(model.parameters(),0.01)