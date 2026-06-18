from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
from nanograd.engine.tensor import Tensor
from nanograd.nn.modules import Module
from nanograd.nn.sequential import Sequential
from nanograd.nn.layers import Linear
from nanograd.nn.layers import ReLU
from nanograd.nn.loss import CrossEntropyLoss
from nanograd.nn.optim import SGD
from nanograd.nn.accuracy import multiclass_Accuracy
from nanograd.nn.metrics import multiclass_precision ,multiclass_recall, f1score
np.random.seed(42)

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
loss=CrossEntropyLoss()
sgd=SGD(model.parameters(),0.01)

batch_size =32
epochs=10
for epoch in range(epochs):
    train_loss=[]
    train_acc=[]
    train_prec = []
    train_rec = []
    train_f1 = []
    print(f'=========Epoch {epoch}=========')
    for i in range(0, len(X_train),batch_size):
        batch_X=X_train[i:i+batch_size]
        batch_y=y_train[i:i+batch_size]
        bx=Tensor(batch_X)
        sgd.zero_grad()
        out=model.forward(bx)
        l=loss(out,batch_y)
        acc=multiclass_Accuracy(out,batch_y)
        prec = multiclass_precision(out.data, batch_y)
        rec = multiclass_recall(out.data, batch_y)
        f1 = f1score(out.data, batch_y)
        train_loss.append(l.data)
        train_acc.append(acc)
        train_prec.append(prec)
        train_rec.append(rec)
        train_f1.append(f1)
        l.backward()
        sgd.step()
    print("Training loss:", np.mean(train_loss))
    print("Training accuracy:",np.mean(train_acc))
    print()


#---TEST DATA -----------------
test_loss=[]
test_acc=[]
test_prec = []
test_rec = []
test_f1 = []


for i in range(0, len(X_test), batch_size):
        batch_X = X_test[i:i+batch_size]
        batch_y = y_test[i:i+batch_size]
        bx = Tensor(batch_X)
        out = model.forward(bx)
        l = loss(out, batch_y)
        acc = multiclass_Accuracy(out, batch_y)
        prec = multiclass_precision(out.data, batch_y)
        rec = multiclass_recall(out.data, batch_y)
        f1 = f1score(out.data, batch_y)
        test_loss.append(l.data)
        test_acc.append(acc)
        test_prec.append(prec)
        test_rec.append(rec)
        test_f1.append(f1)

print("================================")
print("Final Results:")
print("================================")
print("Training loss:", np.mean(train_loss))
print("Training accuracy:",np.mean(train_acc))
print("Training precision:", np.mean(train_prec))
print("Training recall:", np.mean(train_rec))
print("Training F1:", np.mean(train_f1))
print("================================")
print("Test loss:", np.mean(test_loss))
print("Test accuracy:",np.mean(test_acc))
print("Test precision:", np.mean(test_prec))
print("Test recall:", np.mean(test_rec))
print("Test F1:", np.mean(test_f1))
print("================================")
