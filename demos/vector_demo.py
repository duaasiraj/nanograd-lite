import numpy as np
from engine.tensor import Tensor
from engine.utils import gradient_check

#Sanity checks 

np.random.seed(42)

def test_add():
    def func(a, b):
        return (a + b).sum()

    a = np.random.randn(3, 4)
    b = np.random.randn(4) 
    A = Tensor(a)
    B = Tensor(b)

    out = A + B
    out.backward()

    print("ADD grad OK")
    gradient_check(func, [a, b])


def test_mul():
    def func(a, b):
        return (a * b).sum()

    a = np.random.randn(3, 4)
    b = np.random.randn(3, 1) 

    A = Tensor(a)
    B = Tensor(b)

    out = A * B
    out.backward()

    print("MUL grad OK")
    gradient_check(func, [a, b])


def test_matmul():
    def func(a, b):
        return (a @ b).sum()

    a = np.random.randn(3, 4)
    b = np.random.randn(4, 2)

    A = Tensor(a)
    B = Tensor(b)

    out = A@B
    out.backward()

    print("MATMUL grad OK")
    gradient_check(func, [a, b])


def test_exp_mul():
    def func(a, b):
        return ((a * b).exp()).sum()

    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)

    A = Tensor(a)
    B = Tensor(b)

    out = (A * B).exp()
    out.backward()

    print("EXP+MUL grad OK")
    gradient_check(func, [a, b])


def test_pow():
    def func(a):
        return ((a + 1) ** 2).sum()

    a = np.random.randn(4, 4)

    A = Tensor(a)

    out = (A + 1) ** 2
    out.backward()

    print("POW grad OK")
    gradient_check(func, [a])

def test_mixed():
    def func(a, b):
        return ((a @ b) + a).exp().sum()

    a = np.random.randn(3, 3)
    b = np.random.randn(3, 3)

    A = Tensor(a)
    B = Tensor(b)

    out = ((A @ B) + A).exp()
    out.backward()

    print("MIXED GRAPH OK")
    gradient_check(func, [a, b])

def test_sum_axis():
    def func(a):
        return np.sum(a)

    a = np.random.randn(5, 3)

    A = Tensor(a)
    out = A.sum(axis=0, keepdims=True).sum()
    out.backward()

    print("SUM(axis=0) grad OK")
    gradient_check(func, [a])

def test_sum_axis1():
    def func(a):
        return np.sum(a)  

    a = np.random.randn(5, 3)

    A = Tensor(a)
    out = A.sum(axis=1, keepdims=True).sum()  
    out.backward()

    print("SUM(axis=1) grad OK")
    gradient_check(func, [a])

def test_reshape():
    def func(a):
        return a.reshape((6, 2)).sum()

    a = np.random.randn(3, 4)

    A = Tensor(a)
    out = A.reshape((6, 2)).sum()
    out.backward()

    print("RESHAPE grad OK")
    gradient_check(func, [a])

def test_softmax():
    def func(a):
        s = np.exp(a - np.max(a))
        s = s / np.sum(s)
        return np.sum(s)

    a = np.random.randn(5)

    A = Tensor(a)
    out = A.softmax().sum()
    out.backward()

    print("SOFTMAX grad OK")
    gradient_check(func, [a])

def test_softmax_weighted():
    def func(a):
        s = np.exp(a - np.max(a))
        s = s / np.sum(s)
        g = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        return np.sum(s * g)

    a = np.random.randn(5)

    A = Tensor(a)
    s = A.softmax()
    out = (s * Tensor(np.array([1,2,3,4,5]))).sum()
    out.backward()

    print("SOFTMAX weighted grad OK")
    gradient_check(func, [a])

if __name__ == "__main__":
    test_add()
    test_mul()
    test_matmul()
    test_exp_mul()
    test_pow()
    test_mixed()
    test_softmax()
    test_softmax_weighted()
    test_reshape()
    test_sum_axis1()
    test_sum_axis()
