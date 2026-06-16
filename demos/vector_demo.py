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

if __name__ == "__main__":
    test_add()
    test_mul()
    test_matmul()
    test_exp_mul()
    test_pow()
    test_mixed()