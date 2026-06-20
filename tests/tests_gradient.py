import numpy as np
import pytest
from nanograd.engine.tensor import Tensor
from nanograd.engine.utils import gradient_check


np.random.seed(42)


def test_add_broadcast():
    def func(a, b):
        return (a + b).sum()
    gradient_check(func, [np.random.randn(3, 4), np.random.randn(4)])


def test_mul_broadcast():
    def func(a, b):
        return (a * b).sum()
    gradient_check(func, [np.random.randn(3, 4), np.random.randn(3, 1)])


def test_matmul():
    def func(a, b):
        return (a @ b).sum()
    gradient_check(func, [np.random.randn(3, 4), np.random.randn(4, 2)])


def test_exp_mul():
    def func(a, b):
        return ((a * b).exp()).sum()
    gradient_check(func, [np.random.randn(2, 3), np.random.randn(2, 3)])


def test_pow():
    def func(a):
        return ((a + 1) ** 2).sum()
    gradient_check(func, [np.random.randn(4, 4)])


def test_mixed_graph():
    def func(a, b):
        return ((a @ b) + a).exp().sum()
    gradient_check(func, [np.random.randn(3, 3), np.random.randn(3, 3)])


def test_sum_axis0():
    def func(a):
        return a.sum(axis=0, keepdims=True).sum()
    gradient_check(func, [np.random.randn(5, 3)])


def test_sum_axis1():
    def func(a):
        return a.sum(axis=1, keepdims=True).sum()
    gradient_check(func, [np.random.randn(5, 3)])


def test_reshape():
    def func(a):
        return a.reshape((6, 2)).sum()
    gradient_check(func, [np.random.randn(3, 4)])


def test_tanh():
    def func(a):
        return a.tanh().sum()
    gradient_check(func, [np.random.randn(3, 4)])


def test_sigmoid():
    def func(a):
        return a.sigmoid().sum()
    gradient_check(func, [np.random.randn(3, 4)])


def test_relu():
    def func(a):
        # Avoid x=0 where gradient is undefined
        a_shifted = a + 2.0
        return a_shifted.relu().sum()
    gradient_check(func, [np.random.randn(3, 4)])


def test_log():
    def func(a):
        return (a.exp()).log().sum()  # keep values positive
    gradient_check(func, [np.random.randn(3, 4)])


def test_mean():
    def func(a):
        return a.mean()
    gradient_check(func, [np.random.randn(3, 4)])


def test_softmax():
    def func(a):
        return a.softmax().sum()
    gradient_check(func, [np.random.randn(5)])


def test_softmax_weighted():
    def func(a):
        s = a.softmax()
        return (s * Tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))).sum()
    gradient_check(func, [np.random.randn(5)])


def test_transpose_matmul():
    def func(a, b):
        return (a @ b.transpose()).sum()
    gradient_check(func, [np.random.randn(3, 4), np.random.randn(2, 4)])

