# Contributing to nanograd

Thanks for your interest. nanograd is intentionally minimal. Contributions that add ops, layers, or fix correctness bugs are welcome. Contributions that add heavy dependencies or move away from the numpy-only core are out of scope.

---

## What's in scope

- New tensor ops (e.g. `conv2d`, `embedding`, `dropout`)
- New layers (e.g. `BatchNorm`, `Conv2d`)
- New optimizers (e.g. `Adam`, `RMSProp`)
- Bug fixes in backward passes (include a failing gradient check in the PR)
- Improving test coverage

## What's out of scope

- GPU support
- Adding PyTorch/JAX as a dependency
- Rewriting the engine architecture

---

## Adding a new op

All ops live in `nanograd/engine/tensor.py` as methods on `Tensor`. Each op needs:

1. A forward pass that computes the output
2. A `_backward` closure that accumulates gradients into `self.grad` (and other inputs)
3. A gradient check test in `tests/test_gradients.py`

The pattern every op follows:

```python
def your_op(self):
    out = Tensor(np.your_op(self.data), (self,), 'your_op')

    def _backward():
        self.grad += <derivative of your_op w.r.t self> * out.grad

    out._backward = _backward
    return out
```

Use `_unbroadcast` from `nanograd.engine.utils` any time the output shape might differ from the input shape due to broadcasting.

---

## Adding a layer

Layers live in `nanograd/nn/layers.py` and inherit from `Module`:

```python
class YourLayer(Module):
    def __init__(self, ...):
        # initialize Tensor parameters here

    def parameters(self):
        return [self.w, self.b]   # all Tensors that need gradients

    def forward(self, x):
        # return a Tensor
```

Export it from `nanograd/nn/__init__.py`.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Every new op or layer needs a corresponding gradient check:

```python
def test_your_op():
    def func(a):
        return a.your_op().sum()
    gradient_check(func, [np.random.randn(3, 4)])
```

`gradient_check` raises `AssertionError` if analytical and numerical gradients differ by more than `1e-4`. If your op involves discontinuities (like ReLU at 0), shift inputs away from the discontinuity in the test.

---

## Submitting a PR

- One PR per op/feature
- Include the gradient check test
- Don't reformat unrelated files
- PR description should state what op/layer is added and link to any reference (paper, numpy docs) for the gradient derivation if it's non-trivial

---

## Questions

Open an issue or reach out to [@duaasiraj](https://github.com/duaasiraj).

> Disclaimer: This project was developed as a learning project and may contain mistakes, inefficiencies, or incomplete implementations. If you spot an issue or have an improvement, feel free to open an issue or submit a pull request. Check out contribution.md for further details :)