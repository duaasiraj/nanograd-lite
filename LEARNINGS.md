# LEARNINGS.md

> This file documents my exploratory work, flow of thought, mistakes made, and reasoning behind architectural decisions throughout the project. It is written chronologically and is intended to reflect genuine understanding, not polish.

---

## Phase 0 — Building the Tensor Class

### Motivation

The first question I had to answer was: why is backward differentiation necessary at all?

In a neural network, the loss is computed at the very end of the forward pass. To update the weights, we need to know how much each weight contributed to that loss -> which requires computing gradients with respect to every parameter in the network. Doing this manually for each operation would be both error-prone and unscalable. Automatic differentiation solves this by building a record of every operation during the forward pass and using that record to propagate gradients backwards automatically.

### Translating the Chain Rule into Code

Before writing any code, the mathematics had to be clear. The two rules that underpin all of backpropagation are:

**Chain rule for composition:**
```
dy/dx = dy/dt * dt/dx    given y = f(t) and t = g(x)
```

**Chain rule for addition (multivariate):**
```
dy/dx = dz/dx + da/dx    given y = z + a, where both z and a depend on x
```

Every backward rule in the engine is a direct application of one or both of these.

### Implementing Backward Rules

Each operation stores a `_backward` closure that knows how to distribute the incoming gradient to its inputs. Taking `exp` as an example: the derivative of `e^x` with respect to `x` is `e^x` itself, so the backward rule multiplies the output's gradient by the output's value.

**Critical implementation detail:** gradient accumulation must use `+=` rather than `=`.

A tensor can participate in multiple operations. For example, `z = x * x` uses `x` twice. Each operation contributes a separate gradient to `x`. Using `=` would silently overwrite the first contribution with the second, producing incorrect gradients. Using `+=` ensures all contributions are summed, which is what the multivariate chain rule requires. This was discovered through a bug before being properly understood.

### Automating Gradient Propagation

Once individual backward rules existed, the question became how to call them in the right order automatically.

The key insight: any mathematical expression can be represented as a directed acyclic graph (DAG), where nodes are tensors and edges represent operations. If that graph can be traversed in topological order, guaranteeing that every node is visited after all the nodes that depend on it,then calling `_backward` in reverse topological order correctly propagates gradients from the output back to every input.

The topological sort was implemented using a recursive depth-first search with a visited set to prevent revisiting nodes. Reversing the resulting order gives the correct sequence for backward propagation.

### Bugs Encountered

**`_backward` parameter shadowing**
Defining `def _backward(self)` inside `__mul__` caused `self` inside the closure to refer to the function's own parameter rather than the enclosing `Tensor` instance. The fix was removing the parameter entirely ,backward functions are closures, not methods.


**`Not accumulating the gradient**
As mentioned above, not accumulating resulted to overwriting which in turn resulted to wrong results being calculated.

### Tooling Issues (Windows)

**Module not found when running from inside a subfolder**
Running `python viz/graph_viz.py` directly failed because Python's import system resolves modules relative to the directory the script is run from, not where the file lives. The fix was always running from the project root using `python -m viz.graph_viz`.

**Graphviz `ExecutableNotFound` error**
The `graphviz` Python package is a wrapper, it requires the actual Graphviz software to be installed separately and available on the system PATH. On Windows, the installer does not always add the binary to PATH automatically. The temporary fix was:
```powershell
$env:PATH += ";C:\Program Files\Graphviz\bin"
```
The permanent fix was setting the Machine-level PATH variable via PowerShell as Administrator.

---
### Results 
A sanity check was performed to ensure that the backward differentiation works properly. The code was kept simple and can be checked out in `demos\scalar_demo.py`
![Resulting outcome](assets/computationgraph.svg)

---

## Phase 1 — Vector/Tensor Support

### Motivation

The scalar implementation worked, but real neural networks operate on tensors—multi-dimensional arrays of numbers. Extending the Tensor class to handle vector and matrix operations was the natural next step. The core challenge was not just implementing vectorized operations, but correctly handling broadcasting and ensuring gradients propagate correctly through these operations.

### Understanding Broadcasting

The first concept I had to deeply understand was broadcasting. In NumPy-style operations, when two arrays have different shapes, the smaller array is "broadcast" to match the larger one by duplicating data along dimensions of size 1. The rules are:

1. Arrays must have the same number of dimensions, or one can be missing dimensions on the left
2. For each dimension, the sizes must either match, or one of them must be 1 or 0

For example, a (3,1) array added to a (1,4) array results in a (3,4) array. The (3,1) array is stretched horizontally, and the (1,4) array is stretched vertically.

### The Broadcast Gradient Problem

The critical insight came when considering backpropagation through broadcasted operations. Let's trace through the problem:

Consider `a` with shape (3,1):
```
a = [[1],
     [2],
     [3]]
```

During a broadcast operation, `a` effectively becomes:
```
[[1, 1, 1, 1],
 [2, 2, 2, 2],
 [3, 3, 3, 3]]
```

Now, during backward propagation, suppose we receive a gradient of shape (3,4):
```
grad = [[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]
```

The question becomes: what is the gradient with respect to the original `a` of shape (3,1)?

Since each element of `a` influenced multiple outputs (specifically, `a[0][0]` influenced all 4 elements in the first row), the gradient must account for all those contributions. The correct approach is to **sum** along the broadcasted dimensions:

```
a.grad = [[4],
          [4],
          [4]]
```

Why sum? Because in backpropagation, gradients from all computational paths must be added. Each output that a parameter influenced contributes a partial derivative, and the total gradient is the sum of all these contributions. If we were to average or just pick one value, we would either artificially shrink gradients or ignore contributions entirely—both of which would break training.

### Implementing Unbroadcasting

To implement this, I created an `unbroadcast` function that would take the incoming gradient and reduce it along dimensions that were broadcasted. The algorithm works as:

1. Align the shapes of the gradient and the original tensor by left-padding the original tensor's shape with ones
2. Identify which dimensions in the gradient correspond to broadcasted dimensions in the original tensor
3. Sum the gradient along those dimensions

**Critical implementation detail:** The summing must happen in the correct order to maintain correct dimensions. I used `keepdims=True` in the summation to preserve the shape structure until all reductions are complete.

### MatMul Backward Derivation

Matrix multiplication was the most complex operation to implement, and it's where I spent considerable time debugging. Let me walk through the derivation:

Given matrices `A` (shape m×n) and `B` (shape n×p), the output `Out = A @ B` has shape m×p:

```
A = [[a, b],     B = [[e, f],
     [c, d]]          [g, h]]

Out = [[ae + bg,  af + bh],
       [ce + dg,  cf + dh]]
```

To compute the gradient with respect to `A`, we need the partial derivatives of the loss with respect to each element of `A`. If `Out.grad` is known (let's simplify with all ones for this derivation):

```
Out.grad = [[1, 1],
            [1, 1]]
```

For `a`: contributes to `Out[0][0]` through `ae` and `Out[0][1]` through `af`
- ∂L/∂a = Out.grad[0][0] * e + Out.grad[0][1] * f

For `b`: contributes to `Out[0][0]` through `bg` and `Out[0][1]` through `bh`
- ∂L/∂b = Out.grad[0][0] * g + Out.grad[0][1] * h

For `c`: contributes to `Out[1][0]` through `ce` and `Out[1][1]` through `cf`
- ∂L/∂c = Out.grad[1][0] * e + Out.grad[1][1] * f

For `d`: contributes to `Out[1][0]` through `dg` and `Out[1][1]` through `dh`
- ∂L/∂d = Out.grad[1][0] * g + Out.grad[1][1] * h

Observing the pattern, this can be expressed as:
```
A.grad = Out.grad @ B.T
```

Similarly, for `B.grad`:
```
B.grad = A.T @ Out.grad
```

These elegant formulations make the implementation straightforward but require careful attention to shape handling.

### Bugs Encountered

**Circular Dependency in Imports**
When implementing operations, I needed to use the `Tensor` class in functions that were themselves being used by `Tensor`. This created a circular import dependency. The fix was to import inside the function itself where needed, rather than at the module level. This works because Python executes imports at runtime, so by the time the function is called, the module is fully loaded.

**Gradient Check Failures**
The gradient checking function was failing repeatedly, but the issue wasn't in the Tensor implementation—it was in the test itself. The array `x` was being overwritten during the test, corrupting the state. The fix was to store the original array and restore it after each perturbation step. This taught me that gradient checking requires careful state management.

**MatMul Broadcasting Edge Cases**
Matrix multiplication has specific broadcasting rules: the last dimension of A must match the second-to-last dimension of B. When dealing with batched operations, the broadcasting dimensions must be handled differently from the matrix dimensions. I had to carefully separate the batch dimensions (which can broadcast) from the matrix dimensions (which must match exactly).

### Performance Considerations

Vectorized operations were significantly faster than their scalar counterparts, but the broadcasting implementation needed to be efficient. Rather than actually creating the broadcasted arrays in memory (which would be wasteful for large tensors), the backward pass uses the shape information to compute the sum along the correct axes without materializing the full broadcasted tensor.

### Sanity Checks

After implementing tensor operations, I ran comprehensive sanity checks:

1. **Scalar operations on tensors** verified that the tensor implementation was backward compatible
2. **Vector operations** tested dot products, element-wise operations, and their gradients
3. **Matrix multiplication** validated against analytical gradients for random matrices
4. **Broadcasting tests** verified that gradients were correctly accumulated across broadcasted dimensions

All checks passed, confirming that the tensor implementation correctly handles multi-dimensional arrays and their gradients.

### Final Results

A complete sanity check was performed to ensure that vector/tensor operations work correctly across all supported operations. The implementation was validated against analytical gradients using the gradient checking utility. The code can be found in `demos/vector_demo.py`.

![Resulting outcome](assets/phase1_sanitycheck.png)

The visualization shows the computation graph and confirms that gradients propagate correctly through vectorized operations, including those involving broadcasting. The tensor implementation now provides the foundation for building neural networks with multi-dimensional data.