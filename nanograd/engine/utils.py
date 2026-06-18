import numpy as np


def topo_sort(root):
    topo = []
    visited = set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)
    build(root)
    return topo

# part one: remove extra dimension that didn’t exist in original tensor
# part two: handle dimensions where original size was 1
def _unbroadcast(grad, original_shape):
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for i, (g, o) in enumerate(zip(grad.shape, original_shape)):
        if o == 1 and g>1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


#SANITY CHECK-COMPARE NUMERIAL AND AUTOMATIC DIFF T_T

def gradient_check(tensor_func, inputs, epsilon=1e-5, tolerance=1e-4):
    from nanograd.engine.tensor import Tensor
    tensor_inputs = [Tensor(x.copy()) for x in inputs]
    out = tensor_func(*tensor_inputs)
    assert out.data.size == 1, "tensor_func must return a scalar tensor (size=1)."
    out.backward()
    analytical_grads = [t.grad.copy() for t in tensor_inputs]
    numerical_grads = []
    for x in inputs:
        num_grad = np.zeros_like(x)
        with np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) as it:
            for val in it:
                idx = it.multi_index
                original = val.copy()  
                # Evaluate f(x + eps)
                x[idx] = original + epsilon
                out_plus = tensor_func(*[Tensor(inp.copy()) for inp in inputs]).data.item()
                # Evaluate f(x - eps)
                x[idx] = original - epsilon
                out_minus = tensor_func(*[Tensor(inp.copy()) for inp in inputs]).data.item()
                x[idx] = original
                num_grad[idx] = (out_plus - out_minus) / (2 * epsilon)
                
        numerical_grads.append(num_grad)
    all_passed = True
    for i, (ag, ng) in enumerate(zip(analytical_grads, numerical_grads)):
        diff = np.max(np.abs(ag - ng)) 
        passed = diff < tolerance
        
        print(f"Input {i}: max diff = {diff:.2e} → {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False

    return all_passed