# autodiff-engine
Automatic differentiation engine built from scratch with computational graphs and backpropagation.
---
## File Structure
```
autodiff-engine/
├── engine/
│   ├── tensor.py        # core Variable/Tensor class
│   ├── ops.py           # operations (add, mul, relu etc)
│   └── graph.py         # computation graph
├── optim/
│   └── sgd.py           # gradient descent
├── nn/
│   ├── layers.py        # Linear, activations
│   └── loss.py          # MSE, CrossEntropy
├── viz/
│   └── graph_viz.py     # computation graph visualization
├── demos/
│   ├── scalar_demo.py
│   ├── regression_demo.py
│   └── xor_demo.py
├── tests/
├── LEARNINGS.md         # this matters, keep it updated
└── README.md
```
