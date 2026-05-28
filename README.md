# autodiff-engine
Automatic differentiation engine built from scratch with computational graphs and backpropagation.
---
## File Structure
```
autodiff-engine/

├── engine/                  # CORE AUTODIFF SYSTEM (heart of the framework)
│   ├── tensor.py           # Main Tensor class: data, grad, requires_grad, backward trigger
│   ├── ops.py              # Low-level math ops: add, mul, matmul, relu, exp + backward rules
│   ├── autograd.py         # Backprop engine: graph traversal + gradient propagation logic
│   ├── function.py         # Function abstraction: each operation as a class with forward/backward
│   ├── context.py          # Stores saved tensors from forward pass for backward computation
│   ├── graph.py            # Builds computation graph structure for visualization/debugging
│   └── utils.py           # Helpers: topo sort, shape checks, gradient helpers, debugging tools
│
├── nn/                     # NEURAL NETWORK LAYER API (user-facing ML building blocks)
│   ├── modules.py          # Base Module class: parameters(), forward(), zero_grad()
│   ├── layers.py           # Layers like Linear, (future: Conv, Embedding)
│   ├── activation.py       # Activation functions: ReLU, Sigmoid, Tanh, Softmax
│   ├── loss.py             # Loss functions: MSE, CrossEntropy, BCE
│   ├── init.py             # Weight initialization methods: Xavier, He init
│   └── sequential.py       # Container to stack layers like PyTorch Sequential
│
├── optim/                  # OPTIMIZATION / TRAINING ALGORITHMS
│   ├── sgd.py              # Stochastic Gradient Descent optimizer
│   ├── adam.py             # Adam optimizer (momentum + adaptive learning rate)
│   └── base.py            # Base optimizer class interface (step, zero_grad)
│
├── viz/                    # VISUALIZATION + DEBUGGING TOOLS
│   ├── graph_viz.py       # Visualize computation graphs (nodes + edges + ops)
│   ├── tensor_viz.py      # Visualize tensors, shapes, and gradient flow
│   └── training_plots.py  # Plot loss curves, accuracy graphs over epochs
│
├── demos/                 # SMALL EXPERIMENTS (proof engine actually works)
│   ├── scalar_demo.py     # Basic scalar backprop tests (sanity check gradients)
│   ├── xor_demo.py        # XOR problem: proves non-linearity + learning works
│   ├── regression_demo.py # Curve fitting using simple neural network
│   ├── mnist_demo.py      # Full MNIST training experiment
│   └── gradient_check_demo.py # Numerical gradient vs autodiff verification
│
├── tests/                 # UNIT TESTS (ensures correctness of core engine)
│   ├── test_tensor.py     # Tests tensor ops + gradient correctness
│   ├── test_ops.py        # Tests mathematical operations + edge cases
│   ├── test_autograd.py   # Tests backward pass + graph traversal
│   ├── test_nn.py         # Tests neural network layers + forward/backward
│   └── test_optim.py      # Tests optimizer updates (SGD, Adam correctness)
│
├── docs/                  # HUMAN DOCUMENTATION (explain + justify everything)
│   ├── index.md           # Main project overview + how to use
│   ├── autodiff_exp.md # Explains backprop + computational graph intuition
│   ├── tensor_system.md   # Explains Tensor design + data/grad flow
│   ├── backprop_notes.md  # Mathematical + practical notes on gradient flow
│   └── mnist_experiment.md # Results, insights, and training observations
│
├── benchmarks/            # PERFORMANCE + ENGINEERING CREDIBILITY
│   ├── speed_test.py      # Measures forward/backward pass speed
│   ├── memory_test.py     # Measures memory usage of graph + tensors
│   └── vs_numpy.py        # Compares performance with NumPy operations
│
├── assets/                # VISUAL OUTPUTS (used in README + docs)
│   ├── graphs/            # Computation graph images
│   ├── mnist_samples/     # Example predictions (correct vs wrong digits)
│   ├── training_curves/   # Loss/accuracy plots over training
│   └── diagrams/          # Architecture diagrams of engine design
│
├── scripts/               # REAL-WORLD EXECUTION ENTRY POINTS
│   ├── train_mnist.py     # Full MNIST training pipeline script
│   ├── train_xor.py       # Quick training demo for XOR problem
│   ├── run_tests.sh       # Automated test runner script
│   └── export_model.py    # Save trained model parameters for reuse
│
├── examples/              # SIMPLE STARTER CODE (for users/new learners)
│   ├── basic_autograd.py  # Minimal gradient example (x² style)
│   ├── simple_nn.py       # Small neural network training example
│   └── mnist_minimal.py   # Minimal MNIST training script (clean version)
│
├── LEARNINGS.md           # YOUR PERSONAL ENGINEERING JOURNAL (VERY IMPORTANT)
├── README.md              # Main project explanation + quick start + visuals
├── CONTRIBUTING.md        # How others can extend ops/layers/optimizers
├── CHANGELOG.md           # Version history of improvements
├── LICENSE                # Legal usage (MIT recommended)
├── pyproject.toml         # Python packaging config (installable library setup)
├── requirements.txt       # Dependencies (numpy, matplotlib, graphviz, etc.)
└── .gitignore             # Ignore venv, caches, build files
```
