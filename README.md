# learngrad

[![CI](https://github.com/devpparikh/learngrad/actions/workflows/ci.yml/badge.svg)](https://github.com/devpparikh/learngrad/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/learngrad)](https://pypi.org/project/learngrad/)
[![Python](https://img.shields.io/pypi/pyversions/learngrad)](https://pypi.org/project/learngrad/)

A minimal scalar-valued autograd engine with neural network primitives, built for learning purposes. Inspired by [micrograd](https://github.com/karpathy/micrograd).

## Install

```bash
pip install learngrad
```

## What's inside

- **`Value`** — scalar with automatic differentiation via backprop
- **`MLP`** — multi-layer perceptron built on top of `Value`
- **Optimizers** — `SGD`, `SGDMomentum`, `RMSProp`, `Adam`

## Quick example

```python
from learngrad.engine import Value
from learngrad.nn import  MLP
from learngrad.optimizers import Adam

# Autograd
x = Value(2.0)
y = x ** 2 + x * 3
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7.0

# Neural net
model = MLP(2, [4, 4, 1])
opt = Adam(model.parameters(), lr=1e-3)

x = [Value(1.0), Value(0.5)]
out = model(x)
out.backward()
opt.step()
```

## Demo

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) walks through a full training example:

- Binary classification on the `make_circles` dataset
- MLP with hinge loss and L2 regularization
- Mini-batch training loop with Adam
- Train/val accuracy tracking
- Decision boundary visualization

Reaches ~94% val accuracy in 100 epochs.

## Requirements

- Python >= 3.10
- numpy

## Tests

```bash
pip install -e ".[test]"
pytest
```

Every operation's backward pass is checked against a finite-difference
gradient, and each optimizer is checked to fit a simple linear model.
