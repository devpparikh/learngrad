import numpy as np
import random
from itertools import groupby
from .engine import Value


class Module():
    def parameters(self): return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):

    def __init__(self, nin, nonlin=True) -> None:
        super().__init__()
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        act = (sum((wi*xi for wi, xi in zip(self.w, x)), self.b))
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs) -> None:
        super().__init__()
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        result = []
        for key, group in groupby(self.neurons, key=str):
            count = sum(1 for _ in group)
            result.append(f"{count} x {key}" if count > 1 else key)
        return "Layer(" + ", ".join(result) + ")"


class MLP(Module):
    def __init__(self, nin, nouts) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(nin=sz[i], nout=sz[i+1], nonlin=i !=
                             len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        layers = ",\n".join(str(layer) for layer in self.layers)
        return f"MLP(\n{layers}\n)"
