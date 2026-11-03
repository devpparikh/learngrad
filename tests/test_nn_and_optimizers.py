import random

import pytest

from learngrad.engine import Value
from learngrad.nn import MLP, Layer, Neuron
from learngrad.optimizers import SGD, Adam, RMSProp, SGDMomentum


def test_parameter_counts():
    assert len(Neuron(3).parameters()) == 4
    assert len(Layer(3, 2).parameters()) == 8
    # 2->4 (12) + 4->4 (20) + 4->1 (5)
    assert len(MLP(2, [4, 4, 1]).parameters()) == 37


def test_mlp_forward_returns_scalar_for_single_output():
    out = MLP(2, [4, 1])([Value(1.0), Value(-1.0)])
    assert isinstance(out, Value)


def test_reprs():
    assert repr(Neuron(3)) == "ReLUNeuron(3)"
    assert repr(Layer(3, 2, nonlin=False)) == "Layer(2 x LinearNeuron(3))"
    assert repr(MLP(2, [3, 1])) == "MLP(\nLayer(3 x ReLUNeuron(2)),\nLayer(LinearNeuron(3))\n)"


def test_zero_grad_resets_all_parameters():
    model = MLP(2, [3, 1])
    model([Value(1.0), Value(2.0)]).backward()
    model.zero_grad()
    assert all(p.grad == 0 for p in model.parameters())


@pytest.mark.parametrize(
    "make_opt",
    [
        lambda ps: SGD(ps, lr=0.05),
        lambda ps: SGDMomentum(ps, lr=0.05),
        lambda ps: RMSProp(ps, lr=0.01),
        lambda ps: Adam(ps, lr=0.05),
    ],
    ids=["SGD", "SGDMomentum", "RMSProp", "Adam"],
)
def test_optimizers_reduce_loss_on_a_linear_fit(make_opt):
    random.seed(0)
    model = MLP(1, [1])  # a single linear neuron
    opt = make_opt(model.parameters())
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ys = [2 * x + 1 for x in xs]

    def loss():
        return sum((model([Value(x)]) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)

    start = loss().data
    for _ in range(200):
        opt.zero_grad()
        current = loss()
        current.backward()
        opt.step()
    assert loss().data < start * 0.05
