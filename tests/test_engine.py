"""Gradient checks: every op's backward pass must match a finite-difference estimate."""

import math

import pytest

from learngrad.engine import Value

EPS = 1e-6


def numeric_grad(f, *xs):
    """Central-difference gradient of scalar f with respect to each input."""
    grads = []
    for i in range(len(xs)):
        hi = list(xs)
        lo = list(xs)
        hi[i] += EPS
        lo[i] -= EPS
        grads.append((f(*hi) - f(*lo)) / (2 * EPS))
    return grads


def analytic_grad(f, *xs):
    vals = [Value(x) for x in xs]
    f(*vals).backward()
    return [v.grad for v in vals]


CASES = [
    ("add", lambda a, b: a + b, (2.0, -3.0)),
    ("sub", lambda a, b: a - b, (2.0, -3.0)),
    ("mul", lambda a, b: a * b, (2.0, -3.0)),
    ("div", lambda a, b: a / b, (2.0, -3.0)),
    ("pow", lambda a, b: a**3 + b * 0, (1.5, 0.0)),
    ("neg", lambda a, b: -a + b, (2.0, 1.0)),
    ("radd", lambda a, b: 4 + a + b, (2.0, 1.0)),
    ("rsub", lambda a, b: 4 - a * b, (2.0, 1.0)),
    ("rmul", lambda a, b: 4 * a * b, (2.0, 1.0)),
    ("rtruediv", lambda a, b: 4 / (a + b), (2.0, 1.0)),
    ("exp", lambda a, b: (a * b).exp(), (0.5, -1.2)),
    ("composite", lambda a, b: (a * b + a**2 - b / a).exp() * 0.1, (0.7, 0.4)),
]


@pytest.mark.parametrize("name,f,xs", CASES, ids=[c[0] for c in CASES])
def test_gradients_match_finite_differences(name, f, xs):
    analytic = analytic_grad(f, *xs)

    def scalar(*ys):
        out = f(*[Value(y) for y in ys])
        return out.data

    numeric = numeric_grad(scalar, *xs)
    for a, n in zip(analytic, numeric):
        assert math.isclose(a, n, rel_tol=1e-4, abs_tol=1e-6)


def test_relu_gradient_is_zero_for_negative_input():
    x = Value(-2.0)
    x.relu().backward()
    assert x.grad == 0


def test_relu_passes_gradient_for_positive_input():
    x = Value(3.0)
    y = x.relu()
    y.backward()
    assert y.data == 3.0
    assert x.grad == 1


def test_gradient_accumulates_when_a_value_is_reused():
    x = Value(3.0)
    (x * x + x).backward()  # d/dx (x^2 + x) = 2x + 1
    assert x.grad == 7.0


def test_readme_example():
    x = Value(2.0)
    y = x**2 + x * 3
    y.backward()
    assert x.grad == 7.0
