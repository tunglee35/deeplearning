from micrograd.engine import Value
from micrograd.nn import *
import random

def test_basic():
    nin = 3 # number of inputs
    weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
    print(weights)

    # multiply each input by its weight
    bias = Value(3.0)
    inputs = [1.2, 2.5, 3.0]
    act = sum((wi*xi for wi,xi in zip(weights, inputs)), bias)

    print(list((wi*xi for wi,xi in zip(weights, inputs))))
    print(act)

def test_neuron():
    nin = 3 # number of inputs
    n = Neuron(nin)

    # check random weights and bias are intialized
    print(n.parameters())
    x = [1.0, 2.9]
    out = n(x)

    print(out)

def test_layer():
    nin = 3 # number of inputs
    nout = 2 # number of outputs
    l = Layer(nin, nout)

    # check random weights and bias are intialized
    print(l.parameters())
    x = [1.0, 2.9, 0.5]
    out = l(x)

    print(out)

def test_mlp():
    nin = 3 # number of inputs
    nouts = [2, 3, 1] # number of outputs
    mlp = MLP(nin, nouts)

    # check random weights and bias are intialized
    print(mlp.parameters())
    x = [1.0, 2.9, 0.5]
    out = mlp(x)

    print(out)

# test_neuron()
test_layer()
# test_mlp()