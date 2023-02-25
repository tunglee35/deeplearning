from micrograd.engine import Value

a = Value(1.2)
b = Value(2.5)

def test_add():
    c = a + b
    assert (c.data) == 3.7

def test_relu_greater_than_0():
    assert (a+b).relu().data > 0

def test_relu_less_than_0():
    assert (a-b).relu().data == 0
