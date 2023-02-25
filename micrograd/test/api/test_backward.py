from micrograd.engine import Value

def test_add_backward():
    a = Value(1.2)
    b = Value(2.5)

    c = a + b
    c.backward()
    assert a.grad == 1
    assert b.grad == 1

def test_add_backward_twice():
    a = Value(1.2)
    b = Value(2.5)

    c = a + b
    c.backward()
    assert a.grad == 1
    assert b.grad == 1

    c.backward()
    assert a.grad == 2 # ???
    assert b.grad == 2

def test_pow_backward():
    a = Value(1.2)

    c = a ** 2
    print(c)
    c.backward()
    assert a.grad == 2.4
