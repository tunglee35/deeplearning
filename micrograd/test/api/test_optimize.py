from micrograd.engine import Value

def test_optimize_add():
    a = Value(1.2)
    b = Value(2.5)

    loss = a + b # 3.7 is too big, let's get it down to 0

    # Identify the rate of change of the loss
    # compare to each inputs
    loss.backward()

    # Both grads are 1, this means no low hanging fruit changes
    # both a and b can influcence the loss equally

    # Attemp: reduce a by 3.7
    a.data -= 3.7
    loss = a + b

    print(f'loss = #{loss}') # yay !


def test_optimize_multiply():
    a = Value(1.2)
    b = Value(2.5)

    loss = a * b # 3.0 is too big, let's get it down to 0

    # Identify the rate of change of the loss
    loss.backward()
    print('a has more influence on loss than b') if a.grad > b.grad else print('b has more influence on loss than a')

    # Attemp: reduce a by 1.2
    a.data -= 1.2
    loss = a * b

    print(f'loss after reducing a = #{loss}') # yay !

    # Attemp: reduce b by the same amount
    a = Value(1.2)
    b.data -= 1.2
    loss = a * b

    print(f'loss after reducing b = #{loss}') # see for yourself

def test_backward_complex_function():
    a = Value(1.2)
    b = Value(2.5)
    c = a + b

    d = Value(3.0)
    e = c * d
    f = Value(4.0)

    loss = e + f

    # now it gets more tricky, guess which variable impacts loss the most
    loss.backward()
    print(f'orginal loss = #{loss}')

    print(f'a.grad = {a.grad}')
    print(f'b.grad = {b.grad}')
    print(f'c.grad = {c.grad}')
    print(f'd.grad = {d.grad}')
    print(f'e.grad = {e.grad}')
    print(f'f.grad = {f.grad}')

    # Attemp: reduce d by 3.7
    d.data -= 3.7
    e = c * d
    loss = e + f

    print(f'loss after reducing d = #{loss}') # see for yourself

    # reset values
    d.data = 3.0
    e = c * d
    loss = e + f

    # Attemp: reduce c by 3.7
    c.data -= 3.7
    e = c * d
    loss = e + f

    print(f'loss after reducing c = #{loss}') # see for yourself


# Uncomment to see the results
# test_optimize_add()
# test_optimize_multiply()
# test_backward_complex_function()