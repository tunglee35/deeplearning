import torch

def test_basic():
    x1 = torch.Tensor([2.2]).double(); x1.requires_grad = True
    x2 = torch.Tensor([0.5]).double(); x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
    b = torch.Tensor([5.0]).double(); b.requires_grad = True

    net = x1*w1 + x2*w2 + b
    loss = torch.tanh(net)

    print(f'original loss {loss.data.item()}')

    loss.backward()

    print(f'x1.grad = {x1.grad}')
    print(f'x2.grad = {x2.grad}')
    print(f'w1.grad = {w1.grad}')
    print(f'w2.grad = {w2.grad}')

    # Attemp: increase w1 by
    w1.data += 0.555
    net = x1*w1 + x2*w2 + b
    loss = torch.tanh(net)

    print(f'loss after increasing w1 = {loss.data.item()}') # see for yourself


test_basic()


