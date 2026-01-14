import torch
from vanilla_nn.engine import Value

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

def test_activations():
    # Test tanh
    a = Value(-0.5)
    b = a.tanh()
    b.backward()
    amg, bmg = a, b

    a = torch.Tensor([-0.5]).double()
    a.requires_grad = True
    b = torch.tanh(a)
    b.backward()
    apt, bpt = a, b

    tol = 1e-6
    assert abs(bmg.data - bpt.data.item()) < tol
    assert abs(amg.grad - apt.grad.item()) < tol

    # Test sigmoid
    a = Value(0.8)
    b = a.sigmoid()
    b.backward()
    amg, bmg = a, b

    a = torch.Tensor([0.8]).double()
    a.requires_grad = True
    b = torch.sigmoid(a)
    b.backward()
    apt, bpt = a, b

    assert abs(bmg.data - bpt.data.item()) < tol
    assert abs(amg.grad - apt.grad.item()) < tol
