import torch
import numpy as np
from vanilla_nn.engine import Tensor
from vanilla_nn.nn import Linear, MLP

def test_vector_ops():
    # 1. Test Matrix Multiplication Forward & Backward
    x_data = np.random.randn(2, 3)
    w_data = np.random.randn(3, 4)
    
    x_v = Tensor(x_data)
    w_v = Tensor(w_data)
    out_v = x_v @ w_v
    out_v.backward()
    
    x_t = torch.tensor(x_data, requires_grad=True, dtype=torch.float32)
    w_t = torch.tensor(w_data, requires_grad=True, dtype=torch.float32)
    out_t = x_t @ w_t
    out_t.backward(torch.ones_like(out_t))
    
    tol = 1e-5
    assert np.allclose(out_v.data, out_t.detach().numpy(), atol=tol)
    assert np.allclose(x_v.grad, x_t.grad.numpy(), atol=tol)
    assert np.allclose(w_v.grad, w_t.grad.numpy(), atol=tol)
    print("Matrix multiplication test passed!")

def test_linear_layer():
    # 2. Test Linear Layer
    nin, nout = 10, 5
    lin = Linear(nin, nout)
    
    x_data = np.random.randn(3, nin)
    x_v = Tensor(x_data)
    out_v = lin(x_v)
    out_v.backward()
    
    x_t = torch.tensor(x_data, requires_grad=True, dtype=torch.float32)
    w_t = torch.tensor(lin.weight.data.T, requires_grad=True, dtype=torch.float32) # PyTorch expects (nout, nin)
    b_t = torch.tensor(lin.bias.data, requires_grad=True, dtype=torch.float32)
    
    out_t = torch.nn.functional.linear(x_t, w_t, b_t)
    out_t.backward(torch.ones_like(out_t))
    
    tol = 1e-5
    assert np.allclose(out_v.data, out_t.detach().numpy(), atol=tol)
    assert np.allclose(x_v.grad, x_t.grad.numpy(), atol=tol)
    assert np.allclose(lin.weight.grad, w_t.grad.numpy().T, atol=tol)
    assert np.allclose(lin.bias.grad, b_t.grad.numpy(), atol=tol)
    print("Linear layer test passed!")

def test_mlp_vectorized():
    # 3. Test MLP with Batched Input
    batch_size = 4
    nin = 3
    nouts = [4, 4, 1]
    mlp = MLP(nin, nouts, activation='relu')
    
    x_data = np.random.randn(batch_size, nin)
    x_v = Tensor(x_data)
    out_v = mlp(x_v)
    out_v.backward()
    
    # Verify shape
    assert out_v.data.shape == (batch_size, 1)
    print("MLP vectorized forward/backward passed!")

if __name__ == "__main__":
    test_vector_ops()
    test_linear_layer()
    test_mlp_vectorized()
    print("All vectorized tests passed!")
