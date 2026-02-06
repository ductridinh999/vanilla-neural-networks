import numpy as np
from .engine import Tensor

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []

    def train(self, mode=True):
        self.training = mode
        for p in self.__dict__.values():
            if isinstance(p, Module):
                p.train(mode)
            elif isinstance(p, list):
                for item in p:
                    if isinstance(item, Module):
                        item.train(mode)

    def eval(self):
        self.train(False)

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        k = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = Tensor(np.random.uniform(-k, k, (out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = Tensor(np.random.uniform(-k, k, (out_channels,))) if bias else None

    def __call__(self, x):
        N, C, H, W = x.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # im2col
        cols = x.im2col(self.kernel_size, self.stride, self.padding)
        
        # Reshape weights and matmul
        out = self.weight.reshape(self.out_channels, -1) @ cols
        
        # Reshape output
        out = out.reshape(self.out_channels, out_h * out_w, N)
        out = out.transpose(2, 0, 1)
        out = out.reshape(N, self.out_channels, out_h, out_w)
        
        if self.bias:
            out = out + self.bias.reshape(1, self.out_channels, 1, 1)
        return out

    def parameters(self):
        params = [self.weight]
        if self.bias:
            params.append(self.bias)
        return params

    def __repr__(self):
        return f"Conv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size})"

class Linear(Module):

    def __init__(self, nin, nout, bias=True):
        k = 1 / np.sqrt(nin)
        self.weight = Tensor(np.random.uniform(-k, k, (nin, nout)))
        self.bias = Tensor(np.random.uniform(-k, k, (nout,))) if bias else None

    def __call__(self, x):
        out = x @ self.weight
        if self.bias:
            out = out + self.bias
        return out

    def parameters(self):
        params = [self.weight]
        if self.bias:
            params.append(self.bias)
        return params

    def __repr__(self):
        return f"Linear({self.weight.data.shape[0]}, {self.weight.data.shape[1]})"


class ReLU(Module):
    def __call__(self, x): return x.relu()
    def __repr__(self): return "ReLU()"

class Tanh(Module):
    def __call__(self, x): return x.tanh()
    def __repr__(self): return "Tanh()"

class Sigmoid(Module):
    def __call__(self, x): return x.sigmoid()
    def __repr__(self): return "Sigmoid()"

class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def __call__(self, x): return x.leaky_relu(self.alpha)
    def __repr__(self): return f"LeakyReLU({self.alpha})"

class GELU(Module):
    def __call__(self, x): return x.gelu()
    def __repr__(self): return "GELU()"

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

    def __call__(self, x):
        if self.training:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data
            
            x_hat = (x - batch_mean) * (batch_var + self.eps)**-0.5
        else:
            x_hat = (x - Tensor(self.running_mean)) * (Tensor(self.running_var) + self.eps)**-0.5
            
        return self.gamma * x_hat + self.beta

    def parameters(self):
        return [self.gamma, self.beta]

    def __repr__(self):
        return f"BatchNorm1d({self.gamma.data.shape[0]}, eps={self.eps}, momentum={self.momentum})"

class MLP(Module):

    def __init__(self, nin, nouts, activation='relu'):
        sz = [nin] + nouts
        self.layers = []
        
        for i in range(len(nouts)):
            self.layers.append(Linear(sz[i], sz[i+1]))
            
            if i < len(nouts) - 1:
                if activation == 'relu':
                    self.layers.append(ReLU())
                elif activation == 'tanh':
                    self.layers.append(Tanh())
                elif activation == 'sigmoid':
                    self.layers.append(Sigmoid())
                elif activation == 'leaky_relu':
                    self.layers.append(LeakyReLU())
                elif activation == 'gelu':
                    self.layers.append(GELU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"