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