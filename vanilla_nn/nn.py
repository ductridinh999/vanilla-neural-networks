import numpy as np
from .engine import Tensor

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []

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

# --- Containers ---

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