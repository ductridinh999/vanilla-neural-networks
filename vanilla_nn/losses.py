from .engine import Value

def mse_loss(y_pred, y_target):
    # MSE Loss
    diff = y_pred - y_target
    return diff**2

def binary_cross_entropy(y_pred, y_target):
    # Binary Cross Entropy Loss
    return - (y_target * y_pred.log() + (1 - y_target) * (1 - y_pred).log())

def hinge_loss(y_pred, y_target):
    # Hinge Loss
    return (1 + -(y_target * y_pred)).relu()