"""
Neural Network Framework - Built from scratch with LinAlgKit
============================================================

This package provides a minimal but complete neural network implementation
for educational purposes. Every component includes detailed comments explaining
the math behind deep learning.
"""

from .layers import Dense, Dropout
from .activations import ReLU, Sigmoid, Softmax, Tanh
from .losses import CrossEntropyLoss, MSELoss
from .optimizers import SGD, Adam
from .model import Sequential

__all__ = [
    'Dense', 'Dropout',
    'ReLU', 'Sigmoid', 'Softmax', 'Tanh',
    'CrossEntropyLoss', 'MSELoss',
    'SGD', 'Adam',
    'Sequential'
]
