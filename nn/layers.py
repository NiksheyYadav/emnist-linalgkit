
"""
Neural Network Layers
=====================

This module implements the core layers used in neural networks.
Each layer has two main methods:
  - forward(x): Compute output given input
  - backward(grad): Compute gradients given upstream gradient

The key insight is that each layer must:
1. Cache its input during forward pass (needed for gradient computation)
2. Compute gradients for its parameters (W, b)
3. Pass gradients back to the previous layer
"""

import numpy as np
import LinAlgKit as lk


class Layer:
    """Base class for all layers."""
    
    def forward(self, x):
        """Compute output given input x."""
        raise NotImplementedError
    
    def backward(self, grad_output):
        """
        Compute gradients given upstream gradient.
        
        Args:
            grad_output: Gradient flowing from the next layer (dL/dY)
        
        Returns:
            Gradient to pass to the previous layer (dL/dX)
        """
        raise NotImplementedError
    
    def get_params(self):
        """Return list of (param, grad) tuples for optimization."""
        return []


class Dense(Layer):
    """
    Fully Connected (Dense) Layer
    =============================
    
    Computes: Y = X @ W + b
    
    Where:
        - X is the input matrix (batch_size, input_dim)
        - W is the weight matrix (input_dim, output_dim)
        - b is the bias vector (1, output_dim)
        - Y is the output matrix (batch_size, output_dim)
    
    Gradients (using chain rule):
        - dL/dW = X^T @ dL/dY
        - dL/db = sum(dL/dY, axis=0)
        - dL/dX = dL/dY @ W^T
    
    Example:
        >>> layer = Dense(784, 128)
        >>> x = np.random.randn(32, 784)  # batch of 32 samples
        >>> output = layer.forward(x)      # shape: (32, 128)
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initialize the Dense layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features (neurons)
        """
        # Use He initialization - optimal for ReLU activations
        # Scale: sqrt(2 / fan_in) to maintain variance through layers
        self.W = lk.he_normal((input_dim, output_dim))
        self.b = np.zeros((1, output_dim))
        
        # Gradients (will be computed in backward pass)
        self.grad_W = None
        self.grad_b = None
        
        # Cache for backward pass
        self.input = None
    
    def forward(self, x):
        """
        Forward pass: Y = X @ W + b
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Cache input for backward pass
        self.input = x
        
        # Linear transformation: matrix multiply + bias
        return x @ self.W + self.b
    
    def backward(self, grad_output):
        """
        Backward pass: Compute gradients using chain rule.
        
        Math explanation:
        -----------------
        Given: Y = X @ W + b
        
        We need: dL/dX, dL/dW, dL/db (where L is the loss)
        We have: dL/dY (grad_output, from next layer)
        
        Using chain rule:
            dL/dW = dL/dY * dY/dW = X^T @ dL/dY
            dL/db = dL/dY * dY/db = sum(dL/dY, axis=0)
            dL/dX = dL/dY * dY/dX = dL/dY @ W^T
        
        Args:
            grad_output: Gradient from next layer, shape (batch_size, output_dim)
        
        Returns:
            Gradient to previous layer, shape (batch_size, input_dim)
        """
        batch_size = self.input.shape[0]
        
        # Gradient w.r.t weights: X^T @ grad_output
        # Average over batch for stable gradients
        self.grad_W = self.input.T @ grad_output / batch_size
        
        # Gradient w.r.t bias: sum over batch
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True) / batch_size
        
        # Gradient to pass to previous layer: grad_output @ W^T
        grad_input = grad_output @ self.W.T
        
        return grad_input
    
    def get_params(self):
        """Return parameters and their gradients for the optimizer."""
        return [
            (self.W, self.grad_W, 'W'),
            (self.b, self.grad_b, 'b')
        ]


class Dropout(Layer):
    """
    Dropout Layer
    =============
    
    Randomly sets elements to zero with probability p during training.
    This helps prevent overfitting by forcing the network to not rely
    on any single neuron.
    
    During training:
        - Randomly zero out neurons with probability p
        - Scale remaining neurons by 1/(1-p) to maintain expected value
    
    During inference:
        - Do nothing (pass input through unchanged)
    
    Example:
        >>> dropout = Dropout(p=0.5)
        >>> x = np.ones((3, 4))
        >>> dropout.forward(x, training=True)  # ~50% will be zeroed
    """
    
    def __init__(self, p=0.5):
        """
        Initialize Dropout layer.
        
        Args:
            p: Probability of dropping a neuron (0 to 1)
        """
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x, training=True):
        """
        Forward pass with optional dropout.
        
        Args:
            x: Input tensor
            training: If True, apply dropout. If False, pass through.
        
        Returns:
            Output tensor (same shape as input)
        """
        self.training = training
        
        if not training or self.p == 0:
            return x
        
        # Create binary mask: 1 with probability (1-p), 0 with probability p
        self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float64)
        
        # Apply mask and scale to maintain expected value
        # E[output] = E[input] when we scale by 1/(1-p)
        return x * self.mask / (1 - self.p)
    
    def backward(self, grad_output):
        """
        Backward pass: Gradient only flows through non-dropped neurons.
        
        Args:
            grad_output: Gradient from next layer
        
        Returns:
            Gradient to previous layer
        """
        if not self.training or self.p == 0:
            return grad_output
        
        # Same mask, same scaling
        return grad_output * self.mask / (1 - self.p)
