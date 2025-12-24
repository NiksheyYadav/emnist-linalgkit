"""
Optimizers
==========

Optimizers update the model's parameters (weights and biases) to minimize
the loss function. They use the gradients computed during backpropagation
to determine how to adjust each parameter.

The basic update rule is:
    parameter = parameter - learning_rate * gradient

More sophisticated optimizers like Adam use momentum and adaptive learning
rates to converge faster and more reliably.
"""

import numpy as np


class Optimizer:
    """Base class for optimizers."""
    
    def step(self, layers):
        """Update parameters of all layers."""
        raise NotImplementedError
    
    def zero_grad(self, layers):
        """Reset gradients to zero (optional, for API compatibility)."""
        for layer in layers:
            for param_tuple in layer.get_params():
                if len(param_tuple) >= 2:
                    param, grad, *_ = param_tuple
                    if grad is not None:
                        grad.fill(0)


class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    ===========================
    
    The simplest optimization algorithm.
    
    Update rule:
        θ = θ - lr * ∇L
    
    With momentum (optional):
        v = momentum * v + ∇L
        θ = θ - lr * v
    
    Parameters:
        lr: Learning rate (how big each step is)
        momentum: Momentum coefficient (0 = no momentum)
    
    Why momentum helps:
        - Accelerates convergence in consistent gradient direction
        - Dampens oscillations in inconsistent directions
        - Helps escape shallow local minima
    
    Typical values:
        - lr: 0.01 to 0.1
        - momentum: 0.9 to 0.99
    
    Example:
        >>> optimizer = SGD(lr=0.01, momentum=0.9)
        >>> for epoch in range(100):
        ...     loss = model.train_step(x, y, loss_fn, optimizer)
    """
    
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # Store velocity for each parameter
    
    def step(self, layers):
        """
        Update all parameters using SGD (with optional momentum).
        
        Args:
            layers: List of layers with get_params() method
        """
        for layer_idx, layer in enumerate(layers):
            for param_tuple in layer.get_params():
                if len(param_tuple) < 2:
                    continue
                    
                param, grad, name = param_tuple
                if grad is None:
                    continue
                
                # Create unique key for this parameter
                key = (layer_idx, name)
                
                if self.momentum > 0:
                    # Initialize velocity if first time
                    if key not in self.velocity:
                        self.velocity[key] = np.zeros_like(param)
                    
                    # Update velocity: v = momentum * v + grad
                    self.velocity[key] = self.momentum * self.velocity[key] + grad
                    
                    # Update parameter: θ -= lr * v
                    param -= self.lr * self.velocity[key]
                else:
                    # Simple gradient descent: θ -= lr * grad
                    param -= self.lr * grad


class Adam(Optimizer):
    """
    Adam Optimizer (Adaptive Moment Estimation)
    ============================================
    
    One of the most popular optimizers in deep learning. Combines:
        - Momentum (first moment estimate)
        - RMSprop (second moment estimate)
    
    Update rules:
        m = β₁ * m + (1 - β₁) * g           # First moment (momentum)
        v = β₂ * v + (1 - β₂) * g²          # Second moment (RMSprop)
        m̂ = m / (1 - β₁ᵗ)                   # Bias correction
        v̂ = v / (1 - β₂ᵗ)                   # Bias correction
        θ = θ - lr * m̂ / (√v̂ + ε)          # Update
    
    Parameters:
        lr: Learning rate
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        epsilon: Small constant for numerical stability
    
    Why Adam is great:
        - Works well with default parameters
        - Adapts learning rate per-parameter
        - Handles sparse gradients well
        - Fast convergence
    
    Typical values:
        - lr: 0.001 (the paper's recommended default)
        - beta1: 0.9
        - beta2: 0.999
    
    Example:
        >>> optimizer = Adam(lr=0.001)
        >>> # Works great out of the box!
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # State for each parameter
        self.m = {}  # First moment (mean of gradients)
        self.v = {}  # Second moment (mean of squared gradients)
        self.t = 0   # Time step (for bias correction)
    
    def step(self, layers):
        """
        Update all parameters using Adam.
        
        Args:
            layers: List of layers with get_params() method
        """
        self.t += 1  # Increment time step
        
        for layer_idx, layer in enumerate(layers):
            for param_tuple in layer.get_params():
                if len(param_tuple) < 2:
                    continue
                
                param, grad, name = param_tuple
                if grad is None:
                    continue
                
                # Create unique key for this parameter
                key = (layer_idx, name)
                
                # Initialize moments if first time
                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)
                
                # Update biased first moment estimate
                # m = β₁ * m + (1 - β₁) * g
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                # v = β₂ * v + (1 - β₂) * g²
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
                
                # Compute bias-corrected first moment estimate
                # m̂ = m / (1 - β₁ᵗ)
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                
                # Compute bias-corrected second raw moment estimate
                # v̂ = v / (1 - β₂ᵗ)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                # θ = θ - lr * m̂ / (√v̂ + ε)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(Optimizer):
    """
    RMSprop Optimizer
    =================
    
    Divides the learning rate by a running average of recent gradient
    magnitudes. This gives each parameter its own adaptive learning rate.
    
    Update rules:
        v = ρ * v + (1 - ρ) * g²    # Running average of squared gradients
        θ = θ - lr * g / √(v + ε)   # Update with adaptive rate
    
    Parameters:
        lr: Learning rate
        rho: Decay rate for moving average (default: 0.99)
        epsilon: Small constant for numerical stability
    
    Why RMSprop:
        - Handles different gradient magnitudes per parameter
        - Good for recurrent neural networks
        - Precursor to Adam
    """
    
    def __init__(self, lr=0.01, rho=0.99, epsilon=1e-8):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.v = {}  # Running average of squared gradients
    
    def step(self, layers):
        """Update all parameters using RMSprop."""
        for layer_idx, layer in enumerate(layers):
            for param_tuple in layer.get_params():
                if len(param_tuple) < 2:
                    continue
                
                param, grad, name = param_tuple
                if grad is None:
                    continue
                
                key = (layer_idx, name)
                
                # Initialize if first time
                if key not in self.v:
                    self.v[key] = np.zeros_like(param)
                
                # Update running average of squared gradients
                self.v[key] = self.rho * self.v[key] + (1 - self.rho) * (grad ** 2)
                
                # Update parameters with adaptive learning rate
                param -= self.lr * grad / (np.sqrt(self.v[key]) + self.epsilon)
