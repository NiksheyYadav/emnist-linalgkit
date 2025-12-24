"""
Loss Functions
==============

Loss functions measure how well the model's predictions match the targets.
The goal of training is to minimize the loss.

Each loss function has:
  - forward(predictions, targets): Compute the loss value
  - backward(): Compute gradient w.r.t predictions

The backward method returns the gradient that starts backpropagation.
This gradient is then passed through each layer in reverse order.
"""

import numpy as np
import LinAlgKit as lk


class Loss:
    """Base class for loss functions."""
    
    def forward(self, predictions, targets):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy Loss (for Classification)
    ========================================
    
    Measures the difference between two probability distributions:
    the predicted distribution and the true distribution (one-hot labels).
    
    Formula:
        L = -sum(targets * log(predictions))
        
    For single correct class c:
        L = -log(prediction_c)
    
    Combined with Softmax:
    -----------------------
    When the predictions come from a softmax layer, the gradient
    simplifies beautifully:
        dL/d(logits) = predictions - targets
    
    This is one of the most elegant results in deep learning!
    
    Parameters:
        epsilon: Small constant to prevent log(0)
    
    Example:
        >>> loss_fn = CrossEntropyLoss()
        >>> probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])  # 2 samples, 3 classes
        >>> targets = np.array([[1, 0, 0], [0, 1, 0]])  # one-hot
        >>> loss = loss_fn.forward(probs, targets)
    """
    
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.predictions = None
        self.targets = None
    
    def forward(self, predictions, targets):
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Probabilities from softmax, shape (batch, classes)
            targets: One-hot encoded labels, shape (batch, classes)
                     OR integer labels, shape (batch,)
        
        Returns:
            Scalar loss value (mean over batch)
        """
        self.predictions = predictions
        
        # Convert integer labels to one-hot if necessary
        if targets.ndim == 1:
            num_classes = predictions.shape[1]
            targets = lk.one_hot(targets.astype(int), num_classes)
        
        self.targets = targets
        
        # Clip predictions to avoid log(0)
        clipped = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        # Cross-entropy: -sum(targets * log(predictions))
        # Average over batch
        loss = -np.sum(targets * np.log(clipped)) / predictions.shape[0]
        
        return loss
    
    def backward(self):
        """
        Compute gradient of cross-entropy loss.
        
        When combined with softmax, gradient is simply:
            dL/d(logits) = predictions - targets
        
        This beautiful simplification comes from calculus:
            dL/d(logits) = dL/d(probs) * d(probs)/d(logits)
                         = (-targets/probs) * (probs * (I - probs))
                         = probs - targets
        
        Returns:
            Gradient w.r.t predictions, shape (batch, classes)
        """
        # This assumes softmax was used before this loss
        # The combined gradient is wonderfully simple
        batch_size = self.predictions.shape[0]
        return (self.predictions - self.targets) / batch_size


class MSELoss(Loss):
    """
    Mean Squared Error Loss (for Regression)
    =========================================
    
    Measures the average squared difference between predictions and targets.
    
    Formula:
        L = (1/n) * sum((predictions - targets)^2)
    
    Gradient:
        dL/d(predictions) = (2/n) * (predictions - targets)
    
    Why MSE:
        - Penalizes large errors more than small ones (squared)
        - Differentiable everywhere
        - Has a single global minimum
    
    Considerations:
        - Sensitive to outliers (use Huber loss if outliers are a problem)
        - Output should not have activation (linear output for regression)
    
    Example:
        >>> loss_fn = MSELoss()
        >>> predictions = np.array([[3.0], [4.5], [2.0]])
        >>> targets = np.array([[3.2], [4.0], [2.5]])
        >>> loss = loss_fn.forward(predictions, targets)
    """
    
    def __init__(self):
        self.predictions = None
        self.targets = None
    
    def forward(self, predictions, targets):
        """
        Compute MSE loss.
        
        Args:
            predictions: Model predictions, any shape
            targets: True values, same shape as predictions
        
        Returns:
            Scalar loss value (mean over all elements and batch)
        """
        self.predictions = predictions
        self.targets = targets
        
        # MSE = mean((pred - target)^2)
        return lk.mse_loss(predictions, targets, reduction='mean')
    
    def backward(self):
        """
        Compute gradient of MSE loss.
        
        dL/d(predictions) = 2 * (predictions - targets) / n
        
        Returns:
            Gradient w.r.t predictions, same shape as predictions
        """
        n = self.predictions.size
        return 2 * (self.predictions - self.targets) / n


class BinaryCrossEntropyLoss(Loss):
    """
    Binary Cross-Entropy Loss
    =========================
    
    For binary classification (two classes: 0 or 1).
    
    Formula:
        L = -mean(targets * log(pred) + (1 - targets) * log(1 - pred))
    
    Gradient:
        dL/d(pred) = (pred - targets) / (pred * (1 - pred))
    
    Combined with Sigmoid:
        dL/d(logits) = predictions - targets
        (Same elegant simplification as softmax + cross-entropy!)
    
    Use when:
        - Binary classification (spam/not spam, cat/dog)
        - Multi-label classification (image has cat AND dog)
    """
    
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.predictions = None
        self.targets = None
    
    def forward(self, predictions, targets):
        """
        Compute binary cross-entropy loss.
        
        Args:
            predictions: Probabilities from sigmoid, shape (batch, 1) or (batch,)
            targets: Binary labels (0 or 1), same shape
        
        Returns:
            Scalar loss value
        """
        self.predictions = predictions
        self.targets = targets
        
        # Clip to avoid log(0)
        pred_clipped = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        # Binary cross-entropy
        loss = -np.mean(
            targets * np.log(pred_clipped) + 
            (1 - targets) * np.log(1 - pred_clipped)
        )
        
        return loss
    
    def backward(self):
        """
        Compute gradient of binary cross-entropy.
        
        When combined with sigmoid:
            dL/d(logits) = predictions - targets
        """
        batch_size = self.predictions.shape[0] if self.predictions.ndim > 1 else 1
        return (self.predictions - self.targets) / batch_size


class HuberLoss(Loss):
    """
    Huber Loss (Smooth L1)
    ======================
    
    Combines the best of MSE and MAE:
        - Quadratic for small errors (like MSE, smooth gradients)
        - Linear for large errors (like MAE, robust to outliers)
    
    Formula:
        L = 0.5 * (y - ŷ)² if |y - ŷ| <= δ
        L = δ * (|y - ŷ| - 0.5 * δ) otherwise
    
    Gradient:
        dL/dŷ = (ŷ - y) if |y - ŷ| <= δ
        dL/dŷ = δ * sign(ŷ - y) otherwise
    
    Parameters:
        delta: Threshold for switching between quadratic and linear
    """
    
    def __init__(self, delta=1.0):
        self.delta = delta
        self.predictions = None
        self.targets = None
        self.diff = None
    
    def forward(self, predictions, targets):
        """Compute Huber loss."""
        self.predictions = predictions
        self.targets = targets
        self.diff = predictions - targets
        
        return lk.huber_loss(predictions, targets, delta=self.delta, reduction='mean')
    
    def backward(self):
        """Compute gradient of Huber loss."""
        abs_diff = np.abs(self.diff)
        n = self.predictions.size
        
        # Gradient is (pred - target) for small errors, delta * sign for large
        grad = np.where(
            abs_diff <= self.delta,
            self.diff,
            self.delta * np.sign(self.diff)
        )
        
        return grad / n
