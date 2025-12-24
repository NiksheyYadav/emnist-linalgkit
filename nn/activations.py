"""
Activation Functions
====================

Activation functions introduce non-linearity into neural networks.
Without them, stacking linear layers would just produce another linear
transformation (A @ B @ C = D, where D is still linear).

Each activation has:
  - forward(x): Apply the activation function
  - backward(grad): Compute gradient for backpropagation

Key insight: Most activation gradients are element-wise operations,
making them computationally efficient.
"""

import numpy as np
import LinAlgKit as lk


class Activation:
    """Base class for activation functions."""
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def get_params(self):
        """Activations have no learnable parameters."""
        return []


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU)
    ============================
    
    f(x) = max(0, x)
    
    Gradient:
        f'(x) = 1 if x > 0, else 0
    
    Why ReLU is popular:
        - Simple and fast to compute
        - Doesn't saturate for positive values (no vanishing gradient)
        - Sparse activation (many zeros = efficient)
    
    Potential issue: "Dying ReLU" - neurons can get stuck at 0
    Solution: Use Leaky ReLU or proper initialization
    
    Example:
        >>> relu = ReLU()
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> relu.forward(x)
        array([0, 0, 0, 1, 2])
    """
    
    def __init__(self):
        self.input = None
    
    def forward(self, x):
        """Apply ReLU: max(0, x)"""
        self.input = x
        return lk.relu(x)
    
    def backward(self, grad_output):
        """
        Gradient of ReLU.
        
        The gradient is 1 where input > 0, and 0 elsewhere.
        We use a mask created from the original input.
        """
        # Gradient is 1 where x > 0, 0 elsewhere
        grad_mask = (self.input > 0).astype(np.float64)
        return grad_output * grad_mask


class LeakyReLU(Activation):
    """
    Leaky ReLU
    ==========
    
    f(x) = x if x > 0, else alpha * x
    
    Gradient:
        f'(x) = 1 if x > 0, else alpha
    
    Why Leaky ReLU:
        - Fixes the "dying ReLU" problem
        - Allows small gradient for negative values
        - Still computationally efficient
    
    Typical alpha values: 0.01 to 0.3
    """
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.input = None
    
    def forward(self, x):
        """Apply Leaky ReLU."""
        self.input = x
        return lk.leaky_relu(x, self.alpha)
    
    def backward(self, grad_output):
        """Gradient: 1 if x > 0, alpha if x <= 0."""
        grad_mask = np.where(self.input > 0, 1.0, self.alpha)
        return grad_output * grad_mask


class Sigmoid(Activation):
    """
    Sigmoid Activation
    ==================
    
    f(x) = 1 / (1 + exp(-x))
    
    Output range: (0, 1) - useful for probabilities
    
    Gradient:
        f'(x) = f(x) * (1 - f(x)) = sigmoid(x) * (1 - sigmoid(x))
    
    Note: The gradient can be computed directly from the output!
    This is a nice property that saves computation.
    
    Issues:
        - Vanishing gradient for very large/small x
        - Output not centered around 0
    
    Best used for: Binary classification output layer
    """
    
    def __init__(self):
        self.output = None
    
    def forward(self, x):
        """Apply sigmoid: 1 / (1 + exp(-x))"""
        self.output = lk.sigmoid(x)
        return self.output
    
    def backward(self, grad_output):
        """
        Gradient of sigmoid: σ(x) * (1 - σ(x))
        
        Beautiful property: gradient only depends on output, not input!
        """
        return grad_output * self.output * (1 - self.output)


class Tanh(Activation):
    """
    Hyperbolic Tangent (Tanh)
    =========================
    
    f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Output range: (-1, 1) - centered around 0
    
    Gradient:
        f'(x) = 1 - tanh(x)^2
    
    Advantages over sigmoid:
        - Output centered around 0 (better for next layer)
        - Stronger gradients (derivative up to 1 vs 0.25)
    
    Disadvantages:
        - Still saturates for large values
    """
    
    def __init__(self):
        self.output = None
    
    def forward(self, x):
        """Apply tanh."""
        self.output = lk.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        """Gradient: 1 - tanh(x)^2"""
        return grad_output * (1 - self.output ** 2)


class Softmax(Activation):
    """
    Softmax Activation
    ==================
    
    f(x)_i = exp(x_i) / sum(exp(x_j)) for all j
    
    Converts a vector of real numbers into a probability distribution.
    All outputs are positive and sum to 1.
    
    Numerical stability:
        We subtract max(x) before computing exp to avoid overflow.
        This doesn't change the result: exp(x-c)/sum(exp(x-c)) = exp(x)/sum(exp(x))
    
    Gradient:
        The Jacobian of softmax is complex, but when combined with
        cross-entropy loss, the gradient simplifies to: (predictions - targets)
    
    Best used for: Multi-class classification output layer
    
    Example:
        >>> softmax = Softmax()
        >>> logits = np.array([[1, 2, 3], [1, 1, 1]])
        >>> softmax.forward(logits)
        array([[0.09, 0.24, 0.67],   # sums to 1
               [0.33, 0.33, 0.33]])  # sums to 1
    """
    
    def __init__(self):
        self.output = None
    
    def forward(self, x):
        """
        Apply softmax with numerical stability.
        
        Uses LinAlgKit's optimized softmax implementation.
        """
        self.output = lk.softmax(x, axis=-1)
        return self.output
    
    def backward(self, grad_output):
        """
        Backward pass for softmax.
        
        Note: When used with CrossEntropyLoss, the combined gradient
        is simply (predictions - targets), which is computed in the
        loss function. This backward is for standalone use.
        
        Full Jacobian computation (for reference):
            d(softmax_i) / d(x_j) = softmax_i * (δ_ij - softmax_j)
        """
        # For each sample in the batch
        batch_size = self.output.shape[0]
        grad_input = np.zeros_like(self.output)
        
        for i in range(batch_size):
            s = self.output[i].reshape(-1, 1)  # Column vector
            # Jacobian: diag(s) - s @ s^T
            jacobian = np.diagflat(s) - s @ s.T
            grad_input[i] = jacobian @ grad_output[i]
        
        return grad_input


class GELU(Activation):
    """
    Gaussian Error Linear Unit (GELU)
    ==================================
    
    f(x) = x * Φ(x)
    
    where Φ is the cumulative distribution function of the standard
    normal distribution.
    
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    Why GELU:
        - State-of-the-art activation for transformers
        - Smooth, non-monotonic (can decrease for some inputs)
        - Used in GPT, BERT, and many modern models
    """
    
    def __init__(self):
        self.input = None
    
    def forward(self, x):
        """Apply GELU using LinAlgKit's implementation."""
        self.input = x
        return lk.gelu(x)
    
    def backward(self, grad_output):
        """
        Gradient of GELU (using approximation).
        
        f'(x) ≈ 0.5 * (1 + tanh(a)) + 0.5 * x * sech²(a) * (sqrt(2/π) * (1 + 3*0.044715*x²))
        where a = sqrt(2/π) * (x + 0.044715 * x³)
        """
        x = self.input
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        a = sqrt_2_over_pi * (x + 0.044715 * x**3)
        tanh_a = np.tanh(a)
        sech2_a = 1 - tanh_a**2
        
        grad = 0.5 * (1 + tanh_a) + 0.5 * x * sech2_a * sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)
        
        return grad_output * grad
