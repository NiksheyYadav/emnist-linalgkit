"""
Sequential Model
=================

A Sequential model is a linear stack of layers. Data flows through each
layer in order during the forward pass, and gradients flow backwards
during backpropagation.

This is the simplest and most common neural network architecture:
    Input → Layer1 → Layer2 → ... → LayerN → Output

Example:
    model = Sequential([
        Dense(784, 256), ReLU(),
        Dense(256, 128), ReLU(),
        Dense(128, 10), Softmax()
    ])
"""

import numpy as np
from typing import List


class Sequential:
    """
    Sequential Neural Network Model
    ================================
    
    A container that chains layers together. During forward pass,
    each layer's output becomes the next layer's input.
    
    How it works:
    -------------
    Forward pass:
        x → Dense(784,256) → ReLU → Dense(256,128) → ReLU → Dense(128,10) → Softmax → output
    
    Backward pass (gradients flow in reverse):
        ∇output ← ∇Softmax ← ∇Dense ← ∇ReLU ← ∇Dense ← ∇ReLU ← ∇Dense
    
    Training loop:
        1. Forward pass: compute predictions
        2. Compute loss: how wrong are we?
        3. Backward pass: compute gradients
        4. Optimizer step: update weights
    
    Example:
        >>> from nn import Sequential, Dense, ReLU, Softmax, CrossEntropyLoss, Adam
        >>> 
        >>> model = Sequential([
        ...     Dense(784, 128), ReLU(),
        ...     Dense(128, 10), Softmax()
        ... ])
        >>> 
        >>> loss_fn = CrossEntropyLoss()
        >>> optimizer = Adam(lr=0.001)
        >>> 
        >>> # Training step
        >>> loss = model.train_step(x_batch, y_batch, loss_fn, optimizer)
    """
    
    def __init__(self, layers: List = None):
        """
        Initialize Sequential model.
        
        Args:
            layers: List of layer objects (Dense, ReLU, etc.)
        """
        self.layers = layers if layers is not None else []
        self.training = True
    
    def add(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)
    
    def forward(self, x, training=True):
        """
        Forward pass through all layers.
        
        Args:
            x: Input tensor
            training: Whether in training mode (affects Dropout, etc.)
        
        Returns:
            Output tensor after passing through all layers
        """
        self.training = training
        
        for layer in self.layers:
            # Handle layers that need training flag (like Dropout)
            if hasattr(layer, 'training'):
                x = layer.forward(x, training)
            else:
                x = layer.forward(x)
        
        return x
    
    def backward(self, grad):
        """
        Backward pass through all layers (in reverse).
        
        This implements backpropagation:
        1. Start with gradient from loss function
        2. Pass gradient through each layer in reverse
        3. Each layer computes its parameter gradients
        4. Each layer passes gradient to previous layer
        
        Args:
            grad: Gradient from loss function w.r.t model output
        
        Returns:
            Gradient w.r.t input (usually not needed)
        """
        # Traverse layers in reverse order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        return grad
    
    def train_step(self, x, y, loss_fn, optimizer):
        """
        Perform one training step.
        
        This is the core of neural network training:
        1. Forward pass: compute predictions
        2. Loss: measure how wrong we are
        3. Backward pass: compute gradients
        4. Optimizer: update parameters
        
        Args:
            x: Input batch
            y: Target batch
            loss_fn: Loss function object
            optimizer: Optimizer object
        
        Returns:
            Loss value for this batch
        """
        # 1. Forward pass
        predictions = self.forward(x, training=True)
        
        # 2. Compute loss
        loss = loss_fn.forward(predictions, y)
        
        # 3. Backward pass
        grad = loss_fn.backward()
        self.backward(grad)
        
        # 4. Update parameters
        optimizer.step(self.layers)
        
        return loss
    
    def predict(self, x):
        """
        Make predictions (inference mode, no dropout).
        
        Args:
            x: Input tensor
        
        Returns:
            Predictions
        """
        return self.forward(x, training=False)
    
    def evaluate(self, x, y, loss_fn):
        """
        Evaluate model on data.
        
        Args:
            x: Input data
            y: Targets
            loss_fn: Loss function
        
        Returns:
            Tuple of (loss, accuracy)
        """
        predictions = self.predict(x)
        loss = loss_fn.forward(predictions, y)
        
        # Calculate accuracy for classification
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            pred_classes = np.argmax(predictions, axis=1)
            if y.ndim > 1:
                true_classes = np.argmax(y, axis=1)
            else:
                true_classes = y
            accuracy = np.mean(pred_classes == true_classes)
        else:
            # For regression, use R² or just return loss
            accuracy = 0.0
        
        return loss, accuracy
    
    def summary(self):
        """Print a summary of the model architecture."""
        print("=" * 60)
        print("Model Summary")
        print("=" * 60)
        
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            params = 0
            shape_info = ""
            
            # Count parameters for layers with weights
            if hasattr(layer, 'W'):
                params += layer.W.size
                shape_info = f"W: {layer.W.shape}"
            if hasattr(layer, 'b'):
                params += layer.b.size
                if shape_info:
                    shape_info += f", b: {layer.b.shape}"
            
            total_params += params
            
            print(f"Layer {i}: {layer_name:15} | Params: {params:8,} | {shape_info}")
        
        print("=" * 60)
        print(f"Total Parameters: {total_params:,}")
        print("=" * 60)
    
    def get_weights(self):
        """Get all weights and biases as a list of arrays."""
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'W'):
                weights.append(layer.W.copy())
            if hasattr(layer, 'b'):
                weights.append(layer.b.copy())
        return weights
    
    def set_weights(self, weights):
        """Set all weights and biases from a list of arrays."""
        idx = 0
        for layer in self.layers:
            if hasattr(layer, 'W'):
                layer.W = weights[idx].copy()
                idx += 1
            if hasattr(layer, 'b'):
                layer.b = weights[idx].copy()
                idx += 1
    
    def save(self, filepath):
        """Save model weights to file."""
        weights = self.get_weights()
        np.savez(filepath, *weights)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights from file."""
        data = np.load(filepath)
        weights = [data[key] for key in data.files]
        self.set_weights(weights)
        print(f"Model loaded from {filepath}")
