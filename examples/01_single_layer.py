"""
Example 1: Understanding a Single Dense Layer
==============================================

This example breaks down exactly what happens in a single layer
of a neural network. No magic - just matrix multiplication.

What you'll learn:
1. How weights convert input features to output features
2. Why weight initialization matters
3. How forward and backward passes work

Run this script to see step-by-step computation!
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import LinAlgKit as lk
from nn.layers import Dense


def main():
    print("=" * 60)
    print("Understanding a Single Dense Layer")
    print("=" * 60)
    
    # Create a small layer for demonstration
    # 4 input features → 3 output features
    layer = Dense(4, 3)
    
    print("\n📐 Layer Setup")
    print("-" * 40)
    print(f"Input dimension:  4")
    print(f"Output dimension: 3")
    print(f"Weight matrix shape: {layer.W.shape}")
    print(f"Bias vector shape:   {layer.b.shape}")
    
    # Show the weights
    print("\n🔢 Initial Weights (He initialization):")
    print(f"W =\n{layer.W}")
    print(f"\nb = {layer.b}")
    
    # Create a simple input
    # 2 samples, 4 features each
    x = np.array([
        [1.0, 2.0, 3.0, 4.0],   # Sample 1
        [0.5, 1.0, 1.5, 2.0],   # Sample 2
    ])
    
    print("\n" + "=" * 60)
    print("FORWARD PASS: Y = X @ W + b")
    print("=" * 60)
    
    print("\n📥 Input X (2 samples, 4 features):")
    print(x)
    
    # Manual computation
    print("\n🧮 Manual Computation:")
    manual_output = x @ layer.W + layer.b
    print(f"X @ W =\n{x @ layer.W}")
    print(f"\n+ b =\n{layer.b}")
    print(f"\n= Y =\n{manual_output}")
    
    # Using layer.forward()
    output = layer.forward(x)
    print(f"\n✅ layer.forward(x) =\n{output}")
    print(f"\nOutputs match: {np.allclose(output, manual_output)}")
    
    print("\n" + "=" * 60)
    print("BACKWARD PASS: Computing Gradients")
    print("=" * 60)
    
    # Simulate gradient from next layer (loss function)
    # This would come from the loss function in real training
    grad_output = np.array([
        [0.1, -0.2, 0.3],   # Gradient for sample 1
        [0.05, -0.1, 0.15], # Gradient for sample 2
    ])
    
    print("\n📤 Gradient from next layer (dL/dY):")
    print(grad_output)
    
    print("\n🧮 Computing gradients using chain rule:")
    print("\n1. dL/dW = X^T @ dL/dY (how weights affect loss)")
    grad_W_manual = x.T @ grad_output / 2  # /2 for batch average
    print(f"   = {x.T.shape} @ {grad_output.shape}")
    print(f"   =\n{grad_W_manual}")
    
    print("\n2. dL/db = sum(dL/dY) (how bias affects loss)")
    grad_b_manual = np.sum(grad_output, axis=0, keepdims=True) / 2
    print(f"   = {grad_b_manual}")
    
    print("\n3. dL/dX = dL/dY @ W^T (to pass to previous layer)")
    grad_input_manual = grad_output @ layer.W.T
    print(f"   = {grad_output.shape} @ {layer.W.T.shape}")
    print(f"   =\n{grad_input_manual}")
    
    # Using layer.backward()
    grad_input = layer.backward(grad_output)
    
    print("\n✅ Using layer.backward():")
    print(f"grad_W matches: {np.allclose(layer.grad_W, grad_W_manual)}")
    print(f"grad_b matches: {np.allclose(layer.grad_b, grad_b_manual)}")
    print(f"grad_input matches: {np.allclose(grad_input, grad_input_manual)}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. Forward pass: Y = X @ W + b
       - Each output is a weighted sum of inputs plus bias
       - W controls the "importance" of each input feature
    
    2. Backward pass uses chain rule:
       - dL/dW: How changing weights affects the loss
       - dL/db: How changing bias affects the loss  
       - dL/dX: Gradient to pass to previous layer
    
    3. Key shapes:
       - X: (batch, input_features)
       - W: (input_features, output_features)
       - Y: (batch, output_features)
       - All gradients match their parameter shapes!
    """)


if __name__ == '__main__':
    main()
