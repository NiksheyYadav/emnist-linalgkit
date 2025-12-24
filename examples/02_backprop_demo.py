"""
Example 2: Visualizing Backpropagation
======================================

This example shows how gradients flow backwards through a network.
We'll trace the chain rule through multiple layers step by step.

What you'll learn:
1. How the chain rule connects layer gradients
2. Why we traverse layers in reverse order
3. How each layer contributes to parameter updates

Run this to see gradient flow in action!
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from nn.layers import Dense
from nn.activations import ReLU, Softmax
from nn.losses import CrossEntropyLoss


def main():
    print("=" * 60)
    print("Visualizing Backpropagation")
    print("=" * 60)
    
    # Build a small network: 4 → 3 → 2
    layer1 = Dense(4, 3)
    relu1 = ReLU()
    layer2 = Dense(3, 2)
    softmax = Softmax()
    loss_fn = CrossEntropyLoss()
    
    print("\n📐 Network Architecture:")
    print("-" * 40)
    print("Input (4) → Dense(4,3) → ReLU → Dense(3,2) → Softmax → Output (2)")
    print()
    
    # Input and target
    x = np.array([[1.0, 2.0, 3.0, 4.0]])  # 1 sample
    y = np.array([[1, 0]])  # Target: class 0
    
    print("📥 Input:", x.flatten())
    print("🎯 Target: Class 0 (one-hot: [1, 0])")
    
    print("\n" + "=" * 60)
    print("FORWARD PASS")
    print("=" * 60)
    
    # Layer 1: Dense
    z1 = layer1.forward(x)
    print(f"\n1️⃣ Dense(4→3): z1 = x @ W1 + b1")
    print(f"   z1 = {z1.flatten()}")
    
    # ReLU
    a1 = relu1.forward(z1)
    print(f"\n2️⃣ ReLU: a1 = max(0, z1)")
    print(f"   a1 = {a1.flatten()}")
    
    # Layer 2: Dense
    z2 = layer2.forward(a1)
    print(f"\n3️⃣ Dense(3→2): z2 = a1 @ W2 + b2")
    print(f"   z2 = {z2.flatten()}")
    
    # Softmax
    a2 = softmax.forward(z2)
    print(f"\n4️⃣ Softmax: a2 = exp(z2) / sum(exp(z2))")
    print(f"   a2 = {a2.flatten()} (probabilities, sum = {a2.sum():.4f})")
    
    # Loss
    loss = loss_fn.forward(a2, y)
    print(f"\n5️⃣ Cross-Entropy Loss: L = -sum(y * log(a2))")
    print(f"   L = {loss:.4f}")
    
    print("\n" + "=" * 60)
    print("BACKWARD PASS (Gradient Flow)")
    print("=" * 60)
    
    print("\n🔄 Applying chain rule in REVERSE order...")
    
    # Gradient from loss
    grad_a2 = loss_fn.backward()
    print(f"\n1️⃣ dL/da2 = a2 - y (for softmax + cross-entropy)")
    print(f"   = {a2.flatten()} - {y.flatten()}")
    print(f"   = {grad_a2.flatten()}")
    
    # Softmax backward (actually handled in loss for numerical stability)
    grad_z2 = softmax.backward(grad_a2)
    print(f"\n2️⃣ dL/dz2 = dL/da2 @ Jacobian(softmax)")
    print(f"   = {grad_z2.flatten()}")
    
    # Layer 2 backward
    grad_a1 = layer2.backward(grad_z2)
    print(f"\n3️⃣ Dense(3→2) backward:")
    print(f"   dL/dW2 = a1^T @ dL/dz2")
    print(f"   dL/da1 = dL/dz2 @ W2^T = {grad_a1.flatten()}")
    
    # ReLU backward
    grad_z1 = relu1.backward(grad_a1)
    print(f"\n4️⃣ ReLU backward: dL/dz1 = dL/da1 * (z1 > 0)")
    print(f"   Mask (z1 > 0): {(layer1.forward(x) > 0).flatten().astype(int)}")
    print(f"   dL/dz1 = {grad_z1.flatten()}")
    
    # Layer 1 backward
    grad_x = layer1.backward(grad_z1)
    print(f"\n5️⃣ Dense(4→3) backward:")
    print(f"   dL/dW1 = x^T @ dL/dz1")
    print(f"   dL/dx = dL/dz1 @ W1^T = {grad_x.flatten()}")
    
    print("\n" + "=" * 60)
    print("GRADIENT SUMMARY")
    print("=" * 60)
    
    print("\n📊 Gradient magnitudes at each layer:")
    print(f"   |dL/dW2| = {np.mean(np.abs(layer2.grad_W)):.6f}")
    print(f"   |dL/dW1| = {np.mean(np.abs(layer1.grad_W)):.6f}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. CHAIN RULE: Each layer's gradient depends on the next layer's
       dL/dz1 = dL/dz2 * dz2/da1 * da1/dz1
    
    2. REVERSE ORDER: We start from the loss and work backwards
       because each gradient depends on the ones computed after it
    
    3. GRADIENT FLOW:
       Loss → Softmax → Dense2 → ReLU → Dense1 → Input
       Each layer receives gradient from the right, passes it left
    
    4. WEIGHT GRADIENTS: Each Dense layer computes:
       - dL/dW: To update its own weights
       - dL/dx: To send to the previous layer
    
    5. ReLU's ROLE: Acts as a gate
       - Passes gradient where input was positive
       - Blocks gradient where input was negative
    """)


if __name__ == '__main__':
    main()
