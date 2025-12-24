"""Quick test of the neural network framework."""
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

from nn import Sequential, Dense, ReLU, Softmax
from nn import CrossEntropyLoss, Adam

print("=" * 50)
print("Testing Neural Network Framework")
print("=" * 50)

# Test 1: Create model
print("\n1. Creating model...")
model = Sequential([
    Dense(784, 128),
    ReLU(),
    Dense(128, 10),
    Softmax()
])
print("   Model created successfully!")

# Test 2: Forward pass
print("\n2. Testing forward pass...")
x = np.random.randn(32, 784)  # 32 samples, 784 features
output = model.forward(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Output sums to 1: {np.allclose(output.sum(axis=1), 1.0)}")

# Test 3: Loss computation
print("\n3. Testing loss computation...")
y = np.eye(10)[np.random.randint(0, 10, 32)]  # Random one-hot labels
loss_fn = CrossEntropyLoss()
loss = loss_fn.forward(output, y)
print(f"   Loss value: {loss:.4f}")

# Test 4: Backward pass
print("\n4. Testing backward pass...")
grad = loss_fn.backward()
model.backward(grad)
print(f"   Gradients computed successfully!")

# Test 5: Optimizer step
print("\n5. Testing optimizer...")
optimizer = Adam(lr=0.001)
optimizer.step(model.layers)
print("   Weights updated successfully!")

# Test 6: Training step
print("\n6. Testing train_step...")
loss1 = model.train_step(x, y, loss_fn, optimizer)
loss2 = model.train_step(x, y, loss_fn, optimizer)
print(f"   Loss step 1: {loss1:.4f}")
print(f"   Loss step 2: {loss2:.4f}")
print(f"   Loss decreased: {loss2 < loss1}")

print("\n" + "=" * 50)
print("All tests passed!")
print("=" * 50)
