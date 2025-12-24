"""
Example 3: Training on MNIST Step by Step
=========================================

This example shows the complete training process on MNIST,
with detailed explanations at each step.

What you'll learn:
1. How to structure a training loop
2. What happens in each training step
3. How the model improves over time

Run this for a complete training walkthrough!
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from nn import Sequential, Dense, ReLU, Softmax, Dropout
from nn import CrossEntropyLoss, Adam
from data.mnist import load_mnist, DataLoader


def main():
    print("=" * 60)
    print("MNIST Training Walkthrough")
    print("=" * 60)
    
    # ========================================
    # STEP 1: Load and Understand the Data
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training samples: {x_train.shape[0]}")
    print(f"   Test samples:     {x_test.shape[0]}")
    print(f"   Image size:       28x28 = 784 pixels")
    print(f"   Classes:          10 (digits 0-9)")
    
    print(f"\n📈 Data ranges:")
    print(f"   Pixel values: [{x_train.min():.2f}, {x_train.max():.2f}] (normalized)")
    print(f"   Labels: one-hot encoded, e.g., digit 3 → {np.eye(10)[3].astype(int)}")
    
    # Show a sample
    sample_idx = 0
    print(f"\n📷 Sample image (index {sample_idx}):")
    print(f"   Label: {np.argmax(y_train[sample_idx])}")
    print(f"   Pixels (first 20): {x_train[sample_idx][:20].round(2)}")
    
    # ========================================
    # STEP 2: Build the Neural Network
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 2: Building the Neural Network")
    print("=" * 60)
    
    model = Sequential([
        Dense(784, 128),    # 784 inputs → 128 hidden neurons
        ReLU(),             # Non-linearity
        Dropout(0.2),       # Regularization
        Dense(128, 64),     # 128 → 64 hidden neurons
        ReLU(),             
        Dropout(0.2),
        Dense(64, 10),      # 64 → 10 output classes
        Softmax()           # Convert to probabilities
    ])
    
    print("\n🏗️ Network Architecture:")
    model.summary()
    
    # ========================================
    # STEP 3: Define Loss and Optimizer
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 3: Loss Function and Optimizer")
    print("=" * 60)
    
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(lr=0.001)
    
    print("""
    📉 Loss Function: Cross-Entropy
       - Measures how different predicted probabilities are from true labels
       - L = -sum(target * log(prediction))
       - Penalizes confident wrong predictions heavily
    
    ⚡ Optimizer: Adam (lr=0.001)
       - Adaptive learning rates per parameter
       - Uses momentum and RMSprop ideas
       - Works well out of the box
    """)
    
    # ========================================
    # STEP 4: Create Data Loader
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 4: Mini-Batch Training Setup")
    print("=" * 60)
    
    batch_size = 64
    train_loader = DataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)
    
    print(f"""
    📦 Mini-Batch Training:
       - Batch size: {batch_size}
       - Batches per epoch: {len(train_loader)}
       - Why mini-batches?
         • Faster than full batch (don't need all data for each update)
         • More stable than single samples (averaged gradients)
         • Adds regularization through noise
    """)
    
    # ========================================
    # STEP 5: Training Loop
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 5: Training Loop")
    print("=" * 60)
    
    epochs = 5
    
    print(f"\n🏃 Training for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        epoch_losses = []
        correct = 0
        total = 0
        
        # Training
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # This single line does:
            # 1. Forward pass
            # 2. Compute loss
            # 3. Backward pass (compute gradients)
            # 4. Update weights
            loss = model.train_step(x_batch, y_batch, loss_fn, optimizer)
            
            epoch_losses.append(loss)
            
            # Track accuracy
            preds = np.argmax(model.predict(x_batch), axis=1)
            labels = np.argmax(y_batch, axis=1)
            correct += np.sum(preds == labels)
            total += len(y_batch)
            
            # Print progress every 200 batches
            if batch_idx % 200 == 0:
                print(f"   Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss:.4f}")
        
        # Epoch summary
        train_loss = np.mean(epoch_losses)
        train_acc = correct / total
        
        # Validation
        val_loss, val_acc = model.evaluate(x_test, y_test, loss_fn)
        
        print(f"\n   📊 Epoch {epoch+1} Complete:")
        print(f"      Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"      Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print()
    
    # ========================================
    # STEP 6: Final Evaluation
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 6: Final Evaluation")
    print("=" * 60)
    
    final_loss, final_acc = model.evaluate(x_test, y_test, loss_fn)
    
    print(f"""
    🎯 Final Test Results:
       Loss:     {final_loss:.4f}
       Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)
    
    📊 What we achieved:
       - Started with random weights (~10% accuracy, random guessing)
       - After {epochs} epochs: {final_acc*100:.1f}% accuracy
       - The network learned to recognize handwritten digits!
    """)
    
    # Show some predictions
    print("\n🔍 Sample Predictions:")
    print("-" * 40)
    
    sample_x = x_test[:5]
    sample_y = y_test[:5]
    predictions = model.predict(sample_x)
    
    for i in range(5):
        true_label = np.argmax(sample_y[i])
        pred_label = np.argmax(predictions[i])
        confidence = predictions[i][pred_label] * 100
        
        status = "✅" if true_label == pred_label else "❌"
        print(f"   Sample {i+1}: True={true_label}, Pred={pred_label} "
              f"(conf: {confidence:.1f}%) {status}")
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
