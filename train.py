"""
Train a Neural Network on MNIST from Scratch
=============================================

This script trains a fully connected neural network on the MNIST dataset
using only our custom-built framework (built on LinAlgKit).

No PyTorch. No TensorFlow. Just pure understanding.

Usage:
    python train.py                    # Train with defaults
    python train.py --epochs 20        # Train for 20 epochs
    python train.py --lr 0.001         # Use learning rate 0.001
    python train.py --batch-size 128   # Use batch size 128

What happens during training:
1. Load MNIST data (60,000 training images, 10,000 test images)
2. Build a neural network with Dense layers and ReLU activations
3. For each epoch:
   - Iterate through mini-batches
   - Forward pass: compute predictions
   - Compute loss (cross-entropy)
   - Backward pass: compute gradients
   - Update weights with optimizer
4. Evaluate on test set
5. Save model and visualizations
"""

import argparse
import numpy as np
import time
from pathlib import Path

# Our custom neural network framework
from nn import Sequential, Dense, ReLU, Softmax, Dropout
from nn import CrossEntropyLoss, Adam, SGD
from data.mnist import load_mnist, DataLoader
from utils.visualization import (
    plot_training_curves,
    visualize_predictions,
    plot_confusion_matrix,
    plot_weight_histograms,
    print_training_progress
)


def build_model(input_dim=784, hidden_dims=[256, 128], num_classes=10, dropout=0.2):
    """
    Build a fully connected neural network.
    
    Architecture:
        Input (784) → Dense → ReLU → Dropout → Dense → ReLU → Dropout → Dense → Softmax
    
    Args:
        input_dim: Input feature dimension (784 for MNIST)
        hidden_dims: List of hidden layer sizes
        num_classes: Number of output classes (10 for digits)
        dropout: Dropout probability
    
    Returns:
        Sequential model
    """
    layers = []
    
    # Input layer
    prev_dim = input_dim
    
    # Hidden layers
    for hidden_dim in hidden_dims:
        layers.append(Dense(prev_dim, hidden_dim))
        layers.append(ReLU())
        if dropout > 0:
            layers.append(Dropout(dropout))
        prev_dim = hidden_dim
    
    # Output layer
    layers.append(Dense(prev_dim, num_classes))
    layers.append(Softmax())
    
    model = Sequential(layers)
    return model


def train(model, train_loader, val_data, loss_fn, optimizer, epochs):
    """
    Train the model.
    
    This is the heart of deep learning:
    1. Forward pass - compute predictions
    2. Compute loss - measure error
    3. Backward pass - compute gradients
    4. Optimizer step - update weights
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_data: Tuple of (x_val, y_val) for validation
        loss_fn: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
    
    Returns:
        History dict with training curves
    """
    x_val, y_val = val_data
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\n" + "=" * 60)
    print("Training Started")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # Single training step
            loss = model.train_step(x_batch, y_batch, loss_fn, optimizer)
            epoch_losses.append(loss)
            
            # Track accuracy
            predictions = model.predict(x_batch)
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_batch, axis=1)
            epoch_correct += np.sum(pred_classes == true_classes)
            epoch_total += len(y_batch)
        
        # Calculate epoch metrics
        train_loss = np.mean(epoch_losses)
        train_acc = epoch_correct / epoch_total
        
        # Validation phase
        val_loss, val_acc = model.evaluate(x_val, y_val, loss_fn)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training Complete! Total time: {total_time:.1f}s")
    print("=" * 60)
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 128],
                        help='Hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer to use')
    parser.add_argument('--save-dir', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.save_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Print configuration
    print("\n" + "=" * 60)
    print("Neural Network Training Configuration")
    print("=" * 60)
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden layers: {args.hidden}")
    print(f"  Dropout:     {args.dropout}")
    print(f"  Optimizer:   {args.optimizer}")
    print("=" * 60)
    
    # Load data
    print("\nLoading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    # Create data loader
    train_loader = DataLoader(x_train, y_train, batch_size=args.batch_size, shuffle=True)
    
    # Build model
    print("\nBuilding model...")
    model = build_model(
        input_dim=784,
        hidden_dims=args.hidden,
        num_classes=10,
        dropout=args.dropout
    )
    
    # Print model summary
    model.summary()
    
    # Loss function and optimizer
    loss_fn = CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = Adam(lr=args.lr)
    else:
        optimizer = SGD(lr=args.lr, momentum=0.9)
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_data=(x_test, y_test),
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs
    )
    
    # Final evaluation
    print("\nFinal Evaluation on Test Set:")
    test_loss, test_acc = model.evaluate(x_test, y_test, loss_fn)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    
    # Save model
    model_path = output_dir / 'model_weights.npz'
    model.save(model_path)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_training_curves(history, save_path=output_dir / 'training_curves.png')
    visualize_predictions(model, x_test, y_test, num_samples=10,
                         save_path=output_dir / 'predictions.png')
    plot_confusion_matrix(model, x_test, y_test,
                         save_path=output_dir / 'confusion_matrix.png')
    plot_weight_histograms(model, save_path=output_dir / 'weight_histograms.png')
    
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("\n🎉 Training complete! Your neural network achieved {:.2f}% accuracy!".format(
        test_acc * 100))


if __name__ == '__main__':
    main()
