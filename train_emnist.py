"""
Train a Neural Network on EMNIST
=================================

Train on EMNIST - a much larger dataset than MNIST with letters AND digits.

EMNIST Splits:
- balanced: 47 classes, 131,600 samples (recommended)
- letters: 26 classes (A-Z), 145,600 samples
- digits: 10 classes (0-9), 280,000 samples  
- byclass: 62 classes (full), 814,255 samples

Usage:
    python train_emnist.py                      # Train with defaults (balanced)
    python train_emnist.py --split letters      # Train on letters only
    python train_emnist.py --split byclass      # Train on full 62 classes
    python train_emnist.py --epochs 20          # More epochs
"""

import argparse
import numpy as np
import time
from pathlib import Path

from nn import Sequential, Dense, ReLU, Softmax, Dropout
from nn import CrossEntropyLoss, Adam, SGD
from data.emnist import load_emnist, EMNISTDataLoader, get_emnist_label
from utils.visualization import plot_training_curves, print_training_progress


def build_emnist_model(input_dim=784, num_classes=47, architecture='medium'):
    """
    Build a neural network for EMNIST.
    
    Architecture options:
    - small: 784 -> 256 -> 128 -> classes
    - medium: 784 -> 512 -> 256 -> 128 -> classes  
    - large: 784 -> 1024 -> 512 -> 256 -> 128 -> classes
    """
    if architecture == 'small':
        hidden_dims = [256, 128]
    elif architecture == 'medium':
        hidden_dims = [512, 256, 128]
    elif architecture == 'large':
        hidden_dims = [1024, 512, 256, 128]
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(Dense(prev_dim, hidden_dim))
        layers.append(ReLU())
        layers.append(Dropout(0.3))  # Higher dropout for larger dataset
        prev_dim = hidden_dim
    
    layers.append(Dense(prev_dim, num_classes))
    layers.append(Softmax())
    
    return Sequential(layers)


def train_emnist(model, train_loader, val_data, loss_fn, optimizer, epochs, 
                 print_every=100):
    """Train on EMNIST with progress tracking."""
    x_val, y_val = val_data
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\n" + "=" * 70)
    print("EMNIST Training Started")
    print("=" * 70)
    
    total_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        losses = []
        correct, total = 0, 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            loss = model.train_step(x_batch, y_batch, loss_fn, optimizer)
            losses.append(loss)
            
            # Track accuracy
            preds = np.argmax(model.predict(x_batch), axis=1)
            labels = np.argmax(y_batch, axis=1)
            correct += np.sum(preds == labels)
            total += len(y_batch)
            
            # Print progress
            if (batch_idx + 1) % print_every == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss:.4f} | Acc: {correct/total:.4f}")
        
        train_loss = np.mean(losses)
        train_acc = correct / total
        
        # Validation (use subset for speed)
        val_subset = min(10000, len(x_val))
        val_indices = np.random.choice(len(x_val), val_subset, replace=False)
        val_loss, val_acc = model.evaluate(x_val[val_indices], y_val[val_indices], loss_fn)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{epochs} Complete | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s\n")
    
    total_time = time.time() - total_start
    print("=" * 70)
    print(f"Training Complete! Total time: {total_time/60:.1f} minutes")
    print("=" * 70)
    
    return history


def visualize_emnist_predictions(model, x, y, info, num_samples=10, save_path=None):
    """Show predictions with character labels."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return
    
    predictions = model.predict(x[:num_samples])
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y[:num_samples], axis=1)
    
    images = x[:num_samples].reshape(-1, 28, 28)
    
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 5))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        
        pred_label = info['labels'][pred_classes[i]]
        true_label = info['labels'][true_classes[i]]
        
        color = 'green' if pred_classes[i] == true_classes[i] else 'red'
        ax.set_title(f"P:{pred_label} T:{true_label}", color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('EMNIST Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved predictions to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train on EMNIST dataset')
    parser.add_argument('--split', type=str, default='balanced',
                        choices=['byclass', 'bymerge', 'balanced', 'letters', 'digits'],
                        help='EMNIST split to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--arch', type=str, default='medium',
                        choices=['small', 'medium', 'large'],
                        help='Model architecture size')
    parser.add_argument('--save-dir', type=str, default='output_emnist',
                        help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.save_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Print config
    print("\n" + "=" * 70)
    print("EMNIST Training Configuration")
    print("=" * 70)
    print(f"  Split:       {args.split}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Architecture: {args.arch}")
    print("=" * 70)
    
    # Load data
    print("\nLoading EMNIST dataset (this may take a while on first run)...")
    (x_train, y_train), (x_test, y_test), info = load_emnist(args.split)
    
    # Create loader
    train_loader = EMNISTDataLoader(x_train, y_train, batch_size=args.batch_size)
    
    # Build model
    print(f"\nBuilding {args.arch} model for {info['num_classes']} classes...")
    model = build_emnist_model(
        input_dim=784,
        num_classes=info['num_classes'],
        architecture=args.arch
    )
    model.summary()
    
    # Setup training
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(lr=args.lr)
    
    # Train
    history = train_emnist(
        model=model,
        train_loader=train_loader,
        val_data=(x_test, y_test),
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        print_every=200
    )
    
    # Final evaluation
    print("\nFinal Evaluation on Full Test Set:")
    test_loss, test_acc = model.evaluate(x_test, y_test, loss_fn)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    
    # Save
    model.save(output_dir / 'emnist_model.npz')
    plot_training_curves(history, save_path=output_dir / 'emnist_training.png')
    visualize_emnist_predictions(model, x_test, y_test, info, 
                                  save_path=output_dir / 'emnist_predictions.png')
    
    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print(f"\nFinal accuracy: {test_acc*100:.2f}% on {info['num_classes']} classes!")


if __name__ == '__main__':
    main()
