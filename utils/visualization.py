"""
Visualization Utilities
=======================

Tools for visualizing training progress, model predictions,
and what the network has learned.
"""

import numpy as np
from pathlib import Path


def plot_training_curves(history, save_path='training_curves.png'):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dict with 'train_loss', 'train_acc', 'val_loss', 'val_acc' lists
        save_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path}")


def visualize_predictions(model, x, y, num_samples=10, save_path='predictions.png'):
    """
    Visualize model predictions on sample images.
    
    Shows the image, true label, and predicted label.
    Correct predictions in green, wrong in red.
    
    Args:
        model: Trained model with predict() method
        x: Images (N, 784) or (N, 28, 28)
        y: Labels (N,) or (N, 10)
        num_samples: Number of samples to show
        save_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return
    
    # Get predictions
    x_flat = x.reshape(-1, 784) if x.ndim == 3 else x
    predictions = model.predict(x_flat[:num_samples])
    pred_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    if y.ndim == 2:
        true_classes = np.argmax(y[:num_samples], axis=1)
    else:
        true_classes = y[:num_samples]
    
    # Reshape images for display
    images = x[:num_samples]
    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)
    
    # Plot
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(num_samples, 4))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        
        pred = pred_classes[i]
        true = true_classes[i]
        
        color = 'green' if pred == true else 'red'
        ax.set_title(f'Pred: {pred}\nTrue: {true}', color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Model Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved predictions to {save_path}")


def plot_confusion_matrix(model, x, y, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix for classification.
    
    Args:
        model: Trained model
        x: Test images
        y: Test labels
        save_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return
    
    # Get predictions
    x_flat = x.reshape(-1, 784) if x.ndim == 3 else x
    predictions = model.predict(x_flat)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    if y.ndim == 2:
        true_classes = np.argmax(y, axis=1)
    else:
        true_classes = y
    
    # Build confusion matrix
    num_classes = 10
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_classes, pred_classes):
        cm[true, pred] += 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    
    # Labels
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_weight_histograms(model, save_path='weight_histograms.png'):
    """
    Plot histograms of weights in each layer.
    
    Useful for debugging training issues:
    - Weights clustered at 0: Vanishing gradients
    - Weights growing very large: Exploding gradients
    - Nice bell curve: Healthy training
    
    Args:
        model: Model with layers
        save_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return
    
    # Collect weights from layers
    weights = []
    layer_names = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W'):
            weights.append(layer.W.flatten())
            layer_names.append(f'Layer {i} (W)')
    
    if not weights:
        print("No weights to visualize")
        return
    
    # Plot
    num_layers = len(weights)
    fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4))
    
    if num_layers == 1:
        axes = [axes]
    
    for ax, w, name in zip(axes, weights, layer_names):
        ax.hist(w, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title(name)
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Weight Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved weight histograms to {save_path}")


def visualize_gradients(model, x, y, loss_fn, save_path='gradients.png'):
    """
    Visualize gradient magnitudes through the network.
    
    Helps diagnose:
    - Vanishing gradients: Gradients near 0 in early layers
    - Exploding gradients: Very large gradients
    
    Args:
        model: Model to analyze
        x: Sample input
        y: Sample target
        loss_fn: Loss function
        save_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return
    
    # Forward pass
    predictions = model.forward(x, training=True)
    loss = loss_fn.forward(predictions, y)
    
    # Backward pass
    grad = loss_fn.backward()
    
    # Collect gradient magnitudes
    grad_magnitudes = []
    layer_names = []
    
    for i, layer in reversed(list(enumerate(model.layers))):
        grad = layer.backward(grad)
        
        if hasattr(layer, 'grad_W'):
            mag = np.mean(np.abs(layer.grad_W))
            grad_magnitudes.append(mag)
            layer_names.append(f'Layer {i}')
    
    if not grad_magnitudes:
        print("No gradients to visualize")
        return
    
    # Reverse to show in forward order
    grad_magnitudes = grad_magnitudes[::-1]
    layer_names = layer_names[::-1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars = ax.bar(layer_names, grad_magnitudes, color='steelblue', edgecolor='black')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean |Gradient|')
    ax.set_title('Gradient Magnitudes Through Network')
    ax.set_yscale('log')
    
    # Color extreme values
    for bar, mag in zip(bars, grad_magnitudes):
        if mag < 1e-6:
            bar.set_color('red')  # Vanishing
        elif mag > 1:
            bar.set_color('orange')  # Potentially exploding
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved gradient visualization to {save_path}")


def print_training_progress(epoch, num_epochs, train_loss, train_acc, val_loss=None, val_acc=None):
    """
    Print a nicely formatted training progress line.
    
    Args:
        epoch: Current epoch number (0-indexed)
        num_epochs: Total number of epochs
        train_loss: Training loss
        train_acc: Training accuracy
        val_loss: Validation loss (optional)
        val_acc: Validation accuracy (optional)
    """
    progress = f"Epoch {epoch + 1}/{num_epochs} | "
    progress += f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
    
    if val_loss is not None:
        progress += f" | Val Loss: {val_loss:.4f}"
    if val_acc is not None:
        progress += f" | Val Acc: {val_acc:.4f}"
    
    print(progress)
