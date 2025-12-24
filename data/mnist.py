"""
MNIST Data Loader
=================

MNIST is the "Hello World" of machine learning - a dataset of 70,000
handwritten digit images (0-9). Each image is 28x28 grayscale pixels.

This module handles:
1. Downloading MNIST from the internet
2. Parsing the binary format
3. Normalizing pixel values
4. One-hot encoding labels
5. Creating mini-batch iterators

For educational purposes, we show the complete pipeline from raw data
to training-ready batches.
"""

import numpy as np
import gzip
import os
import struct
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request


# MNIST URLs (hosted by Yann LeCun)
MNIST_URLS = {
    'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
}


def download_file(url, filepath):
    """
    Download a file from URL to local path.
    
    Args:
        url: URL to download from
        filepath: Local path to save to
    """
    print(f"Downloading {url}...")
    
    if HAS_REQUESTS:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        urllib.request.urlretrieve(url, filepath)
    
    print(f"Saved to {filepath}")


def parse_idx_images(filepath):
    """
    Parse IDX file format for images.
    
    IDX is a simple binary format:
    - Bytes 0-3: Magic number (checking file type)
    - Bytes 4-7: Number of images
    - Bytes 8-11: Number of rows (28 for MNIST)
    - Bytes 12-15: Number of columns (28 for MNIST)
    - Remaining: Pixel data (unsigned bytes, 0-255)
    
    Args:
        filepath: Path to .gz file
    
    Returns:
        numpy array of shape (num_images, 28, 28)
    """
    with gzip.open(filepath, 'rb') as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        
        # Verify magic number (2051 for images)
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")
        
        # Read all pixel data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Reshape to (num_images, rows, cols)
        images = data.reshape(num_images, rows, cols)
    
    return images


def parse_idx_labels(filepath):
    """
    Parse IDX file format for labels.
    
    Format:
    - Bytes 0-3: Magic number (2049 for labels)
    - Bytes 4-7: Number of labels
    - Remaining: Label data (unsigned bytes, 0-9)
    
    Args:
        filepath: Path to .gz file
    
    Returns:
        numpy array of shape (num_labels,)
    """
    with gzip.open(filepath, 'rb') as f:
        # Read header
        magic, num_labels = struct.unpack('>II', f.read(8))
        
        # Verify magic number (2049 for labels)
        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")
        
        # Read all labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return labels


def load_mnist(data_dir='./data/mnist', normalize=True, flatten=True, one_hot=True):
    """
    Load MNIST dataset.
    
    Downloads if not present, then parses and preprocesses.
    
    Args:
        data_dir: Directory to store/load data
        normalize: If True, scale pixels to [0, 1]
        flatten: If True, reshape images to (num_samples, 784)
        one_hot: If True, convert labels to one-hot encoding
    
    Returns:
        Tuple of ((x_train, y_train), (x_test, y_test))
        
    Example:
        >>> (x_train, y_train), (x_test, y_test) = load_mnist()
        >>> print(x_train.shape)  # (60000, 784)
        >>> print(y_train.shape)  # (60000, 10)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths
    files = {
        'train_images': data_dir / 'train-images-idx3-ubyte.gz',
        'train_labels': data_dir / 'train-labels-idx1-ubyte.gz',
        'test_images': data_dir / 't10k-images-idx3-ubyte.gz',
        'test_labels': data_dir / 't10k-labels-idx1-ubyte.gz',
    }
    
    # Download missing files
    for key, filepath in files.items():
        if not filepath.exists():
            download_file(MNIST_URLS[key], filepath)
    
    # Parse files
    print("Loading MNIST...")
    x_train = parse_idx_images(files['train_images'])
    y_train = parse_idx_labels(files['train_labels'])
    x_test = parse_idx_images(files['test_images'])
    y_test = parse_idx_labels(files['test_labels'])
    
    # Preprocessing
    # Convert to float
    x_train = x_train.astype(np.float64)
    x_test = x_test.astype(np.float64)
    
    if normalize:
        # Scale pixels from [0, 255] to [0, 1]
        x_train /= 255.0
        x_test /= 255.0
    
    if flatten:
        # Reshape from (N, 28, 28) to (N, 784)
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)
    
    if one_hot:
        # Convert labels to one-hot encoding
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]
    
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


class DataLoader:
    """
    Mini-batch Data Loader
    ======================
    
    Iterates over data in mini-batches, optionally shuffling.
    
    Why mini-batches?
    -----------------
    - Full batch (all data): Stable gradients, but slow and memory-heavy
    - Single sample (SGD): Fast iterations, but noisy gradients
    - Mini-batch: Best of both worlds - fast and stable
    
    Typical batch sizes: 32, 64, 128, 256
    
    Example:
        >>> loader = DataLoader(x_train, y_train, batch_size=64, shuffle=True)
        >>> for x_batch, y_batch in loader:
        ...     loss = model.train_step(x_batch, y_batch, loss_fn, optimizer)
    """
    
    def __init__(self, x, y, batch_size=32, shuffle=True):
        """
        Initialize DataLoader.
        
        Args:
            x: Input data of shape (num_samples, ...)
            y: Labels of shape (num_samples, ...)
            batch_size: Number of samples per batch
            shuffle: If True, shuffle data each epoch
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = x.shape[0]
    
    def __len__(self):
        """Return number of batches per epoch."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """
        Iterate over mini-batches.
        
        The key insight: Shuffling introduces randomness that helps
        the model generalize better and escape local minima.
        """
        # Create indices
        indices = np.arange(self.num_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Yield batches
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield self.x[batch_indices], self.y[batch_indices]


def visualize_samples(x, y, num_samples=10):
    """
    Visualize MNIST samples (requires matplotlib).
    
    Args:
        x: Images, either (N, 784) or (N, 28, 28)
        y: Labels, either (N,) or (N, 10)
        num_samples: Number of samples to display
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return
    
    # Reshape if flattened
    if x.ndim == 2 and x.shape[1] == 784:
        x = x.reshape(-1, 28, 28)
    
    # Get class labels
    if y.ndim == 2:
        labels = np.argmax(y, axis=1)
    else:
        labels = y
    
    # Plot
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1.5))
    
    for i, ax in enumerate(axes):
        ax.imshow(x[i], cmap='gray')
        ax.set_title(str(labels[i]))
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png')
    plt.close()
    print("Saved sample visualization to mnist_samples.png")


if __name__ == '__main__':
    # Test the data loader
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    print(f"\nDataset loaded:")
    print(f"  Training:   {x_train.shape[0]} samples")
    print(f"  Test:       {x_test.shape[0]} samples")
    print(f"  Image size: {x_train.shape[1]} features (28x28 flattened)")
    print(f"  Classes:    {y_train.shape[1]} (digits 0-9)")
    
    # Test batch iteration
    loader = DataLoader(x_train, y_train, batch_size=64)
    print(f"\nBatches per epoch: {len(loader)}")
    
    # Visualize
    visualize_samples(x_train, y_train)
