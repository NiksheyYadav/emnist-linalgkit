"""
EMNIST Data Loader
==================

EMNIST (Extended MNIST) is a much larger dataset with handwritten letters
AND digits. It has 814,000 samples - over 10x larger than MNIST!

Available splits:
- ByClass: 62 classes (0-9, A-Z, a-z) - 814,255 samples
- ByMerge: 47 classes (merged similar chars) - 814,255 samples
- Balanced: 47 classes, equal samples - 131,600 samples
- Letters: 26 classes (A-Z) - 145,600 samples
- Digits: 10 classes (0-9) - 280,000 samples
- MNIST: 10 classes - 70,000 samples (original MNIST)

This module handles:
1. Downloading EMNIST from the internet
2. Parsing the binary IDX format
3. Normalizing and preprocessing
4. Creating mini-batch iterators
"""

import numpy as np
import gzip
import os
import struct
import zipfile
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request


# EMNIST download URL (hosted by NIST)
EMNIST_URL = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'

# Alternative mirror
EMNIST_MIRROR = 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'


# Class mappings for different splits
EMNIST_MAPPINGS = {
    'byclass': {
        'num_classes': 62,
        'labels': list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    },
    'bymerge': {
        'num_classes': 47,
        'labels': list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt')
    },
    'balanced': {
        'num_classes': 47,
        'labels': list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt')
    },
    'letters': {
        'num_classes': 26,
        'labels': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    },
    'digits': {
        'num_classes': 10,
        'labels': list('0123456789')
    },
    'mnist': {
        'num_classes': 10,
        'labels': list('0123456789')
    }
}


def download_file(url, filepath, description="file"):
    """Download a file from URL with progress."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    
    if HAS_REQUESTS:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded * 100 // total_size
                    print(f"\r  Progress: {pct}% ({downloaded // 1024 // 1024} MB)", end="")
        print()
    else:
        urllib.request.urlretrieve(url, filepath)
    
    print(f"  Saved to {filepath}")


def parse_idx_images(filepath):
    """Parse IDX file format for images."""
    with gzip.open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # EMNIST images need to be transposed
        images = data.reshape(num_images, rows, cols)
        images = np.transpose(images, (0, 2, 1))  # Fix orientation
    return images


def parse_idx_labels(filepath):
    """Parse IDX file format for labels."""
    with gzip.open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_emnist(split='balanced', data_dir='./data/emnist', normalize=True, 
                flatten=True, one_hot=True):
    """
    Load EMNIST dataset.
    
    Args:
        split: One of 'byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist'
        data_dir: Directory to store/load data
        normalize: If True, scale pixels to [0, 1]
        flatten: If True, reshape images to (num_samples, 784)
        one_hot: If True, convert labels to one-hot encoding
    
    Returns:
        Tuple of ((x_train, y_train), (x_test, y_test), info)
        info contains: num_classes, labels list
    
    Example:
        >>> (x_train, y_train), (x_test, y_test), info = load_emnist('balanced')
        >>> print(f"Training samples: {x_train.shape[0]}")
        >>> print(f"Classes: {info['num_classes']}")
    """
    split = split.lower()
    if split not in EMNIST_MAPPINGS:
        raise ValueError(f"Unknown split: {split}. Choose from: {list(EMNIST_MAPPINGS.keys())}")
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download EMNIST if not present
    zip_path = data_dir / 'gzip.zip'
    gzip_dir = data_dir / 'gzip'
    
    if not gzip_dir.exists():
        if not zip_path.exists():
            try:
                download_file(EMNIST_URL, zip_path, "EMNIST dataset (~500MB)")
            except Exception as e:
                print(f"Primary download failed: {e}")
                print("Trying mirror...")
                download_file(EMNIST_MIRROR, zip_path, "EMNIST dataset (~500MB)")
        
        print("Extracting EMNIST...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
        print("Extraction complete!")
    
    # File paths for the selected split
    prefix = f'emnist-{split}'
    files = {
        'train_images': gzip_dir / f'{prefix}-train-images-idx3-ubyte.gz',
        'train_labels': gzip_dir / f'{prefix}-train-labels-idx1-ubyte.gz',
        'test_images': gzip_dir / f'{prefix}-test-images-idx3-ubyte.gz',
        'test_labels': gzip_dir / f'{prefix}-test-labels-idx1-ubyte.gz',
    }
    
    # Check files exist
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
    
    # Parse files
    print(f"Loading EMNIST-{split}...")
    x_train = parse_idx_images(files['train_images'])
    y_train = parse_idx_labels(files['train_labels'])
    x_test = parse_idx_images(files['test_images'])
    y_test = parse_idx_labels(files['test_labels'])
    
    # Preprocessing
    x_train = x_train.astype(np.float64)
    x_test = x_test.astype(np.float64)
    
    if normalize:
        x_train /= 255.0
        x_test /= 255.0
    
    if flatten:
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)
    
    # Get class info
    info = EMNIST_MAPPINGS[split].copy()
    
    if one_hot:
        num_classes = info['num_classes']
        y_train = np.eye(num_classes)[y_train]
        y_test = np.eye(num_classes)[y_test]
    
    print(f"  Split: {split}")
    print(f"  Train: {x_train.shape[0]:,} samples")
    print(f"  Test:  {x_test.shape[0]:,} samples")
    print(f"  Classes: {info['num_classes']}")
    
    return (x_train, y_train), (x_test, y_test), info


def get_emnist_label(class_idx, split='balanced'):
    """Get human-readable label for a class index."""
    return EMNIST_MAPPINGS[split.lower()]['labels'][class_idx]


class EMNISTDataLoader:
    """
    Mini-batch data loader for EMNIST.
    
    Same as regular DataLoader but with EMNIST-specific features.
    """
    
    def __init__(self, x, y, batch_size=128, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = x.shape[0]
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield self.x[batch_indices], self.y[batch_indices]


if __name__ == '__main__':
    # Test loading EMNIST
    print("=" * 60)
    print("EMNIST Dataset Test")
    print("=" * 60)
    
    # Load balanced split (good for training)
    (x_train, y_train), (x_test, y_test), info = load_emnist('balanced')
    
    print(f"\nDataset loaded successfully!")
    print(f"  x_train shape: {x_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  x_test shape:  {x_test.shape}")
    print(f"  y_test shape:  {y_test.shape}")
    
    print(f"\nClass labels: {info['labels'][:10]}... ({info['num_classes']} total)")
    
    # Test data loader
    loader = EMNISTDataLoader(x_train, y_train, batch_size=128)
    print(f"\nBatches per epoch: {len(loader)}")
