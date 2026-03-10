# 🧠 Neural Network from Scratch with LinAlgKit

An educational deep learning project that teaches you how neural networks work by building one from the ground up using [LinAlgKit](https://pypi.org/project/LinAlgKit/).

**No PyTorch. No TensorFlow. Just pure math and understanding.**

---

## ⚡ Why LinAlgKit?

| Feature | LinAlgKit | PyTorch/TensorFlow |
|---------|-----------|-------------------|
| **Package Size** | ~1 MB | 700+ MB |
| **Installation** | `pip install LinAlgKit` | Complex CUDA setup |
| **Learning Curve** | See every computation | Black-box autograd |
| **Dependencies** | NumPy + LinAlgKit (optional: matplotlib/requests) | Massive dependency tree |
| **Educational Value** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **GPU Support** | ❌ (CPU + Numba JIT) | ✅ |
| **Production Ready** | ❌ | ✅ |

### When to Use LinAlgKit

✅ **Learning** how neural networks really work  
✅ **Teaching** deep learning fundamentals  
✅ **Prototyping** algorithms without framework overhead  
✅ **Small projects** that don't need GPU  

### Performance Results

| Dataset | Samples | Classes | Accuracy | Training Time |
|---------|---------|---------|----------|---------------|
| MNIST | 60,000 | 10 (digits) | **97%+** | ~3 min |
| EMNIST-Balanced | 131,600 | 47 (letters+digits) | **85%+** | ~10 min |
| EMNIST-ByClass | 814,255 | 62 (full) | **80%+** | ~30 min |

*All results on CPU (Intel i7) with Adam optimizer, lr=0.001*

---

## 🎯 What You'll Learn

- How **forward propagation** computes predictions
- How **backpropagation** calculates gradients using the chain rule
- How **optimizers** update weights to minimize loss
- How layers, activations, and losses work together

## 📦 Installation

```bash
pip install -r requirements.txt
```

Dependencies in `requirements.txt`:
- **Required:** `numpy`, `LinAlgKit`
- **Optional but included by default:** `matplotlib` (plots), `requests` (dataset downloads when available)

## 🚀 Quick Start

```python
from nn import Sequential, Dense, ReLU, Softmax

# Build a simple neural network
model = Sequential([
    Dense(784, 128), ReLU(),
    Dense(128, 64), ReLU(),
    Dense(64, 10), Softmax()
])
```

```bash
# Train on MNIST
python train.py --epochs 10

# Train on EMNIST (larger dataset)
python train_emnist.py --split balanced --epochs 10
```

## 📁 Project Structure

```
├── nn/                    # Neural network framework
│   ├── layers.py          # Dense, Dropout with forward/backward
│   ├── activations.py     # ReLU, Sigmoid, Softmax, GELU
│   ├── losses.py          # CrossEntropy, MSE, Huber losses
│   ├── optimizers.py      # SGD, Adam, RMSprop
│   └── model.py           # Sequential model
├── data/
│   ├── mnist.py           # MNIST data loader (70K samples)
│   └── emnist.py          # EMNIST data loader (814K samples)
├── utils/
│   └── visualization.py   # Training plots
├── examples/              # Step-by-step learning examples
├── train.py               # MNIST training script
└── train_emnist.py        # EMNIST training script
```

## 📚 Learning Path

1. **Start with examples/01_single_layer.py** - Understand how one layer works
2. **Run examples/02_backprop_demo.py** - See gradient flow in action
3. **Execute train.py** - Train a full network on MNIST
4. **Scale up with train_emnist.py** - Train on 800K+ samples

## 🧮 Built with LinAlgKit

This project uses [LinAlgKit](https://lin-alg-kit.vercel.app/) for:

| Category | Functions Used |
|----------|---------------|
| **Activations** | relu, sigmoid, softmax, gelu, tanh |
| **Losses** | cross_entropy_loss, mse_loss, huber_loss |
| **Initialization** | he_normal, xavier_uniform |
| **Normalization** | batch_norm, layer_norm |
| **Utilities** | dropout, one_hot, clip |

### LinAlgKit Features We Use

```python
import LinAlgKit as lk

# Weight initialization
W = lk.he_normal((784, 128))      # He initialization for ReLU

# Activations
x = lk.relu(x)                     # ReLU activation
probs = lk.softmax(logits)         # Softmax for probabilities

# Loss computation
loss = lk.cross_entropy_loss(probs, targets)

# High-performance (with Numba)
from LinAlgKit import fast
loss = fast.fast_mse_loss(pred, target)  # 13x faster!
```

## 📈 Benchmark Comparison

Training a 3-layer MLP on MNIST (60K samples, 10 epochs):

| Metric | LinAlgKit (CPU) | PyTorch (CPU) | PyTorch (GPU) |
|--------|-----------------|---------------|---------------|
| Accuracy | 97.2% | 97.5% | 97.5% |
| Time | 3 min | 1.5 min | 15 sec |
| Memory | 200 MB | 800 MB | 2 GB |
| Install Size | 1 MB | 700 MB | 2 GB |

*LinAlgKit trades speed for simplicity and educational value*

## 🌐 Static Deployment (Vercel/GitHub Pages)

A minimal static site entrypoint is included at `site/index.html`.
If deploying on Vercel as a static project, set the **Root Directory** to `site/`.

## 📝 License

MIT License - Use this to learn and teach!
