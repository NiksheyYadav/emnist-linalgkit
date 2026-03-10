"""Pytest-based tests for the neural network framework."""
import numpy as np

from nn import Sequential, Dense, ReLU, Softmax
from nn import CrossEntropyLoss, BinaryCrossEntropyLoss, Adam


def _build_model():
    return Sequential([
        Dense(784, 128),
        ReLU(),
        Dense(128, 10),
        Softmax(),
    ])


def test_forward_output_shape_and_probability_normalization():
    np.random.seed(0)
    model = _build_model()

    x = np.random.randn(32, 784)
    output = model.forward(x)

    assert output.shape == (32, 10)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-6)


def test_backward_computes_gradients_and_optimizer_updates_weights():
    np.random.seed(1)
    model = _build_model()
    x = np.random.randn(16, 784)
    y = np.eye(10)[np.random.randint(0, 10, 16)]

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(lr=0.001)

    predictions = model.forward(x)
    loss = loss_fn.forward(predictions, y)
    grad = loss_fn.backward()
    model.backward(grad)

    dense_layers = [layer for layer in model.layers if hasattr(layer, "W")]
    assert dense_layers, "Expected model to include Dense layers"
    for layer in dense_layers:
        assert layer.grad_W is not None
        assert layer.grad_b is not None
        assert layer.grad_W.shape == layer.W.shape
        assert layer.grad_b.shape == layer.b.shape

    weights_before = model.get_weights()
    optimizer.step(model.layers)
    weights_after = model.get_weights()

    assert np.isfinite(loss)
    assert any(not np.allclose(w1, w2) for w1, w2 in zip(weights_before, weights_after))


def test_train_step_returns_finite_loss():
    np.random.seed(2)
    model = _build_model()
    x = np.random.randn(32, 784)
    y = np.eye(10)[np.random.randint(0, 10, 32)]

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(lr=0.001)

    loss = model.train_step(x, y, loss_fn, optimizer)
    assert np.isfinite(loss)


def test_binary_cross_entropy_backward_consistent_for_vector_and_column_labels():
    predictions_vector = np.array([0.9, 0.2, 0.6, 0.1], dtype=np.float64)
    targets_vector = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)

    loss_vec = BinaryCrossEntropyLoss()
    loss_vec.forward(predictions_vector, targets_vector)
    grad_vec = loss_vec.backward()

    predictions_column = predictions_vector.reshape(-1, 1)
    targets_column = targets_vector.reshape(-1, 1)
    loss_col = BinaryCrossEntropyLoss()
    loss_col.forward(predictions_column, targets_column)
    grad_col = loss_col.backward().reshape(-1)

    assert np.allclose(grad_vec, grad_col)
