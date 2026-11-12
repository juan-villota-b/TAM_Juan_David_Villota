"""MNIST loading and normalization.

The dataset is exactly the one used in the notebook: 60k train / 10k test
images, flattened for the dense network and kept as 28x28x1 for the CNN,
scaled to the [0, 1] range.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

INPUT_DIM = 28 * 28
NUM_CLASSES = 10


def normalize(images: np.ndarray) -> np.ndarray:
    """Flatten (N, 28, 28) uint8 images to (N, 784) float32 in [0, 1]."""
    return images.reshape(images.shape[0], INPUT_DIM).astype("float32") / 255.0


def load_mnist(
    npz_path: str | Path | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load MNIST as (N, 28, 28, 1) float32 in [0, 1].

    `npz_path` points at a local mnist.npz copy; when given, it is read with
    NumPy directly (no Keras `get_file`, no download). That is how the Kaggle
    kernel runs with internet disabled.
    """
    if npz_path:
        with np.load(str(Path(npz_path).expanduser()), allow_pickle=False) as data:
            x_train, y_train = data["x_train"], data["y_train"]
            x_test, y_test = data["x_test"], data["y_test"]
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    def prepare(images: np.ndarray) -> np.ndarray:
        return images[..., np.newaxis].astype("float32") / 255.0

    return (prepare(x_train), y_train), (prepare(x_test), y_test)
