"""Model architectures and artifact loading.

Both networks are literal transcriptions of the cells in MLP_mnist.ipynb.
"""

from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf

MODEL_FILES = {
    "dense": "mnist_mlp.keras",
    "conv": "mnist_cnn.keras",
}
MODEL_NAMES = {
    "dense": "Red densa",
    "conv": "Convolucional",
}


def build_dense() -> tf.keras.Model:
    """Dense network exactly as defined in MLP_mnist.ipynb."""
    In = tf.keras.layers.Input(shape=(784,))
    Cp1 = tf.keras.layers.Dense(64, activation="relu")(In)
    Cp2 = tf.keras.layers.Dense(16, activation="relu")(Cp1)  # El relu quita los valores que sean menores a 0
    Out = tf.keras.layers.Dense(10, activation="softmax")(Cp2)
    return tf.keras.Model(inputs=In, outputs=Out)


def build_cnn() -> tf.keras.Model:
    """Convolutional network exactly as defined in MLP_mnist.ipynb."""
    In = tf.keras.layers.Input(shape=(28, 28, 1))
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(In)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(conv1)
    ft = tf.keras.layers.Flatten()(conv2)
    Out = tf.keras.layers.Dense(10, activation="softmax")(ft)
    return tf.keras.Model(inputs=In, outputs=Out)


BUILDERS = {
    "dense": build_dense,
    "conv": build_cnn,
}


def project_root() -> Path:
    """Project directory, whether the package is installed editable or from a wheel."""
    # src/mnist_app/model.py -> src/mnist_app -> src -> project root
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


def model_path(key: str) -> Path:
    """Path of a named artifact inside models/."""
    return project_root() / "models" / MODEL_FILES[key]


def default_model_path() -> Path:
    """Path of the default (dense) artifact, honoring MNIST_MODEL_PATH."""
    env_path = os.environ.get("MNIST_MODEL_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return model_path("dense")


def resolve_model(value: str | Path) -> Path:
    """Accept a model key ("dense"/"conv") or a filesystem path."""
    if isinstance(value, str) and value in MODEL_FILES:
        return model_path(value)
    return Path(value).expanduser()


def load_model(path: str | Path | None = None) -> tf.keras.Model:
    """Load a trained artifact by key, path or default, failing with a clear message."""
    artifact = resolve_model(path) if path else default_model_path()
    if not artifact.is_file():
        raise FileNotFoundError(
            f"No trained model at '{artifact}'.\n"
            "Train it once: `uv run mnist-train --model dense|conv`, "
            "or pass --model with the same key (or an explicit path)."
        )
    return tf.keras.models.load_model(str(artifact))


def load_models(overrides: dict[str, Path] | None = None) -> dict[str, tf.keras.Model]:
    """Load every named model (dense and conv), applying any path overrides."""
    selected = {key: model_path(key) for key in MODEL_FILES}
    selected.update(overrides or {})
    return {key: load_model(path) for key, path in selected.items()}
