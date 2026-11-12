"""Train a model locally or send it to Kaggle, with the same command.

    uv run mnist-train --model conv                 # local
    uv run mnist-train --model conv --where kaggle  # Kaggle (push, wait, download)

The training function is shared by both paths, so hyperparameters and the
resulting artifact are comparable no matter where the run happens.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from mnist_app.data import load_mnist
from mnist_app.model import BUILDERS, MODEL_FILES, model_path

KAGGLE_WORKING = Path("/kaggle/working")


def for_model(model_key: str, images: np.ndarray) -> np.ndarray:
    """Dense layers take a 784-vector; the CNN takes 28x28x1."""
    return images.reshape(len(images), -1) if model_key == "dense" else images


def run(
    model_key: str,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 0,
    out: Path | None = None,
    mnist_npz: Path | None = None,
    verbose: int = 2,
) -> tuple[Path, dict]:
    """Fit one model, evaluate on test, save artifact + metrics."""
    tf.keras.utils.set_random_seed(seed)
    model = BUILDERS[model_key]()
    (x_train, y_train), (x_test, y_test) = load_mnist(mnist_npz)
    x_train = for_model(model_key, x_train)
    x_test = for_model(model_key, x_test)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    if out is not None:
        destination = Path(out).expanduser()
    elif KAGGLE_WORKING.is_dir():
        destination = KAGGLE_WORKING / MODEL_FILES[model_key]
    else:
        destination = model_path(model_key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination))

    metrics = {
        "model": model_key,
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "epochs": epochs,
        "batch_size": batch_size,
        "params": int(model.count_params()),
        "tensorflow": tf.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metrics_path = destination.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"\nTest loss: {loss:.4f} | Test accuracy: {accuracy * 100:.2f}%")
    print(f"Model saved to {destination}")
    print(f"Metrics saved to {metrics_path}")
    return destination, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an MNIST model locally or on Kaggle.")
    parser.add_argument("--model", choices=sorted(MODEL_FILES), default="dense")
    parser.add_argument(
        "--where",
        choices=("pc", "kaggle"),
        default="pc",
        help="Train on this machine (pc) or send the run to Kaggle.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="Artifact path (pc only).")
    parser.add_argument("--force", action="store_true", help="Retrain even if the artifact exists.")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Kaggle: push the run and exit without waiting for it.",
    )
    args = parser.parse_args(argv)

    if args.where == "kaggle":
        from mnist_app.kaggle import train_remote

        if args.out is not None:
            print("Note: --out is ignored on Kaggle; artifacts download into models/.")
        return train_remote(
            args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            wait_for_run=not args.no_wait,
        )

    destination = args.out or model_path(args.model)
    if destination.is_file() and not args.force:
        print(f"Model already exists at {destination}; nothing to do.")
        print("Use --force to retrain it.")
        return 0

    run(args.model, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, out=destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
