"""Make inference independent of the model's input rank.

The MLP consumes a flat 784-vector; the CNN consumes (28, 28, 1). Both are fed
the same preprocessed uint8 image, so the batch shape comes from the model.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mnist_app.model import load_model
from mnist_app.preprocess import preprocess_image


@dataclass(frozen=True)
class Prediction:
    """Classification result for a single image."""

    digit: int
    confidence: float
    probabilities: list[float]

    def as_dict(self) -> dict:
        return {
            "digit": self.digit,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


def input_shape(model) -> tuple:
    """Return the model's input shape, unwrapping multi-input models."""
    shape = model.input_shape
    return tuple(shape[0] if isinstance(shape, list) else shape)


def architecture_label(model) -> str:
    """Short human label derived from the input shape, e.g. 'CNN 28×28×1'."""
    shape = input_shape(model)
    if len(shape) == 4:
        return f"CNN {shape[1]}×{shape[2]}×{shape[3]}"
    return f"MLP {shape[1] if len(shape) > 1 else '?'}"


def predict_array(model, image_28x28: np.ndarray) -> Prediction:
    """Classify a preprocessed uint8 (28, 28) image with an MLP or a CNN."""
    if len(input_shape(model)) == 4:
        batch = image_28x28[np.newaxis, ..., np.newaxis]
    else:
        batch = image_28x28.reshape(1, -1)

    probabilities = model.predict(batch.astype("float32") / 255.0, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    return Prediction(
        digit=digit,
        confidence=float(probabilities[digit]),
        probabilities=[float(p) for p in probabilities],
    )


def png_data_url(image_28x28: np.ndarray) -> str:
    """Encode the processed 28x28 array as a PNG data URL for the browser."""
    buffer = io.BytesIO()
    Image.fromarray(image_28x28).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify an image of a handwritten digit.")
    parser.add_argument("image", type=Path, help="Path to the image (PNG or JPG).")
    parser.add_argument(
        "--model",
        default="dense",
        help="Model key ('dense' or 'conv') or path to a .keras artifact (default: dense).",
    )
    args = parser.parse_args(argv)

    model = load_model(args.model)
    processed = preprocess_image(Image.open(args.image))
    prediction = predict_array(model, processed)

    print(f"Model: {architecture_label(model)}")
    print(f"Prediction: {prediction.digit}")
    print(f"Confidence: {prediction.confidence * 100:.2f}%")
    print("Probabilities:")
    for digit, probability in enumerate(prediction.probabilities):
        print(f"  {digit}: {probability * 100:6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
