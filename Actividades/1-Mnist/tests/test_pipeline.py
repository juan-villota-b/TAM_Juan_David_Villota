"""End-to-end checks for the preprocessing, model and HTTP pipeline."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mnist_app.model import build_dense, model_path
from mnist_app.predict import architecture_label, predict_array
from mnist_app.preprocess import EmptyImageError, preprocess_image


def _digit_image(size=(220, 220), background=(255, 255, 255), ink=(20, 20, 20)) -> Image.Image:
    """Draw a thick vertical stroke, like a handwritten '1'."""
    image = Image.new("L", size, background[0])
    draw = ImageDraw.Draw(image)
    draw.line([(size[0] // 2, 30), (size[0] // 2, size[1] - 30)], fill=ink[0], width=18)
    return image


def test_preprocess_produces_mnist_frame() -> None:
    canvas = preprocess_image(_digit_image())

    assert canvas.shape == (28, 28)
    assert canvas.dtype == np.uint8
    assert canvas.max() == 255
    assert canvas[:2, :2].max() == 0, "the frame must stay black around the glyph"


def test_preprocess_accepts_dark_background() -> None:
    light = preprocess_image(_digit_image())
    dark = preprocess_image(_digit_image(background=(0, 0, 0), ink=(255, 255, 255)))

    # Both polarities must end up as white ink on black.
    assert dark.shape == (28, 28)
    assert dark.max() == 255
    assert np.array_equal(light > 127, dark > 127)


def test_preprocess_handles_transparency() -> None:
    image = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(image).line([(80, 20), (80, 140)], fill=(0, 0, 0, 255), width=16)

    canvas = preprocess_image(image)
    assert canvas.shape == (28, 28)
    assert canvas.max() == 255


def test_preprocess_rejects_blank_image() -> None:
    with pytest.raises(EmptyImageError):
        preprocess_image(Image.new("L", (100, 100), 255))


def test_model_topology_matches_notebook() -> None:
    model = build_dense()
    hidden = [layer.units for layer in model.layers if layer.__class__.__name__ == "Dense"]

    assert hidden == [64, 16, 10]
    assert model.count_params() == 784 * 64 + 64 + 64 * 16 + 16 + 16 * 10 + 10


def test_predict_returns_distribution() -> None:
    model = build_dense()
    prediction = predict_array(model, np.zeros((28, 28), dtype=np.uint8))

    assert len(prediction.probabilities) == 10
    assert 0 <= prediction.digit <= 9
    assert sum(prediction.probabilities) == pytest.approx(1.0, abs=1e-5)
    assert prediction.confidence == pytest.approx(max(prediction.probabilities))


def test_predict_supports_convnet_input() -> None:
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    features = tf.keras.layers.Conv2D(4, (3, 3), activation="relu")(inputs)
    features = tf.keras.layers.Flatten()(features)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(features)
    convnet = tf.keras.Model(inputs=inputs, outputs=outputs)

    prediction = predict_array(convnet, np.zeros((28, 28), dtype=np.uint8))

    assert len(prediction.probabilities) == 10
    assert sum(prediction.probabilities) == pytest.approx(1.0, abs=1e-5)
    assert architecture_label(convnet) == "CNN 28×28×1"
    assert architecture_label(build_dense()) == "MLP 784"


ARTIFACTS = [model_path("dense"), model_path("conv")]

TRAINED = pytest.mark.skipif(
    not all(path.is_file() for path in ARTIFACTS),
    reason="train both artifacts first with mnist-train",
)


@TRAINED
def test_http_serves_both_models() -> None:
    from mnist_app.app import create_app

    app = create_app()
    client = app.test_client()

    page = client.get("/").get_data(as_text=True)
    assert "Red densa" in page and "Convolucional" in page

    buffer = io.BytesIO()
    _digit_image().save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        data={"image": (buffer, "digit.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    for key in ("dense", "conv"):
        assert 0 <= payload[key]["digit"] <= 9
        assert sum(payload[key]["probabilities"]) == pytest.approx(1.0, abs=1e-5)
    assert payload["processed_png"].startswith("data:image/png;base64,")


@TRAINED
def test_http_predict_rejects_missing_file() -> None:
    from mnist_app.app import create_app

    client = create_app().test_client()
    response = client.post("/predict", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_kaggle_launcher_and_metadata() -> None:
    from mnist_app.kaggle import dataset_id, kernel_metadata, launcher

    code = launcher("conv", 30, 32, 0)
    assert 'MOUNT = Path("/kaggle/input/mnist-repo")' in code
    assert '"conv"' in code and "mnist_npz=REPO" in code

    metadata = kernel_metadata("conv", "someone")
    assert metadata["id"] == "someone/mnist-conv-training"
    assert metadata["dataset_sources"] == [dataset_id("someone")]
    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is True
    assert metadata["is_private"] is True


def test_load_mnist_from_npz_offline() -> None:
    from mnist_app.data import load_mnist

    npz = Path.home() / ".keras" / "datasets" / "mnist.npz"
    if not npz.is_file():
        pytest.skip("MNIST cache not present")

    (x_train, y_train), (x_test, y_test) = load_mnist(npz)

    assert x_train.shape == (60000, 28, 28, 1)
    assert x_test.shape == (10000, 28, 28, 1)
    assert x_train.dtype == np.float32
    assert float(x_train.max()) <= 1.0
