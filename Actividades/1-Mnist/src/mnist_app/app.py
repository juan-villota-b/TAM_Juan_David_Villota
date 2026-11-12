"""Flask server exposing both trained models on localhost.

The upload stays in memory: nothing is written to disk, and only images are
accepted. Defaults to 127.0.0.1 so the interface is not exposed to the network.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from mnist_app.model import load_models
from mnist_app.predict import architecture_label, png_data_url, predict_array
from mnist_app.preprocess import EmptyImageError, preprocess_image

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 40_000_000
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}


def create_app(model_paths: dict[str, Path] | None = None) -> Flask:
    """Build the Flask app with both models loaded once at startup."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    models = load_models(model_paths)
    app.config["MNIST_MODELS"] = models

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            models={key: architecture_label(model) for key, model in models.items()},
        )

    @app.post("/predict")
    def predict():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="No se recibió ningún archivo. Enviá uno en el campo 'image'."), 400

        suffix = Path(upload.filename).suffix.lower()
        if suffix and suffix not in ALLOWED_EXTENSIONS:
            return jsonify(error=f"Extensión no soportada: '{suffix}'. Usá PNG, JPG o WEBP."), 415

        try:
            image = Image.open(upload.stream)
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return jsonify(error="El archivo no es una imagen legible."), 400

        if image.width * image.height > MAX_PIXELS:
            return jsonify(error="La imagen es demasiado grande: el máximo es 40 megapíxeles."), 422

        try:
            processed = preprocess_image(image)
        except EmptyImageError as exc:
            return jsonify(error=str(exc)), 422

        payload = {"processed_png": png_data_url(processed)}
        for key, model in models.items():
            payload[key] = predict_array(model, processed).as_dict()
        return jsonify(payload)

    @app.errorhandler(413)
    def too_large(_error):
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        return jsonify(error=f"La imagen supera el límite de {limit_mb} MB."), 413

    return app


def lan_ip() -> str | None:
    """Best-effort LAN address, used to print a usable URL when bound to 0.0.0.0."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("8.8.8.8", 80))  # picks a route, sends no packets
            return probe.getsockname()[0]
        except OSError:
            return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve both MNIST classifiers on localhost.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000).")
    parser.add_argument("--dense", type=Path, default=None, help="Override the dense model artifact.")
    parser.add_argument("--conv", type=Path, default=None, help="Override the convolutional artifact.")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    args = parser.parse_args(argv)

    overrides = {
        key: path for key, path in (("dense", args.dense), ("conv", args.conv)) if path
    } or None

    try:
        app = create_app(overrides)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    labels = {key: architecture_label(model) for key, model in app.config["MNIST_MODELS"].items()}
    print("Loaded: " + ", ".join(f"{key} ({label})" for key, label in labels.items()))
    if args.host in {"0.0.0.0", "::"}:
        address = lan_ip()
        public = f" → http://{address}:{args.port}" if address else ""
        print(f"Serving on all interfaces{public}  (Ctrl+C to stop)")
    else:
        print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
