"""Turn an arbitrary photo of a handwritten digit into a 28x28 MNIST-like image.

Ported from the notebook's preprocessing. Two robustness changes for real
photos: the light-background test uses the median border instead of the four
corners, and an image with no ink raises a clear error instead of failing later.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift

CANVAS = 28
DIGIT_BOX = 20  # MNIST puts the glyph in a 20x20 box inside the 28x28 frame


class EmptyImageError(ValueError):
    """The image has no detectable glyph."""


def _to_grayscale(img: Image.Image) -> Image.Image:
    """Flatten transparency onto white, then convert to grayscale."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("L")


def _has_light_background(arr: np.ndarray) -> bool:
    """Photos of paper have a light border; MNIST-style captures have a dark one."""
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return float(np.median(border)) > 127


def _keep_largest_component(binary: np.ndarray) -> np.ndarray:
    """Drop shadows, borders and noise, keeping only the biggest blob."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def _fit_to_canvas(glyph: np.ndarray) -> np.ndarray:
    """Scale the glyph into a 20x20 box preserving aspect ratio, then center it."""
    height, width = glyph.shape
    if height >= width:
        new_h = DIGIT_BOX
        new_w = max(1, int(round(width * (DIGIT_BOX / height))))
    else:
        new_w = DIGIT_BOX
        new_h = max(1, int(round(height * (DIGIT_BOX / width))))

    resized = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    top = (CANVAS - new_h) // 2
    left = (CANVAS - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _center_of_mass_canvas(canvas: np.ndarray) -> np.ndarray:
    """Shift the glyph so its center of mass sits at the canvas center."""
    cy, cx = center_of_mass(canvas)
    if not np.isfinite(cy) or not np.isfinite(cx):
        return canvas
    offset_y, offset_x = np.round((CANVAS / 2) - cy), np.round((CANVAS / 2) - cx)
    shifted = shift(canvas, shift=(offset_y, offset_x), cval=0)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert a PIL image into a uint8 (28, 28) MNIST-like array (ink = high)."""
    arr = np.array(_to_grayscale(img), dtype=np.uint8)

    if _has_light_background(arr):
        arr = 255 - arr

    # Otsu: pure black background, pure white ink, shadows removed.
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    arr = _keep_largest_component(arr)

    coords = cv2.findNonZero(arr)
    if coords is None:
        raise EmptyImageError("No se detectó ningún trazo: la imagen parece vacía tras el umbral.")

    x, y, w, h = cv2.boundingRect(coords)
    if w < 2 or h < 2:
        raise EmptyImageError("El trazo detectado es demasiado pequeño para clasificarlo.")

    canvas = _fit_to_canvas(arr[y : y + h, x : x + w])
    return _center_of_mass_canvas(canvas)
