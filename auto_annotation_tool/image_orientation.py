"""The oriented pixel space used by MT; never rewrites image files or GT."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ORIENTATION_POLICY = "alpr.mt.exif_oriented_xy.v1"
DECODER = "opencv.imdecode.IMREAD_COLOR"


def image_orientation_metadata(path):
    with Image.open(path) as image:
        orientation = image.getexif().get(274, 1)
        if type(orientation) is not int or orientation not in range(1, 9):
            raise ValueError(f"INVALID_EXIF_ORIENTATION: {orientation!r}")
        raw_size = image.size
    size = raw_size[::-1] if orientation in (5, 6, 7, 8) else raw_size
    return {"exif_orientation": orientation, "raw_size": list(raw_size), "oriented_size": list(size)}


def read_oriented_bgr(path):
    """Decode once with OpenCV EXIF handling, including Windows Unicode paths."""
    metadata = image_orientation_metadata(path)
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("MODEL_INPUT_DECODE_FAILED")
    if [image.shape[1], image.shape[0]] != metadata["oriented_size"]:
        raise ValueError("EXIF_DECODER_DIMENSION_MISMATCH")
    return image


def model_input_size(source, path=None):
    """Prefer the actual array sent to YOLO; legacy path-only fallback is oriented."""
    if source is True and path is not None:
        return tuple(image_orientation_metadata(path)["oriented_size"])
    if not isinstance(source, np.ndarray) or source.ndim != 3 or min(source.shape[:2]) <= 0:
        raise ValueError("INVALID_MODEL_INPUT_ARRAY")
    return int(source.shape[1]), int(source.shape[0])


def inspect_oriented_image(path, *, decoder=read_oriented_bgr):
    """Cross-check pixels against Z2's Pillow space, without running a model."""
    metadata = image_orientation_metadata(path)
    image = decoder(path)
    width, height = model_input_size(image)
    if [width, height] != metadata["oriented_size"]:
        raise ValueError("EXIF_DECODER_DIMENSION_MISMATCH")
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source).convert("RGB")
        oriented.thumbnail((192, 192))
        reference = np.asarray(oriented)
        # Reproduce the existing read-only preflight comparison exactly.
        target = oriented.size
        decoded = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), target, interpolation=cv2.INTER_AREA)
    difference = np.abs(decoded.astype(np.int16) - reference.astype(np.int16))
    mean, p95 = float(difference.mean()), float(np.percentile(difference, 95))
    if mean > 6.0 or p95 > 18.0:
        raise ValueError(f"EXIF_DECODER_PIXEL_MISMATCH: MAE={mean:.4f}, p95={p95:.4f}")
    metadata.update(decoder=DECODER, policy=ORIENTATION_POLICY, pixel_format="BGR uint8",
                    decoded_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    decoder_comparison={"mae": mean, "p95": p95, "mae_limit": 6.0, "p95_limit": 18.0})
    return image, metadata
