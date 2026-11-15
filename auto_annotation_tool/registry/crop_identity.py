"""Deterministyczna tożsamość logicznego cropa tablicy.

Warstwa jest celowo niezależna od SQLite, Tkintera i ścieżek lokalnych.
AZ002 definiuje kontrakt identity; rejestracja cropów w SQLite nastąpi w AZ003.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


CROP_IDENTITY_SCHEMA = "alpr.crop_identity.v1"
CROP_IDENTITY_MODE_LINEAGE = "lineage_v1"

CROP_PIXEL_IDENTITY_SCHEMA = "alpr.crop_pixels.v1"
CROP_IDENTITY_MODE_PIXELS = "pixels_v1"

PZ1_CROP_CONTRACT_VERSION = "pz1_crop.v1"
PZ1_CROP_GENERATOR = "PlateGenerator"

PZ1_OUTPUT_WIDTH = 256
PZ1_SINGLE_ROW_HEIGHT = 64
PZ1_TWO_ROW_HEIGHT = 128

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Zwróć deterministyczną reprezentację JSON do content hashing.

    Kontrakt:
    - UTF-8,
    - sortowanie kluczy,
    - brak zbędnych whitespace,
    - brak NaN/Infinity.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload musi być mapowaniem")

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_crop_identity_sha256(payload: Mapping[str, Any]) -> str:
    """Policz SHA-256 canonical JSON kontraktu crop identity."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_pz1_crop_identity(
    *,
    source_image_id: str,
    source_annotation_id: str,
    source_geometry_hash: str,
    rectify: bool = True,
    do_deskew: bool = False,
    enhance_contrast: bool = False,
    interpolation: str = "lanczos4",
    output_width: int = PZ1_OUTPUT_WIDTH,
    output_height: int = PZ1_SINGLE_ROW_HEIGHT,
    contract_version: str = PZ1_CROP_CONTRACT_VERSION,
) -> dict[str, Any]:
    """Zbuduj logiczny kontrakt identity cropa generowanego przez PZ1.

    GT, projekt, iteracja, nazwa pliku, ścieżka i timestamp NIE są elementami
    identity. Zmiana geometrii lub kontraktu generowania zmienia identity.
    """
    source_image_id = _require_text("source_image_id", source_image_id)
    source_annotation_id = _require_text(
        "source_annotation_id",
        source_annotation_id,
    )
    source_geometry_hash = _require_text(
        "source_geometry_hash",
        source_geometry_hash,
    ).lower()
    contract_version = _require_text(
        "contract_version",
        contract_version,
    )
    interpolation = _normalize_interpolation(interpolation)

    output_width = _require_positive_int("output_width", output_width)
    output_height = _require_positive_int("output_height", output_height)

    return {
        "schema": CROP_IDENTITY_SCHEMA,
        "identity_mode": CROP_IDENTITY_MODE_LINEAGE,
        "source_image_id": source_image_id,
        "source_annotation_id": source_annotation_id,
        "source_geometry_hash": source_geometry_hash,
        "crop_contract": {
            "generator": PZ1_CROP_GENERATOR,
            "contract_version": contract_version,
            "rectify": bool(rectify),
            "do_deskew": bool(do_deskew),
            "enhance_contrast": bool(enhance_contrast),
            "interpolation": interpolation,
            "output_width": output_width,
            "output_height": output_height,
        },
    }


def compute_pz1_crop_identity_sha256(**kwargs: Any) -> str:
    """Skrót: zbuduj kontrakt PZ1 i policz jego SHA-256."""
    return compute_crop_identity_sha256(build_pz1_crop_identity(**kwargs))


def verify_crop_identity_sha256(
    payload: Mapping[str, Any],
    expected_sha256: str,
) -> bool:
    """Zweryfikuj deklarowany SHA kontraktu identity."""
    expected = str(expected_sha256 or "").strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        return False
    return compute_crop_identity_sha256(payload) == expected


def compute_pixel_fallback_sha256(
    *,
    width: int,
    height: int,
    channels: int,
    decoded_rgb_bytes: bytes | bytearray | memoryview,
) -> str:
    """Fallback identity dla cropa bez lineage.

    Wejściem są już zdekodowane bajty RGB/RGBA w ustalonej kolejności.
    Funkcja celowo NIE dekoduje JPG/PNG, aby warstwa identity była czysta
    i niezależna od konkretnej biblioteki obrazu.
    """
    width = _require_positive_int("width", width)
    height = _require_positive_int("height", height)
    channels = _require_positive_int("channels", channels)

    if channels not in (1, 3, 4):
        raise ValueError("channels musi mieć wartość 1, 3 albo 4")

    try:
        pixels = bytes(decoded_rgb_bytes)
    except Exception as exc:
        raise TypeError("decoded_rgb_bytes musi być bytes-like") from exc

    expected_size = width * height * channels
    if len(pixels) != expected_size:
        raise ValueError(
            "Nieprawidłowa liczba bajtów pikseli: "
            f"otrzymano {len(pixels)}, oczekiwano {expected_size}"
        )

    header = (
        f"{CROP_PIXEL_IDENTITY_SCHEMA}\0"
        f"{width}\0{height}\0{channels}\0"
    ).encode("ascii")

    return hashlib.sha256(header + pixels).hexdigest()


def build_pixel_fallback_descriptor(
    *,
    width: int,
    height: int,
    channels: int,
    pixel_sha256: str,
) -> dict[str, Any]:
    """Zbuduj serializowalny opis fallback identity do metadata/manifestu."""
    width = _require_positive_int("width", width)
    height = _require_positive_int("height", height)
    channels = _require_positive_int("channels", channels)
    pixel_sha256 = str(pixel_sha256 or "").strip().lower()

    if channels not in (1, 3, 4):
        raise ValueError("channels musi mieć wartość 1, 3 albo 4")
    if not _SHA256_RE.fullmatch(pixel_sha256):
        raise ValueError("pixel_sha256 musi być pełnym SHA-256 hex")

    return {
        "schema": CROP_PIXEL_IDENTITY_SCHEMA,
        "identity_mode": CROP_IDENTITY_MODE_PIXELS,
        "width": width,
        "height": height,
        "channels": channels,
        "pixel_sha256": pixel_sha256,
    }


def _normalize_interpolation(value: str) -> str:
    normalized = _require_text("interpolation", value).lower()
    aliases = {
        "nearest": "nearest",
        "linear": "linear",
        "bilinear": "linear",
        "cubic": "cubic",
        "bicubic": "cubic",
        "lanczos": "lanczos4",
        "lanczos4": "lanczos4",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(
            "Nieobsługiwana interpolacja crop identity: "
            f"{value!r}"
        )
    return resolved


def _require_text(name: str, value: Any) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} nie może być puste")
    return result


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} musi być dodatnią liczbą całkowitą")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} musi być dodatnią liczbą całkowitą"
        ) from exc
    if result <= 0:
        raise ValueError(f"{name} musi być > 0")
    return result
