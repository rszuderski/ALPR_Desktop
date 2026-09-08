#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-safe annotator factory helpers.

GUI flows should not know low-level constructor defaults. In particular, plate
autoannotation and model ranking need plate geometry only, so OCR stays off
unless a caller explicitly asks for it.
"""

from pathlib import Path

from ..config import CONFIG
from .combined_annotator import CombinedAnnotator
from .plate_annotator import PlateAnnotator
from .vehicle_annotator import VehicleAnnotator


def validate_pt_model_path_for_runtime(raw_path, model_label: str) -> Path:
    path_text = str(raw_path or "").strip()
    if not path_text:
        raise ValueError(f"Wskaż model {model_label} (.pt)!")
    model_path = Path(path_text)
    if not model_path.exists():
        raise ValueError(f"Nie znaleziono modelu {model_label}: {model_path}")
    if not model_path.is_file():
        raise ValueError(f"Ścieżka modelu {model_label} nie wskazuje pliku: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Model {model_label} musi być plikiem .pt.")
    try:
        if model_path.stat().st_size <= 0:
            raise ValueError(f"Model {model_label} jest pusty: {model_path}")
    except OSError as exc:
        raise ValueError(f"Nie można odczytać modelu {model_label}: {exc}") from exc
    return model_path


def create_plate_annotator(
    model_path,
    confidence: float,
    device: str,
    *,
    enable_ocr: bool = False,
    ocr_confidence_threshold: float = 0.3,
    enable_rectification: bool = True,
) -> PlateAnnotator:
    return PlateAnnotator(
        Path(model_path),
        confidence=confidence,
        device=device,
        enable_ocr=enable_ocr,
        ocr_confidence_threshold=ocr_confidence_threshold,
        enable_rectification=enable_rectification,
    )


def create_combined_plate_annotator(
    vehicle_model_path,
    plate_model_path,
    confidence: float,
    device: str,
    *,
    plate_inside_threshold: float | None = None,
) -> CombinedAnnotator:
    return CombinedAnnotator(
        Path(vehicle_model_path),
        Path(plate_model_path),
        vehicle_confidence=confidence,
        plate_confidence=confidence,
        plate_inside_threshold=(
            CONFIG.PLATE_INSIDE_THRESHOLD
            if plate_inside_threshold is None
            else float(plate_inside_threshold)
        ),
        device=device,
    )


def create_vehicle_annotator(
    model_path,
    confidence: float,
    device: str,
) -> VehicleAnnotator:
    return VehicleAnnotator(Path(model_path), confidence, device)
