#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eksport do YOLO Pose (plate4) na podstawie polygonu 4 punktów.
Format (Ultralytics Pose, kpt_shape [4,2]):

cls xc yc w h x1 y1 x2 y2 x3 y3 x4 y4

Wszystko znormalizowane 0..1.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import shutil

from ..data_models import ImageAnnotation, Detection
from ..config import logger
from ..rectification.quad_geometry import validated_pose_label_points, pose_label_identity_confirmation, format_pose_label


def _sort_corners_clockwise(corners: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Legacy entry point: validate label indices without silently sorting."""
    return validated_pose_label_points(corners)


class YOLOPosePlate4Exporter:
    """
    Eksportuje tylko label 'plate' do YOLO Pose (4 keypointy).
    Struktura:
      output_dir/
        images/   (opcjonalnie kopiowane)
        labels/
        data.yaml (pomocniczy)
    """

    def __init__(self, copy_images: bool = False, class_id: int = 0):
        self.copy_images = copy_images
        self.class_id = class_id  # domyślnie 0: plate

    def export(self, annotations: List[ImageAnnotation], images_dir: Path, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        labels_dir = output_dir / "labels"
        images_out = output_dir / "images"

        # Validate every plate before creating/copying/writing any output.
        prepared = []
        for ann in annotations:
            lines = []
            for det in ann.detections:
                if det.label.lower() != "plate":
                    continue

                try:
                    confirmed = pose_label_identity_confirmation(det.attributes, getattr(det, "geometry", None))
                    poly = validated_pose_label_points(det.polygon, (ann.width, ann.height),
                                                       corner_identity_confirmed=confirmed)
                    line = format_pose_label(poly, (ann.width, ann.height), class_id=self.class_id)
                except ValueError as exc:
                    raise ValueError(f"{ann.filename}: {exc}") from exc
                lines.append(line)

            prepared.append((ann, "\n".join(lines)))

        labels_dir.mkdir(parents=True, exist_ok=True)
        if self.copy_images:
            images_out.mkdir(parents=True, exist_ok=True)
        for ann, content in prepared:
            (labels_dir / f"{Path(ann.filename).stem}.txt").write_text(content, encoding="utf-8")

            if self.copy_images:
                src = Path(images_dir) / ann.filename
                if src.exists():
                    shutil.copy2(src, images_out / ann.filename)

        # ========================================================
        # POPRAWKA PORTABILITY: Usunięto linijkę 'path: C:\...'
        # ========================================================
        data_yaml = output_dir / "data.yaml"
        if not data_yaml.exists():
            data_yaml.write_text(
                """# YOLO Pose dataset (plate4) - Portable version
train: images
val: images

nc: 1
names:
  0: plate

kpt_shape: [4, 2]
flip_idx: [1, 0, 3, 2]
""",
                encoding="utf-8"
            )

        logger.info(f"YOLO Pose export zapisany do (wersja przenośna!): {output_dir}")
        return output_dir
