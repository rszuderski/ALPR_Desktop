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


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sort_corners_clockwise(corners: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sortuje 4 punkty do kolejności TL, TR, BR, BL."""
    if len(corners) != 4:
        return corners

    cx = sum(p[0] for p in corners) / 4
    cy = sum(p[1] for p in corners) / 4

    top = [p for p in corners if p[1] < cy]
    bottom = [p for p in corners if p[1] >= cy]

    if len(top) != 2 or len(bottom) != 2:
        s = sorted(corners, key=lambda p: p[1])
        top = sorted(s[:2], key=lambda p: p[0])
        bottom = sorted(s[2:], key=lambda p: p[0], reverse=True)
    else:
        top = sorted(top, key=lambda p: p[0])
        bottom = sorted(bottom, key=lambda p: p[0], reverse=True)

    return [top[0], top[1], bottom[0], bottom[1]]


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

        labels_dir.mkdir(parents=True, exist_ok=True)
        if self.copy_images:
            images_out.mkdir(parents=True, exist_ok=True)

        for ann in annotations:
            lines = []
            for det in ann.detections:
                if det.label.lower() != "plate":
                    continue

                poly = det.polygon
                if not poly or len(poly) < 4:
                    # fallback: z bbox
                    x1, y1, x2, y2 = det.bbox
                    poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

                poly = _sort_corners_clockwise(poly[:4])

                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # bbox YOLO
                xc = ((x_min + x_max) / 2.0) / ann.width
                yc = ((y_min + y_max) / 2.0) / ann.height
                bw = (x_max - x_min) / ann.width
                bh = (y_max - y_min) / ann.height

                xc, yc, bw, bh = map(_clamp01, (xc, yc, bw, bh))

                # keypoints (x,y) 4 szt.
                kpts = []
                for x, y in poly:
                    kpts.append(f"{_clamp01(x/ann.width):.6f}")
                    kpts.append(f"{_clamp01(y/ann.height):.6f}")

                line = f"{self.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} " + " ".join(kpts)
                lines.append(line)

            (labels_dir / f"{Path(ann.filename).stem}.txt").write_text("\n".join(lines), encoding="utf-8")

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