#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bazowa klasa annotatora.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Callable, Tuple
import threading

from ..config import CONFIG, logger, YOLO_AVAILABLE, resolve_runtime_device, CV2_AVAILABLE, cv2, PIL_AVAILABLE
from ..data_models import Detection, ImageAnnotation, AnnotationReport, AnnotationStatus
from ..utils import get_image_files, cleanup_gpu_memory, get_image_size


class BaseAnnotator(ABC):
    """Bazowa klasa dla annotatorów."""
    
    def __init__(self, 
                 confidence: float = 0.25,
                 device: str = "auto"):
        self.confidence = confidence
        self.device = resolve_runtime_device(device)
        self.report = AnnotationReport()
        
        self._stop_event = threading.Event()
    
    @abstractmethod
    def load_models(self) -> Tuple[bool, str]:
        """Ładuje modele."""
        pass
    
    @abstractmethod
    def unload_models(self):
        """Zwalnia modele."""
        pass
    
    @abstractmethod
    def process_image(self, image_path: Path) -> ImageAnnotation:
        """Przetwarza pojedynczy obraz."""
        pass
    
    def stop(self):
        """Zatrzymuje przetwarzanie."""
        logger.info("[STOP] Sygnał zatrzymania wysłany do annotatora")
        self._stop_event.set()
    
    def is_stopped(self) -> bool:
        """Sprawdza czy przetwarzanie zostało zatrzymane."""
        return self._stop_event.is_set()
    
    def reset_stop(self):
        """Resetuje flagę zatrzymania."""
        self._stop_event.clear()

    def _make_image_error_annotation(self, image_path: Path, message: str) -> ImageAnnotation:
        annotation = ImageAnnotation(
            filename=Path(image_path).name,
            width=0,
            height=0,
        )
        annotation.status = AnnotationStatus.ERROR
        annotation.status_message = str(message or "Nie można załadować obrazu")
        return annotation

    def _describe_image_read_error(self, image_path: Path) -> str:
        try:
            prefix = Path(image_path).read_bytes()[:128].lstrip().lower()
            if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
                return "Plik ma rozszerzenie obrazu, ale zawiera HTML zamiast danych obrazu"
        except Exception:
            pass
        return "Nie można załadować obrazu przed detekcją YOLO"

    def _read_image_for_yolo(self, image_path: Path):
        """
        Wstępnie waliduje obraz przed przekazaniem ścieżki do Ultralytics.

        YOLO potrafi zakończyć pojedynczy nieczytelny plik błędem
        `need at least one array to stack`; tutaj sprowadzamy to do statusu
        ERROR dla jednego obrazu i pozwalamy kontynuować cały run.
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists() or not image_path.is_file():
                return None
        except Exception:
            return None

        if CV2_AVAILABLE and cv2 is not None:
            try:
                image = cv2.imread(str(image_path))
                return image if image is not None else None
            except Exception:
                return None

        if PIL_AVAILABLE:
            try:
                from PIL import Image
                with Image.open(image_path) as image_obj:
                    image_obj.verify()
                return True
            except Exception:
                return None

        return True
    
    def _normalize_keypoints(self, raw_keypoints) -> list[tuple[float, float, float]]:
        """
        Normalizuje wynik keypointow YOLO do postaci (x, y, conf).

        Ultralytics potrafi zwracac keypointy jako (x, y) albo (x, y, conf).
        """
        normalized: list[tuple[float, float, float]] = []
        if raw_keypoints is None:
            return normalized

        for kp in raw_keypoints:
            try:
                if len(kp) < 2:
                    continue
                x = float(kp[0])
                y = float(kp[1])
                conf = float(kp[2]) if len(kp) >= 3 else 1.0
                normalized.append((x, y, conf))
            except Exception:
                continue

        return normalized

    @staticmethod
    def _bbox_iou(
        bbox_a: tuple[float, float, float, float],
        bbox_b: tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in bbox_b[:4]]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        denom = area_a + area_b - inter_area
        if denom <= 0.0:
            return 0.0
        return float(inter_area / denom)

    @staticmethod
    def _bbox_overlap_over_smaller(
        bbox_a: tuple[float, float, float, float],
        bbox_b: tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in bbox_b[:4]]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        smaller_area = min(area_a, area_b)
        if smaller_area <= 0.0:
            return 0.0
        return float(inter_area / smaller_area)

    def _suppress_overlapping_detections(
        self,
        detections: list[Detection],
        *,
        overlap_threshold: float = 0.80,
        iou_threshold: float = 0.55,
    ) -> list[Detection]:
        if len(detections or []) <= 1:
            return list(detections or [])

        ordered = sorted(
            list(detections or []),
            key=lambda det: (
                -float(getattr(det, "confidence", 0.0) or 0.0),
                -float(getattr(det, "get_area", lambda: 0.0)() or 0.0),
            ),
        )
        kept: list[Detection] = []
        for candidate in ordered:
            is_duplicate = False
            for existing in kept:
                if str(getattr(candidate, "label", "") or "").lower() != str(getattr(existing, "label", "") or "").lower():
                    continue
                overlap = self._bbox_overlap_over_smaller(candidate.bbox, existing.bbox)
                iou = self._bbox_iou(candidate.bbox, existing.bbox)
                if overlap >= float(overlap_threshold) or iou >= float(iou_threshold):
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(candidate)
        return kept

    def process_directory(self,
                          images_dir: Path,
                          progress_callback: Optional[Callable[..., None]] = None
                          ) -> Tuple[List[ImageAnnotation], AnnotationReport]:
        """Przetwarza wszystkie obrazy w folderze."""
        self.report = AnnotationReport()
        annotations = []
        
        self.reset_stop()
        
        image_files = get_image_files(images_dir)
        
        if not image_files:
            logger.warning(f"Brak obrazów w {images_dir}")
            return annotations, self.report
        
        logger.info(f"Przetwarzanie {len(image_files)} obrazów...")
        
        for i, img_path in enumerate(image_files):
            if self.is_stopped():
                logger.warning(f"[STOP] Przetwarzanie przerwane na obrazie {i+1}/{len(image_files)}")
                break
            
            ann = self.process_image(img_path)
            annotations.append(ann)
            self.report.add_result(ann)

            if progress_callback:
                try:
                    progress_callback(i + 1, len(image_files), img_path.name, self.report.successful)
                except TypeError:
                    progress_callback(i + 1, len(image_files), img_path.name)
        
        logger.info(f"Zakończono: {self.report.successful}/{self.report.total_images} udanych")
        
        return annotations, self.report
