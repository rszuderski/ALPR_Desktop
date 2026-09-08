#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detekcja znaków na tablicach (OCR + YOLO).
"""

from pathlib import Path
from typing import List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import numpy as np

from ..config import logger, CV2_AVAILABLE, cv2
from ..utils import cleanup_gpu_memory
from .reading_order import group_records_into_reading_rows, sort_records_reading_order


class DetectionMethod(Enum):
    OCR = "ocr"
    YOLO = "yolo"
    BOTH = "both"
    YOLO_OCR = "yolo_ocr"
    YOLO_BOX = "yolo_box"
    YOLO_SYMBOL = "yolo_symbol"


@dataclass
class CharacterDetection:
    character: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    method: str = "ocr"
    source_tag: str = ""
    
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def polygon(self) -> List[Tuple[float, float]]:
        x1, y1, x2, y2 = self.bbox
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    def to_dict(self) -> dict:
        return {
            'character': str(self.character),
            'bbox': [float(x) for x in self.bbox],
            'confidence': float(self.confidence),
            'method': str(self.method),
            'source_tag': str(self.source_tag),
        }


class CharacterDetector:
    def __init__(
        self,
        method: DetectionMethod = DetectionMethod.OCR,
        ocr_engine = None,
        yolo_model = None,
        yolo_device = None,
        yolo_confidence: float = 0.25,
        yolo_box_confidence: float | None = None,
        yolo_symbol_confidence: float | None = None,
        yolo_iou: float = 0.45,
        yolo_agnostic_nms: bool = False,
        yolo_overlap_threshold: float = 0.70,
        yolo_sequence_center_y_tolerance: float = 0.60,
        yolo_sequence_min_height_ratio: float = 0.55,
        yolo_sequence_max_height_ratio: float = 1.80,
        yolo_sequence_max_width_ratio: float = 2.60,
        yolo_sequence_soft_overlap: float = 0.18,
        yolo_sequence_hard_overlap: float = 0.30,
        ocr_min_height_ratio: float = 0.58,
    ):
        self.method = method
        self.ocr_engine = ocr_engine
        self.yolo_model = yolo_model
        self.yolo_device = yolo_device
        self.yolo_confidence = float(yolo_confidence)
        self.yolo_box_confidence = float(yolo_box_confidence if yolo_box_confidence is not None else yolo_confidence)
        self.yolo_symbol_confidence = float(yolo_symbol_confidence if yolo_symbol_confidence is not None else yolo_confidence)
        self.yolo_iou = float(yolo_iou)
        self.yolo_agnostic_nms = bool(yolo_agnostic_nms)
        self.yolo_overlap_threshold = float(yolo_overlap_threshold)
        self.yolo_sequence_center_y_tolerance = float(yolo_sequence_center_y_tolerance)
        self.yolo_sequence_min_height_ratio = float(yolo_sequence_min_height_ratio)
        self.yolo_sequence_max_height_ratio = float(yolo_sequence_max_height_ratio)
        self.yolo_sequence_max_width_ratio = float(yolo_sequence_max_width_ratio)
        self.yolo_sequence_soft_overlap = float(yolo_sequence_soft_overlap)
        self.yolo_sequence_hard_overlap = float(yolo_sequence_hard_overlap)
        try:
            self.ocr_min_height_ratio = max(0.20, min(1.00, float(ocr_min_height_ratio)))
        except Exception:
            self.ocr_min_height_ratio = 0.58
        self.last_ocr_detections: List[CharacterDetection] = []
        self.last_yolo_raw_detections: List[CharacterDetection] = []
        self.last_yolo_nms_detections: List[CharacterDetection] = []
        self.last_yolo_detections: List[CharacterDetection] = []
        self.last_yolo_ocr_detections: List[CharacterDetection] = []
        self.last_yolo_requested_device = yolo_device
        self.last_yolo_runtime_device = ""
        self.expected_character_count = 0
        self._cuda_runtime_broken = False
        self._cuda_runtime_break_reason = ""
        self._cuda_runtime_fallback_logged = False
    
    def detect(self, plate_image: np.ndarray) -> List[CharacterDetection]:
        detections = []
        self.last_ocr_detections = []
        self.last_yolo_raw_detections = []
        self.last_yolo_nms_detections = []
        self.last_yolo_detections = []
        self.last_yolo_ocr_detections = []

        if self.method in [DetectionMethod.OCR, DetectionMethod.BOTH]:
            self.last_ocr_detections = self._detect_with_ocr(plate_image)
            detections.extend(self.last_ocr_detections)
        if self.method in [
            DetectionMethod.YOLO,
            DetectionMethod.BOTH,
            DetectionMethod.YOLO_OCR,
            DetectionMethod.YOLO_BOX,
            DetectionMethod.YOLO_SYMBOL,
        ]:
            self.last_yolo_detections = self._detect_with_yolo(plate_image)
            if self.method == DetectionMethod.YOLO_OCR:
                yolo_boxes_for_ocr = (
                    list(self.last_yolo_nms_detections)
                    or list(self.last_yolo_detections)
                    or list(self.last_yolo_raw_detections)
                )
                if yolo_boxes_for_ocr:
                    self.last_yolo_ocr_detections = self._detect_with_yolo_boxes_and_ocr(
                        plate_image,
                        yolo_boxes_for_ocr,
                    )
                    detections.extend(self.last_yolo_ocr_detections)
                elif self._cuda_runtime_broken and self.ocr_engine is not None and getattr(self.ocr_engine, "is_loaded", False):
                    # Jeśli YOLO padło na GPU, nie zostawiaj reszty paczki bez żadnego wyniku.
                    self.last_ocr_detections = self._detect_with_ocr(plate_image)
                    detections.extend(self.last_ocr_detections)
            elif self.method == DetectionMethod.YOLO_BOX:
                detections.extend(
                    list(self.last_yolo_nms_detections)
                    or list(self.last_yolo_detections)
                    or list(self.last_yolo_raw_detections)
                )
            else:
                detections.extend(self.last_yolo_detections)
            
        return sort_records_reading_order(detections)

    @staticmethod
    def _is_cuda_runtime_error(error: Exception | str | None) -> bool:
        text = str(error or "").strip().lower()
        if not text:
            return False
        tokens = (
            "cuda error",
            "out of memory",
            "cuda out of memory",
            "cublas",
            "cudnn",
            "device-side assert",
            "device-side assertion",
            "illegal memory access",
            "no kernel image is available",
            "unspecified launch failure",
        )
        return any(token in text for token in tokens)

    def _ocr_runs_on_cuda(self) -> bool:
        engine = getattr(self, "ocr_engine", None)
        if engine is None:
            return False
        try:
            device = str(getattr(engine, "device", "") or "").strip().lower()
        except Exception:
            device = ""
        return device.startswith("cuda")

    def _handle_cuda_runtime_failure(self, backend: str, error: Exception) -> bool:
        if not self._is_cuda_runtime_error(error):
            return False

        self._cuda_runtime_broken = True
        self._cuda_runtime_break_reason = str(error or "").strip()

        if backend == "yolo":
            try:
                self.yolo_model = None
            except Exception:
                pass
            self.last_yolo_raw_detections = []
            self.last_yolo_nms_detections = []
            self.last_yolo_detections = []
            self.last_yolo_ocr_detections = []

        cpu_fallback_ready = False
        engine = getattr(self, "ocr_engine", None)
        if engine is not None and self._ocr_runs_on_cuda():
            try:
                cpu_fallback_ready = bool(engine.fallback_to_cpu())
            except Exception as fallback_error:
                logger.debug(f"Fallback OCR->CPU nie powiódł się: {fallback_error}")
                cpu_fallback_ready = False
            if not cpu_fallback_ready:
                try:
                    engine.unload()
                except Exception:
                    pass
                self.ocr_engine = None

        try:
            cleanup_gpu_memory()
        except Exception:
            pass

        if not self._cuda_runtime_fallback_logged:
            if cpu_fallback_ready:
                logger.warning(
                    "Detekcja znaków napotkała awarię CUDA. OCR przełączono na CPU, "
                    "a backend YOLO znaków zostanie pominięty do końca tego przebiegu."
                )
            else:
                logger.warning(
                    "Detekcja znaków napotkała awarię CUDA. Backendy GPU dla tego przebiegu "
                    "zostały wyłączone, aby zatrzymać lawinę błędów."
                )
            self._cuda_runtime_fallback_logged = True

        return cpu_fallback_ready

    @staticmethod
    def _refine_ocr_segment_bbox(
        processed_img: np.ndarray,
        *,
        seg_x1: float,
        seg_x2: float,
        y1: float,
        y2: float,
        scale_x: float,
        scale_y: float,
        orig_w: int,
        orig_h: int,
        min_height_ratio: float = 0.58,
    ) -> tuple[float, float, float, float] | None:
        if processed_img is None or getattr(processed_img, "size", 0) == 0:
            return None

        try:
            proc_h, proc_w = processed_img.shape[:2]
            px1 = max(0, min(proc_w, int(np.floor(float(seg_x1)))))
            px2 = max(0, min(proc_w, int(np.ceil(float(seg_x2)))))
            py1 = max(0, min(proc_h, int(np.floor(float(y1)))))
            py2 = max(0, min(proc_h, int(np.ceil(float(y2)))))
        except Exception:
            return None

        if px2 <= px1 or py2 <= py1:
            return None

        roi = processed_img[py1:py2, px1:px2]
        if roi is None or getattr(roi, "size", 0) == 0:
            return None

        try:
            if len(roi.shape) == 3:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi
        except Exception:
            roi_gray = roi if len(getattr(roi, "shape", ())) == 2 else None
        if roi_gray is None or getattr(roi_gray, "size", 0) == 0:
            return None

        # OCR preprocess zwykle kończy na jasnym tle i ciemnych znakach.
        # Szukamy realnego "atramentu" zamiast brać pełną wysokość wspólnego boxu.
        try:
            foreground_mask = roi_gray < 245
            rows = np.where(foreground_mask.any(axis=1))[0]
            cols = np.where(foreground_mask.any(axis=0))[0]
        except Exception:
            return None

        if len(rows) == 0 or len(cols) == 0:
            return None

        top = int(rows[0])
        bottom = int(rows[-1]) + 1
        left = int(cols[0])
        right = int(cols[-1]) + 1

        # Drobny margines, żeby nie obcinać końcówek znaków.
        pad_x = 1
        pad_y = 1
        top = max(0, top - pad_y)
        bottom = min(roi_gray.shape[0], bottom + pad_y)
        left = max(0, left - pad_x)
        right = min(roi_gray.shape[1], right + pad_x)

        # Pure OCR refines X tightly, but Y must stay comparable with the
        # EasyOCR text-line box. Otherwise glyphs such as Y/1 can become tiny
        # vertical boxes and break row/layout heuristics before YOLO is used.
        parent_height = max(1, int(roi_gray.shape[0]))
        try:
            safe_min_height_ratio = max(0.20, min(1.00, float(min_height_ratio)))
        except Exception:
            safe_min_height_ratio = 0.58
        min_height = max(4, int(round(parent_height * safe_min_height_ratio)))
        refined_height = max(1, int(bottom - top))
        if refined_height < min_height and parent_height >= min_height:
            center_y = (float(top) + float(bottom)) / 2.0
            top = int(round(center_y - (float(min_height) / 2.0)))
            bottom = top + min_height
            if top < 0:
                bottom = min(parent_height, bottom - top)
                top = 0
            if bottom > parent_height:
                top = max(0, top - (bottom - parent_height))
                bottom = parent_height

        refined_x1 = max(0.0, min(float(orig_w), float((px1 + left) * scale_x)))
        refined_x2 = max(0.0, min(float(orig_w), float((px1 + right) * scale_x)))
        refined_y1 = max(0.0, min(float(orig_h), float((py1 + top) * scale_y)))
        refined_y2 = max(0.0, min(float(orig_h), float((py1 + bottom) * scale_y)))

        if refined_x2 <= refined_x1:
            refined_x2 = min(float(orig_w), refined_x1 + 1.0)
        if refined_y2 <= refined_y1:
            refined_y2 = min(float(orig_h), refined_y1 + 1.0)

        return (refined_x1, refined_y1, refined_x2, refined_y2)

    def _run_ocr_detection_pass(self, plate_image: np.ndarray, *, allow_cpu_fallback: bool = True) -> List[CharacterDetection]:
        if self.ocr_engine is None or not getattr(self.ocr_engine, 'is_loaded', False):
            return []

        try:
            orig_h, orig_w = plate_image.shape[:2]

            prep_kwargs = getattr(self.ocr_engine, 'custom_prep_params', {})
            if hasattr(self.ocr_engine, 'preprocess_plate'):
                clean_kwargs = {k: v for k, v in prep_kwargs.items() if k != "padding_pct"}
                processed_img = self.ocr_engine.preprocess_plate(plate_image, **clean_kwargs)
            else:
                processed_img = plate_image

            proc_h, proc_w = processed_img.shape[:2]
            scale_x = orig_w / float(proc_w) if proc_w > 0 else 1.0
            scale_y = orig_h / float(proc_h) if proc_h > 0 else 1.0

            padding_pct = prep_kwargs.get("padding_pct", 20)
            pad_y = int(proc_h * (padding_pct / 100.0))
            pad_x = int(proc_w * (padding_pct / 100.0))

            if len(processed_img.shape) == 2:
                padded_img = cv2.copyMakeBorder(processed_img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=255)
            else:
                padded_img = cv2.copyMakeBorder(processed_img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            if hasattr(self.ocr_engine, 'read_text_aggressive'):
                results = self.ocr_engine.read_text_aggressive(padded_img)
            else:
                results = self.ocr_engine.reader.readtext(
                    padded_img,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    mag_ratio=1.5, text_threshold=0.2, link_threshold=0.4
                )

            detections = []

            for (bbox_ocr, text, conf) in results:
                threshold = getattr(self.ocr_engine, 'confidence_threshold', 0.15)
                if conf < threshold:
                    continue

                text_clean = "".join([c for c in text if c.isalnum()]).upper()
                if not text_clean:
                    continue

                x_coords = [p[0] - pad_x for p in bbox_ocr]
                y_coords = [p[1] - pad_y for p in bbox_ocr]

                x1_orig = int(min(x_coords) * scale_x)
                x2_orig = int(max(x_coords) * scale_x)
                y1_orig = int(min(y_coords) * scale_y)
                y2_orig = int(max(y_coords) * scale_y)

                x1 = max(0, min(orig_w, x1_orig))
                x2 = max(0, min(orig_w, x2_orig))
                y1 = max(0, min(orig_h, y1_orig))
                y2 = max(0, min(orig_h, y2_orig))

                if x2 <= x1:
                    x2 = x1 + 1
                if y2 <= y1:
                    y2 = y1 + 1

                char_width = (x2 - x1) / len(text_clean)
                for i, char in enumerate(text_clean):
                    char_x1 = x1 + (i * char_width)
                    char_x2 = char_x1 + char_width
                    refined_bbox = self._refine_ocr_segment_bbox(
                        processed_img,
                        seg_x1=(min(x_coords) + (i * ((max(x_coords) - min(x_coords)) / len(text_clean)))),
                        seg_x2=(min(x_coords) + ((i + 1) * ((max(x_coords) - min(x_coords)) / len(text_clean)))),
                        y1=min(y_coords),
                        y2=max(y_coords),
                        scale_x=scale_x,
                        scale_y=scale_y,
                        orig_w=orig_w,
                        orig_h=orig_h,
                        min_height_ratio=self.ocr_min_height_ratio,
                    )
                    if refined_bbox is None:
                        refined_bbox = (char_x1, y1, char_x2, y2)
                    det = CharacterDetection(character=char, bbox=refined_bbox, confidence=float(conf), method="ocr")
                    detections.append(det)

            return detections

        except Exception as e:
            logger.error(f"Błąd OCR detection: {e}")
            fallback_ready = self._handle_cuda_runtime_failure("ocr", e)
            if allow_cpu_fallback and fallback_ready and self.ocr_engine is not None and getattr(self.ocr_engine, "is_loaded", False):
                return self._run_ocr_detection_pass(plate_image, allow_cpu_fallback=False)
            return []
    
    def _detect_with_ocr(self, plate_image: np.ndarray) -> List[CharacterDetection]:
        if self.ocr_engine is None or not getattr(self.ocr_engine, 'is_loaded', False):
            return []
        detections = self._run_ocr_detection_pass(plate_image)
        if detections:
            return detections

        try:
            orig_h, orig_w = plate_image.shape[:2]
        except Exception:
            return detections

        # Fallback ratunkowy: jeśli crop ma "wysoki" profil (np. długi numer
        # błędnie wyeksportowany jak tablica square 256x128), kompresujemy go
        # pionowo i dajemy OCR drugi przebieg. Pomaga to odzyskać długie
        # tablice bez psucia zwykłej ścieżki.
        try:
            aspect_ratio = (float(orig_w) / float(orig_h)) if orig_h > 0 else 0.0
            if orig_h >= 96 and aspect_ratio <= 2.3:
                rescue_h = max(56, min(72, int(round(orig_h * 0.5))))
                rescue_img = cv2.resize(plate_image, (int(orig_w), int(rescue_h)), interpolation=cv2.INTER_CUBIC)
                rescue_detections = self._run_ocr_detection_pass(rescue_img)
                if rescue_detections:
                    scale_back_y = float(orig_h) / float(rescue_h)
                    normalized = []
                    for det in rescue_detections:
                        x1, y1, x2, y2 = det.bbox
                        normalized.append(
                            CharacterDetection(
                                character=str(det.character),
                                bbox=(float(x1), float(y1) * scale_back_y, float(x2), float(y2) * scale_back_y),
                                confidence=float(det.confidence),
                                method=str(det.method),
                                source_tag="ocr_tall_rescue",
                            )
                        )
                    return normalized
        except Exception as e:
            logger.debug(f"Fallback OCR tall rescue nie powiódł się: {e}")

        return detections
        
    def _detect_with_yolo(self, plate_image: np.ndarray) -> List[CharacterDetection]:
        if self.yolo_model is None:
            return []
        try:
            box_conf = max(0.00001, min(float(getattr(self, "yolo_box_confidence", self.yolo_confidence)), 1.0))
            symbol_conf = max(0.00001, min(float(getattr(self, "yolo_symbol_confidence", self.yolo_confidence)), 1.0))
            final_conf = symbol_conf
            conf = max(0.00001, min(float(getattr(self, "yolo_confidence", min(box_conf, symbol_conf))), 1.0))
            conf = min(conf, box_conf, symbol_conf)
            iou = max(0.01, min(float(self.yolo_iou), 0.99))
            predict_kwargs = {
                "conf": conf,
                "iou": iou,
                "agnostic_nms": bool(self.yolo_agnostic_nms),
                "verbose": False,
            }
            if self.yolo_device is not None:
                predict_kwargs["device"] = self.yolo_device
            self.last_yolo_requested_device = predict_kwargs.get("device", None)

            results = self.yolo_model(plate_image, **predict_kwargs)
            if not results or len(results) == 0: return []
            result = results[0]
            if not hasattr(result, 'boxes') or result.boxes is None: return []
            try:
                self.last_yolo_runtime_device = str(getattr(result.boxes.xyxy, "device", "") or "")
            except Exception:
                self.last_yolo_runtime_device = ""
            
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            detections = []
            for box, conf, cls_id in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(float, box)
                class_name = self.yolo_model.names.get(int(cls_id), str(cls_id)) if hasattr(self.yolo_model, 'names') else str(cls_id)
                det = CharacterDetection(character=class_name.upper(), bbox=(x1, y1, x2, y2), confidence=float(conf), method="yolo")
                detections.append(det)
            self.last_yolo_raw_detections = list(detections)
            box_candidates = [det for det in detections if float(getattr(det, "confidence", 0.0) or 0.0) >= box_conf]
            symbol_candidates = [det for det in detections if float(getattr(det, "confidence", 0.0) or 0.0) >= final_conf]
            box_deduplicated = self._suppress_overlapping_yolo_detections(box_candidates)
            symbol_deduplicated = self._suppress_overlapping_yolo_detections(symbol_candidates)
            self.last_yolo_nms_detections = list(box_deduplicated)
            return self._filter_yolo_sequence_consistency(symbol_deduplicated)
        except Exception as e:
            logger.error(f"Błąd YOLO detection na znakach: {e}")
            self._handle_cuda_runtime_failure("yolo", e)
            return []

    def detect_yolo_candidate_pass(
        self,
        plate_image: np.ndarray,
        *,
        confidence: float,
    ) -> dict:
        previous_confidence = self.yolo_confidence
        previous_box_confidence = getattr(self, "yolo_box_confidence", previous_confidence)
        previous_symbol_confidence = getattr(self, "yolo_symbol_confidence", previous_confidence)
        previous_raw = list(self.last_yolo_raw_detections or [])
        previous_nms = list(self.last_yolo_nms_detections or [])
        previous_filtered = list(self.last_yolo_detections or [])
        previous_requested_device = self.last_yolo_requested_device
        previous_runtime_device = self.last_yolo_runtime_device

        try:
            self.yolo_confidence = float(confidence)
            self.yolo_box_confidence = float(confidence)
            self.yolo_symbol_confidence = float(confidence)
            filtered = self._detect_with_yolo(plate_image)
            return {
                "raw": list(self.last_yolo_raw_detections or []),
                "nms": list(self.last_yolo_nms_detections or []),
                "filtered": list(filtered or []),
                "requested_device": self.last_yolo_requested_device,
                "runtime_device": self.last_yolo_runtime_device,
                "confidence": float(confidence),
            }
        finally:
            self.yolo_confidence = previous_confidence
            self.yolo_box_confidence = previous_box_confidence
            self.yolo_symbol_confidence = previous_symbol_confidence
            self.last_yolo_raw_detections = previous_raw
            self.last_yolo_nms_detections = previous_nms
            self.last_yolo_detections = previous_filtered
            self.last_yolo_requested_device = previous_requested_device
            self.last_yolo_runtime_device = previous_runtime_device

    def _expand_crop_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        image_shape,
        *,
        pad_x_ratio: float = 0.18,
        pad_y_ratio: float = 0.16,
        min_pad: int = 2,
    ) -> Tuple[int, int, int, int]:
        img_h, img_w = image_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        pad_x = max(int(round(width * float(pad_x_ratio))), int(min_pad))
        pad_y = max(int(round(height * float(pad_y_ratio))), int(min_pad))

        crop_x1 = max(0, int(np.floor(x1 - pad_x)))
        crop_y1 = max(0, int(np.floor(y1 - pad_y)))
        crop_x2 = min(int(img_w), int(np.ceil(x2 + pad_x)))
        crop_y2 = min(int(img_h), int(np.ceil(y2 + pad_y)))

        if crop_x2 <= crop_x1:
            crop_x2 = min(int(img_w), crop_x1 + 1)
        if crop_y2 <= crop_y1:
            crop_y2 = min(int(img_h), crop_y1 + 1)

        return crop_x1, crop_y1, crop_x2, crop_y2

    def _recognize_char_from_crop_with_ocr(self, char_image: np.ndarray, *, allow_cpu_fallback: bool = True) -> Tuple[str, float]:
        if self.ocr_engine is None or not getattr(self.ocr_engine, "is_loaded", False):
            return "", 0.0
        if char_image is None or getattr(char_image, "size", 0) == 0:
            return "", 0.0

        try:
            prep_kwargs = getattr(self.ocr_engine, "custom_prep_params", {}) or {}
            if hasattr(self.ocr_engine, "preprocess_plate"):
                clean_kwargs = {k: v for k, v in prep_kwargs.items() if k != "padding_pct"}
                processed = self.ocr_engine.preprocess_plate(char_image, **clean_kwargs)
            else:
                processed = char_image

            if processed is None or getattr(processed, "size", 0) == 0:
                return "", 0.0

            bg_value = 255 if len(processed.shape) == 2 else [255, 255, 255]
            proc_h, proc_w = processed.shape[:2]
            pad_y = max(2, int(round(proc_h * 0.16)))
            pad_x = max(2, int(round(proc_w * 0.20)))
            padded = cv2.copyMakeBorder(
                processed,
                pad_y,
                pad_y,
                pad_x,
                pad_x,
                cv2.BORDER_CONSTANT,
                value=bg_value,
            )

            if hasattr(self.ocr_engine, "read_text_aggressive"):
                results = self.ocr_engine.read_text_aggressive(padded)
            else:
                results = self.ocr_engine.reader.readtext(
                    padded,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    mag_ratio=2.0,
                    text_threshold=0.3,
                    link_threshold=0.6,
                    width_ths=0.8,
                    decoder="beamsearch",
                )
        except Exception as e:
            logger.debug(f"Błąd OCR dla cropa znaku: {e}")
            fallback_ready = self._handle_cuda_runtime_failure("ocr", e)
            if allow_cpu_fallback and fallback_ready and self.ocr_engine is not None and getattr(self.ocr_engine, "is_loaded", False):
                return self._recognize_char_from_crop_with_ocr(char_image, allow_cpu_fallback=False)
            return "", 0.0

        best_char = ""
        best_conf = 0.0
        threshold = float(getattr(self.ocr_engine, "confidence_threshold", 0.15) or 0.15)
        for result in list(results or []):
            try:
                _bbox, text, conf = result
            except Exception:
                continue

            clean_text = "".join(ch for ch in str(text or "").upper() if ch.isalnum())
            if not clean_text:
                continue

            candidate_char = clean_text[0]
            candidate_conf = float(conf or 0.0)
            if candidate_conf < threshold:
                continue
            if candidate_conf > best_conf:
                best_char = candidate_char
                best_conf = candidate_conf

        return best_char, best_conf

    def _detect_with_yolo_boxes_and_ocr(
        self,
        plate_image: np.ndarray,
        yolo_detections: List[CharacterDetection],
    ) -> List[CharacterDetection]:
        if not isinstance(yolo_detections, list) or not yolo_detections:
            return []
        if plate_image is None or getattr(plate_image, "size", 0) == 0:
            return []

        recognized = []
        for det in sort_records_reading_order(list(yolo_detections)):
            try:
                crop_x1, crop_y1, crop_x2, crop_y2 = self._expand_crop_bbox(det.bbox, plate_image.shape)
                crop = plate_image[crop_y1:crop_y2, crop_x1:crop_x2]
            except Exception:
                crop = None

            symbol, ocr_conf = self._recognize_char_from_crop_with_ocr(crop)
            recognized.append(
                CharacterDetection(
                    character=str(symbol or ""),
                    bbox=tuple(float(v) for v in det.bbox),
                    confidence=float(ocr_conf if symbol else 0.0),
                    method="ocr",
                    source_tag="yolo_box_ocr",
                )
            )

        return recognized

    def _bbox_area(self, bbox: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

    def _smaller_box_overlap(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        x1 = max(float(bbox1[0]), float(bbox2[0]))
        y1 = max(float(bbox1[1]), float(bbox2[1]))
        x2 = min(float(bbox1[2]), float(bbox2[2]))
        y2 = min(float(bbox1[3]), float(bbox2[3]))

        if x1 >= x2 or y1 >= y2:
            return 0.0

        inter_area = (x2 - x1) * (y2 - y1)
        smaller_area = min(self._bbox_area(bbox1), self._bbox_area(bbox2))
        if smaller_area <= 0.0:
            return 0.0

        return inter_area / smaller_area

    def _bbox_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        x1 = max(float(bbox1[0]), float(bbox2[0]))
        y1 = max(float(bbox1[1]), float(bbox2[1]))
        x2 = min(float(bbox1[2]), float(bbox2[2]))
        y2 = min(float(bbox1[3]), float(bbox2[3]))

        if x1 >= x2 or y1 >= y2:
            return 0.0

        inter_area = (x2 - x1) * (y2 - y1)
        area1 = self._bbox_area(bbox1)
        area2 = self._bbox_area(bbox2)
        union = area1 + area2 - inter_area
        if union <= 0.0:
            return 0.0

        return inter_area / union

    def _normalized_center_distance(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        cx1 = (float(bbox1[0]) + float(bbox1[2])) / 2.0
        cy1 = (float(bbox1[1]) + float(bbox1[3])) / 2.0
        cx2 = (float(bbox2[0]) + float(bbox2[2])) / 2.0
        cy2 = (float(bbox2[1]) + float(bbox2[3])) / 2.0

        width_norm = max(1.0, min(abs(float(bbox1[2]) - float(bbox1[0])), abs(float(bbox2[2]) - float(bbox2[0]))))
        height_norm = max(1.0, min(abs(float(bbox1[3]) - float(bbox1[1])), abs(float(bbox2[3]) - float(bbox2[1]))))

        return abs(cx1 - cx2) / width_norm, abs(cy1 - cy2) / height_norm

    def _looks_like_duplicate_detection(
        self,
        candidate: CharacterDetection,
        kept: CharacterDetection,
        threshold: float
    ) -> bool:
        overlap = self._smaller_box_overlap(candidate.bbox, kept.bbox)
        if overlap >= threshold:
            return True

        iou = self._bbox_iou(candidate.bbox, kept.bbox)
        dx_ratio, dy_ratio = self._normalized_center_distance(candidate.bbox, kept.bbox)

        close_centers = dx_ratio <= 0.35 and dy_ratio <= 0.45
        moderate_overlap = overlap >= max(0.18, threshold * 0.45)
        moderate_iou = iou >= max(0.12, threshold * 0.35)

        return close_centers and (moderate_overlap or moderate_iou)

    def _suppress_overlapping_yolo_detections(self, detections: List[CharacterDetection]) -> List[CharacterDetection]:
        threshold = max(0.0, min(float(self.yolo_overlap_threshold), 1.0))
        if threshold <= 0.0 or len(detections) <= 1:
            return detections

        ordered = sorted(
            detections,
            key=lambda det: (-float(det.confidence), float(det.bbox[0]), float(det.bbox[1]))
        )

        filtered: List[CharacterDetection] = []
        for candidate in ordered:
            skip_candidate = False
            for kept in filtered:
                if self._looks_like_duplicate_detection(candidate, kept, threshold):
                    skip_candidate = True
                    break
            if not skip_candidate:
                filtered.append(candidate)

        return sort_records_reading_order(filtered)

    def _detection_width(self, det: CharacterDetection) -> float:
        return max(1.0, float(det.bbox[2]) - float(det.bbox[0]))

    def _detection_height(self, det: CharacterDetection) -> float:
        return max(1.0, float(det.bbox[3]) - float(det.bbox[1]))

    def _detection_center_y(self, det: CharacterDetection) -> float:
        return (float(det.bbox[1]) + float(det.bbox[3])) / 2.0

    def _sequence_reference_subset_score(self, detections: List[CharacterDetection]) -> float:
        if not detections:
            return 0.0

        heights = np.array([self._detection_height(det) for det in detections], dtype=float)
        median_height = max(1.0, float(np.median(heights)))
        confidence_sum = sum(max(0.0, float(det.confidence)) for det in detections)
        return (float(len(detections)) * median_height) + (0.35 * confidence_sum)

    def _get_expected_character_count(self) -> int:
        try:
            return max(0, int(getattr(self, "expected_character_count", 0) or 0))
        except Exception:
            return 0

    def _select_sequence_best_count_candidates(
        self,
        detections: List[CharacterDetection],
        target_count: int,
        stats: dict | None = None,
    ) -> List[CharacterDetection]:
        ordered = sorted(list(detections or []), key=lambda det: float(det.bbox[0]))
        try:
            target = int(target_count)
        except Exception:
            target = 0
        if target <= 0 or len(ordered) <= target:
            return sort_records_reading_order(ordered)

        reference_stats = stats if isinstance(stats, dict) else self._build_sequence_reference_stats(ordered)
        ranked = sorted(
            enumerate(ordered),
            key=lambda item: (
                -self._sequence_candidate_score(item[1], reference_stats),
                float(item[1].bbox[0]),
            ),
        )
        selected_indices = sorted(idx for idx, _det in ranked[:target])
        return sort_records_reading_order([ordered[idx] for idx in selected_indices])

    def _collect_sequence_reference_cluster(
        self,
        anchor: CharacterDetection,
        detections: List[CharacterDetection]
    ) -> List[CharacterDetection]:
        anchor_height = max(1.0, self._detection_height(anchor))
        anchor_center_y = self._detection_center_y(anchor)
        cluster: List[CharacterDetection] = []

        for det in detections:
            det_height = max(1.0, self._detection_height(det))
            det_center_y = self._detection_center_y(det)
            height_similarity = min(anchor_height, det_height) / max(anchor_height, det_height)
            center_y_gap = abs(det_center_y - anchor_center_y) / max(anchor_height, det_height, 1.0)

            if height_similarity >= self.yolo_sequence_min_height_ratio and center_y_gap <= self.yolo_sequence_center_y_tolerance:
                cluster.append(det)

        return cluster

    def _select_sequence_reference_detections(
        self,
        detections: List[CharacterDetection]
    ) -> List[CharacterDetection]:
        if len(detections) <= 2:
            return detections

        anchors = sorted(
            detections,
            key=lambda det: (
                -float(det.confidence),
                -self._detection_height(det),
                float(det.bbox[0]),
            )
        )

        best_subset = list(detections)
        best_score = self._sequence_reference_subset_score(best_subset)

        for anchor in anchors:
            cluster = self._collect_sequence_reference_cluster(anchor, detections)
            if not cluster:
                continue

            cluster_score = self._sequence_reference_subset_score(cluster)
            if cluster_score > best_score + 1e-6:
                best_subset = cluster
                best_score = cluster_score

        return sort_records_reading_order(best_subset)

    def _build_sequence_reference_stats(self, detections: List[CharacterDetection]) -> dict:
        if not detections:
            return {
                "median_width": 1.0,
                "median_height": 1.0,
                "median_center_y": 0.0,
                "reference_count": 0,
            }

        reference_subset = self._select_sequence_reference_detections(detections)
        widths = np.array([self._detection_width(det) for det in reference_subset], dtype=float)
        heights = np.array([self._detection_height(det) for det in reference_subset], dtype=float)
        centers_y = np.array([self._detection_center_y(det) for det in reference_subset], dtype=float)

        return {
            "median_width": max(1.0, float(np.median(widths))),
            "median_height": max(1.0, float(np.median(heights))),
            "median_center_y": float(np.median(centers_y)),
            "reference_count": int(len(reference_subset)),
        }

    def _is_sequence_geometry_outlier(
        self,
        det: CharacterDetection,
        stats: dict
    ) -> bool:
        width = self._detection_width(det)
        height = self._detection_height(det)
        center_y = self._detection_center_y(det)

        median_width = max(1.0, float(stats.get("median_width", 1.0)))
        median_height = max(1.0, float(stats.get("median_height", 1.0)))
        median_center_y = float(stats.get("median_center_y", center_y))

        width_ratio = width / median_width
        height_ratio = height / median_height
        center_y_offset = abs(center_y - median_center_y) / median_height

        if center_y_offset > self.yolo_sequence_center_y_tolerance:
            return True
        if height_ratio < self.yolo_sequence_min_height_ratio or height_ratio > self.yolo_sequence_max_height_ratio:
            return True
        if width_ratio > self.yolo_sequence_max_width_ratio:
            return True
        if width_ratio < 0.12 and float(det.confidence) < 0.60:
            return True

        return False

    def _sequence_candidate_score(self, det: CharacterDetection, stats: dict) -> float:
        confidence = float(det.confidence)
        width = self._detection_width(det)
        height = self._detection_height(det)
        center_y = self._detection_center_y(det)

        median_width = max(1.0, float(stats.get("median_width", 1.0)))
        median_height = max(1.0, float(stats.get("median_height", 1.0)))
        median_center_y = float(stats.get("median_center_y", center_y))

        center_penalty = min(1.5, abs(center_y - median_center_y) / median_height)
        height_penalty = min(1.5, abs(height - median_height) / median_height)

        width_ratio = width / median_width
        width_penalty = 0.0
        if width_ratio > 1.90:
            width_penalty += min(1.5, width_ratio - 1.90)
        if width_ratio < 0.18:
            width_penalty += min(1.0, (0.18 - width_ratio) / 0.18)

        return confidence - (0.20 * center_penalty) - (0.12 * height_penalty) - (0.06 * width_penalty)

    def _sequence_neighbor_overlap_ratio(
        self,
        left: CharacterDetection,
        right: CharacterDetection
    ) -> float:
        overlap = float(left.bbox[2]) - float(right.bbox[0])
        if overlap <= 0.0:
            return 0.0

        min_width = max(1.0, min(self._detection_width(left), self._detection_width(right)))
        return overlap / min_width

    def _looks_like_sequence_conflict(
        self,
        left: CharacterDetection,
        right: CharacterDetection,
        stats: dict
    ) -> bool:
        overlap_ratio = self._sequence_neighbor_overlap_ratio(left, right)
        if overlap_ratio <= 0.0:
            return False

        median_height = max(1.0, float(stats.get("median_height", 1.0)))
        center_y_gap = abs(self._detection_center_y(left) - self._detection_center_y(right)) / median_height

        left_height = self._detection_height(left)
        right_height = self._detection_height(right)
        height_similarity = min(left_height, right_height) / max(left_height, right_height)

        if overlap_ratio >= self.yolo_sequence_hard_overlap:
            return True

        return (
            overlap_ratio >= self.yolo_sequence_soft_overlap
            and center_y_gap <= self.yolo_sequence_center_y_tolerance
            and height_similarity >= self.yolo_sequence_min_height_ratio
        )

    def _filter_yolo_sequence_consistency_single_row(self, detections: List[CharacterDetection]) -> List[CharacterDetection]:
        if len(detections) <= 1:
            return detections

        ordered = sorted(detections, key=lambda det: float(det.bbox[0]))
        expected_count = self._get_expected_character_count()
        initial_stats = self._build_sequence_reference_stats(ordered)

        geometry_filtered = [
            det for det in ordered
            if not self._is_sequence_geometry_outlier(det, initial_stats)
        ]
        if 0 < expected_count <= len(ordered) and len(geometry_filtered) < expected_count:
            geometry_filtered = self._select_sequence_best_count_candidates(
                ordered,
                expected_count,
                initial_stats,
            )

        if len(geometry_filtered) <= 1:
            return geometry_filtered

        stats = self._build_sequence_reference_stats(geometry_filtered)
        filtered: List[CharacterDetection] = []

        for candidate in geometry_filtered:
            candidate_kept = True

            while filtered and self._looks_like_sequence_conflict(filtered[-1], candidate, stats):
                prev = filtered[-1]
                prev_score = self._sequence_candidate_score(prev, stats)
                candidate_score = self._sequence_candidate_score(candidate, stats)

                if candidate_score > prev_score + 1e-6:
                    filtered.pop()
                    continue

                candidate_kept = False
                break

            if candidate_kept:
                filtered.append(candidate)

        if 0 < expected_count <= len(geometry_filtered) and len(filtered) < expected_count:
            return self._select_sequence_best_count_candidates(
                geometry_filtered,
                expected_count,
                stats,
            )
        return filtered

    def _filter_yolo_sequence_consistency(self, detections: List[CharacterDetection]) -> List[CharacterDetection]:
        if len(detections) <= 1:
            return detections
        expected_count = self._get_expected_character_count()

        rows = group_records_into_reading_rows(detections)
        if len(rows) <= 1:
            return sort_records_reading_order(self._filter_yolo_sequence_consistency_single_row(detections))

        filtered: List[CharacterDetection] = []
        for row in rows:
            filtered.extend(self._filter_yolo_sequence_consistency_single_row(list(row)))
        if 0 < expected_count <= len(detections) and len(filtered) < expected_count:
            return self._select_sequence_best_count_candidates(
                sort_records_reading_order(list(detections)),
                expected_count,
            )
        return sort_records_reading_order(filtered)
