#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotator tablic (Tryb B) - z OCR dla rozpoznawania znaków.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from ..config import CONFIG, logger, YOLO_AVAILABLE, get_yolo_class, CV2_AVAILABLE, cv2
from ..data_models import Detection, ImageAnnotation, AnnotationStatus
from ..quality_metrics import compute_plate_polygon_fit_metrics
from ..utils import get_image_size, cleanup_gpu_memory
from ..rectification import PlateRectifier
from ..ocr import PlateOCR
from .base import BaseAnnotator


class PlateAnnotator(BaseAnnotator):
    """
    Annotator wykrywający tablice rejestracyjne + OCR.
    
    Tryb B: 
    - Używa modelu YOLO Pose z 4 keypointami
    - Prostuje tablice (PlateRectifier)
    - Rozpoznaje znaki (PlateOCR)
    
    Wyjście: <polygon label="plate" points="x1,y1;x2,y2;x3,y3;x4,y4">
                 <text>ABC 1234</text>
                 <attributes>
                     <attribute name="format">PL</attribute>
                     <attribute name="confidence">0.85</attribute>
                 </attributes>
             </polygon>
    """
    
    def __init__(self,
                 model_path: Path,
                 confidence: float = 0.25,
                 device: str = "auto",
                 enable_ocr: bool = False,
                 ocr_confidence_threshold: float = 0.3,
                 enable_rectification: bool = True):
        super().__init__(confidence, device)
        self.model_path = Path(model_path)
        self.model: Optional[object] = None
        self.is_pose_model = False
        
        # OCR
        self.enable_ocr = enable_ocr
        self.ocr_engine: Optional[PlateOCR] = None
        self.ocr_confidence_threshold = ocr_confidence_threshold
        
        # Rektyfikacja
        self.enable_rectification = enable_rectification
        self.rectifier = PlateRectifier
    
    def load_models(self) -> Tuple[bool, str]:
        """Ładuje modele: YOLO + OCR."""
        if not YOLO_AVAILABLE:
            return False, "YOLO niedostępny"
        YoloClass = get_yolo_class()
        if YoloClass is None:
            return False, "YOLO niedostępny"
        
        try:
            # Załaduj YOLO
            logger.info(f"Ładowanie modelu tablic: {self.model_path}")
            self.model = YoloClass(str(self.model_path))
            
            # Sprawdź, czy model zwraca keypointy.
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'kpt_shape'):
                self.is_pose_model = True
                kpt_shape = self.model.model.kpt_shape
                logger.info(f"[OK] Model POSE (keypoints: {kpt_shape})")
            else:
                logger.warning("[WARN] Model nie jest typu POSE - użyję bbox jako polygon")
            
            # Załaduj OCR, jeśli jest włączony.
            if self.enable_ocr:
                try:
                    logger.info("Ładowanie engine OCR...")
                    self.ocr_engine = PlateOCR(
                        device=self.device,
                        confidence_threshold=self.ocr_confidence_threshold
                    )
                    if self.ocr_engine.is_loaded:
                        logger.info("[OK] OCR engine załadowany")
                    else:
                        logger.warning("[WARN] OCR engine nie załadował się - będzie pominięty")
                        self.enable_ocr = False
                except Exception as e:
                    logger.warning(f"[WARN] Błąd ładowania OCR: {e} - będzie pominięty")
                    self.enable_ocr = False
            
            return True, "Modele załadowane"
            
        except Exception as e:
            return False, f"Błąd: {e}"
    
    def unload_models(self):
        """Zwalnia modele."""
        if self.model:
            del self.model
            self.model = None
        
        if self.ocr_engine:
            self.ocr_engine.unload()
            self.ocr_engine = None
        
        cleanup_gpu_memory()
    
    def _extract_plate_region(self, 
                              image: np.ndarray,
                              polygon: List[Tuple[float, float]]
                              ) -> Optional[np.ndarray]:
        """
        Ekstraktuje region tablicy z obrazu.
        
        Args:
            image: Obraz (BGR)
            polygon: 4 rogi tablicy
            
        Returns:
            Obraz wyciętej tablicy lub None
        """
        if not CV2_AVAILABLE or len(polygon) < 4:
            return None
        
        try:
            pts = np.array(polygon[:4], dtype=np.float32)
            
            # Wylicz rozmiar wyjściowy
            width, height = self.rectifier.polygon_wh_px(polygon)
            width, height = max(int(width), 50), max(int(height), 20)
            
            # Rektyfikuj tablicę
            rectified = self.rectifier.rectify(
                image,
                polygon[:4],
                out_w_px=width,
                out_h_px=height,
                interpolation="lanczos4",
                enhance_contrast=True,
                do_deskew=True
            )
            
            return rectified
        
        except Exception as e:
            logger.debug(f"Błąd ekstrakcji: {e}")
            return None
    
    def _recognize_plate_text(self, 
                              plate_image: np.ndarray
                              ) -> Tuple[Optional[str], float, dict]:
        """
        Rozpoznaje tekst na tablicy.
        
        Args:
            plate_image: Obraz tablicy
            
        Returns:
            Tuple[tekst, pewność, atrybuty]
        """
        if not self.enable_ocr or self.ocr_engine is None:
            return None, 0.0, {}
        
        try:
            # Przygotuj obraz dla OCR.
            processed = self.ocr_engine.preprocess_plate(
                plate_image,
                enhance_contrast=True,
                enhance_sharpness=True
            )
            
            # Wykonaj OCR wraz z metadanymi walidacji.
            result = self.ocr_engine.recognize(processed, return_details=True)
            
            if result is None:
                return None, 0.0, {}
            
            text, ocr_conf, validation = result
            
            # Zapisz atrybuty pomocnicze OCR.
            attributes = {
                'format': validation.format.value,
                'ocr_confidence': f"{ocr_conf:.2f}",
                'is_valid': str(validation.is_valid),
            }
            
            # Dla poprawnego formatu zwróć tekst po normalizacji.
            if validation.is_valid:
                return validation.text, ocr_conf, attributes
            else:
                # Dla niepewnego formatu zwróć surowy wynik z obniżoną pewnością.
                logger.warning(f"[WARN] Tablica ma nieznany format: {text}")
                return text, ocr_conf * 0.7, attributes  # Obniż confidence
        
        except Exception as e:
            logger.debug(f"Błąd OCR: {e}")
            return None, 0.0, {}
    
    def process_image(self, image_path: Path) -> ImageAnnotation:
        """Wykrywa tablice + rozpoznaje znaki."""
        
        if self.is_stopped():
            return ImageAnnotation(
                filename=image_path.name,
                width=0,
                height=0,
                status=AnnotationStatus.SKIPPED,
                status_message="Przetwarzanie przerwane"
            )
        
        width, height = get_image_size(image_path)
        
        annotation = ImageAnnotation(
            filename=image_path.name,
            width=width,
            height=height
        )
        
        try:
            if not CV2_AVAILABLE:
                annotation.status = AnnotationStatus.ERROR
                annotation.status_message = "OpenCV niedostępny"
                return annotation
            
            image = self._read_image_for_yolo(image_path)
            if image is None:
                annotation.width = 0
                annotation.height = 0
                annotation.status = AnnotationStatus.ERROR
                annotation.status_message = self._describe_image_read_error(image_path)
                return annotation
            yolo_source = str(image_path) if image is True else image
            
            results = self.model(
                yolo_source,
                conf=self.confidence,
                device=self.device,
                verbose=False
            )
            
            if not results or len(results) == 0 or results[0].boxes is None:
                annotation.status = AnnotationStatus.NO_PLATE
                annotation.status_message = "Nie wykryto żadnej tablicy"
                return annotation
            
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            keypoints = None
            if self.is_pose_model and hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confs)):
                x1, y1, x2, y2 = map(float, box)
                
                polygon = None
                kpts_list = None
                ocr_text = None
                ocr_conf = 0.0
                ocr_attrs = {}
                
                # Zbuduj poligon z keypointów, jeśli model je zwraca.
                if keypoints is not None and i < len(keypoints):
                    kpts = keypoints[i]
                    kpts_list = self._normalize_keypoints(kpts)
                    
                    if len(kpts_list) >= 4:
                        corners = [(float(kpts_list[j][0]), float(kpts_list[j][1])) for j in range(4)]
                        
                        # Odrzuć punkty poza granicami obrazu.
                        valid = all(0 <= p[0] <= width and 0 <= p[1] <= height for p in corners)
                        
                        if valid:
                            polygon = self._sort_corners_clockwise(corners)
                
                # W razie braku keypointów użyj prostokąta z bboxa.
                if polygon is None:
                    polygon = [
                        (x1, y1), (x2, y1), (x2, y2), (x1, y2)
                    ]
                
                # Wytnij tablicę i uruchom OCR.
                if self.enable_ocr and self.ocr_engine and self.ocr_engine.is_loaded:
                    plate_region = self._extract_plate_region(image, polygon)
                    
                    if plate_region is not None:
                        ocr_text, ocr_conf, ocr_attrs = self._recognize_plate_text(plate_region)
                        
                        if ocr_text:
                            logger.debug(f"OCR: {ocr_text} (conf: {ocr_conf:.2f})")
                
                detection = Detection(
                    label="plate",
                    confidence=float(conf),
                    bbox=(x1, y1, x2, y2),
                    keypoints=kpts_list,
                    polygon=polygon,
                    text=ocr_text,
                    text_confidence=ocr_conf,
                    attributes=ocr_attrs
                )
                try:
                    fit_metrics = compute_plate_polygon_fit_metrics(
                        float(conf),
                        polygon,
                        (x1, y1, x2, y2),
                        keypoints=kpts_list,
                        image_size=(width, height),
                    )
                    detection.attributes.update({
                        "fit_score": f"{float(fit_metrics.get('fit_score', 0.0) or 0.0):.3f}",
                        "fit_label": str(fit_metrics.get("fit_label") or "").strip(),
                        "fit_keypoint_score": f"{float(fit_metrics.get('keypoint_score', 0.0) or 0.0):.3f}",
                        "fit_shape_score": f"{float(fit_metrics.get('shape_score', 0.0) or 0.0):.3f}",
                        "fit_bbox_score": f"{float(fit_metrics.get('bbox_alignment_score', 0.0) or 0.0):.3f}",
                        "fit_size_score": f"{float(fit_metrics.get('size_score', 0.0) or 0.0):.3f}",
                    })
                except Exception:
                    pass
                
                annotation.detections.append(detection)

            deduplicated = self._suppress_overlapping_detections(
                annotation.detections,
                overlap_threshold=0.82,
                iou_threshold=0.58,
            )
            if len(deduplicated) != len(annotation.detections):
                logger.debug(
                    f"[PlateAnnotator] Odrzucono {len(annotation.detections) - len(deduplicated)} nakładających się detekcji tablic dla {image_path.name}."
                )
                annotation.detections = deduplicated
            
            if annotation.detections:
                annotation.status = AnnotationStatus.SUCCESS
                annotation.status_message = f"Wykryto {len(annotation.detections)} tablic"
            else:
                annotation.status = AnnotationStatus.NO_PLATE
            
            return annotation
            
        except Exception as e:
            logger.exception(f"Błąd przetwarzania: {e}")
            annotation.status = AnnotationStatus.ERROR
            annotation.status_message = str(e)
            return annotation
    
    def _sort_corners_clockwise(self, corners: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Sortuje 4 rogi: TL, TR, BR, BL."""
        if len(corners) != 4:
            return corners
        
        cx = sum(p[0] for p in corners) / 4
        cy = sum(p[1] for p in corners) / 4
        
        top = [p for p in corners if p[1] < cy]
        bottom = [p for p in corners if p[1] >= cy]
        
        if len(top) != 2 or len(bottom) != 2:
            sorted_by_y = sorted(corners, key=lambda p: p[1])
            top = sorted(sorted_by_y[:2], key=lambda p: p[0])
            bottom = sorted(sorted_by_y[2:], key=lambda p: p[0], reverse=True)
        else:
            top = sorted(top, key=lambda p: p[0])
            bottom = sorted(bottom, key=lambda p: p[0], reverse=True)
        
        return [top[0], top[1], bottom[0], bottom[1]]
