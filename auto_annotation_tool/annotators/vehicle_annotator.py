#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotator pojazdów (Tryb A).
"""

from pathlib import Path
from typing import Tuple, Optional

from ..config import CONFIG, logger, YOLO_AVAILABLE, get_yolo_class
from ..data_models import Detection, ImageAnnotation, AnnotationStatus
from ..utils import get_image_size, cleanup_gpu_memory
from .base import BaseAnnotator


class VehicleAnnotator(BaseAnnotator):
    """
    Annotator wykrywający tylko pojazdy.
    
    Tryb A: Używa modelu YOLO detect (COCO lub własnego).
    Wyjście: <box label="vehicle">
    """
    
    # Klasy pojazdów w COCO
    COCO_VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
    
    def __init__(self,
                 model_path: Path,
                 confidence: float = 0.25,
                 device: str = "auto"):
        super().__init__(confidence, device)
        self.model_path = Path(model_path)
        self.model: Optional[object] = None
        self.class_names = {}
    
    def load_models(self) -> Tuple[bool, str]:
        """Ładuje model detekcji pojazdów."""
        if not YOLO_AVAILABLE:
            return False, "YOLO niedostępny"
        YoloClass = get_yolo_class()
        if YoloClass is None:
            return False, "YOLO niedostępny"
        
        try:
            logger.info(f"Ładowanie modelu pojazdów: {self.model_path}")
            self.model = YoloClass(str(self.model_path))
            
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            
            return True, "Model załadowany"
            
        except Exception as e:
            return False, f"Błąd: {e}"
    
    def unload_models(self):
        """Zwalnia model."""
        if self.model:
            del self.model
            self.model = None
        cleanup_gpu_memory()
    
    def process_image(self, image_path: Path) -> ImageAnnotation:
        """Wykrywa pojazdy na obrazie."""
        width, height = get_image_size(image_path)
        
        annotation = ImageAnnotation(
            filename=image_path.name,
            width=width,
            height=height
        )
        
        try:
            image_source = self._read_image_for_yolo(image_path)
            if image_source is None:
                return self._make_image_error_annotation(
                    image_path,
                    self._describe_image_read_error(image_path),
                )
            if image_source is True:
                image_source = str(image_path)

            results = self.model(
                image_source,
                conf=self.confidence,
                device=self.device,
                verbose=False
            )
            
            if not results or len(results) == 0 or results[0].boxes is None:
                annotation.status = AnnotationStatus.NO_VEHICLE
                annotation.status_message = "Nie wykryto żadnego pojazdu"
                return annotation
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confs, classes):
                # Sprawdź czy to pojazd
                class_name = self.class_names.get(cls_id, "").lower()
                is_vehicle = (cls_id in self.COCO_VEHICLE_CLASSES or 
                             class_name in CONFIG.VEHICLE_LABELS)
                
                if is_vehicle:
                    x1, y1, x2, y2 = map(float, box)
                    annotation.detections.append(Detection(
                        label="vehicle",
                        confidence=float(conf),
                        bbox=(x1, y1, x2, y2)
                    ))
            
            if not annotation.detections:
                annotation.status = AnnotationStatus.NO_VEHICLE
                annotation.status_message = "Nie wykryto pojazdów (znaleziono inne obiekty)"
            else:
                annotation.status = AnnotationStatus.SUCCESS
                annotation.status_message = f"Wykryto {len(annotation.detections)} pojazdów"
            
            return annotation
            
        except Exception as e:
            annotation.status = AnnotationStatus.ERROR
            annotation.status_message = str(e)
            return annotation
