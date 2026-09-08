#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anotator znaków na tablicach.
"""

from pathlib import Path
from typing import List, Optional, Dict, Callable, Tuple
import numpy as np

from ..config import logger, CV2_AVAILABLE, cv2
from ..data_models import Detection, ImageAnnotation
from .char_detector import CharacterDetector, CharacterDetection, DetectionMethod


class CharacterAnnotation:
    """Anotacja znaków dla jednej tablicy."""
    
    def __init__(self, 
                 plate_path: Path,
                 plate_detection: Detection,
                 characters: List[CharacterDetection]):
        self.plate_path = plate_path
        self.plate_detection = plate_detection
        self.characters = characters
    
    def to_dict(self) -> dict:
        return {
            'plate_path': str(self.plate_path),
            'plate_id': self.plate_detection.plate_id if hasattr(self.plate_detection, 'plate_id') else None,
            'plate_text': self.plate_detection.text,
            'num_characters': len(self.characters),
            'characters': [c.to_dict() for c in self.characters],
        }


class CharacterAnnotator:
    """
    Tworzy anotacje znaków na tablicach.
    Używa CharacterDetector (OCR/YOLO) do lokalizacji znaków.
    """
    
    def __init__(self,
                 detector: CharacterDetector,
                 save_visualizations: bool = False):
        """
        Args:
            detector: CharacterDetector instance
            save_visualizations: Czy zapisywać wizualizacje?
        """
        self.detector = detector
        self.save_visualizations = save_visualizations
    
    def annotate_plate(self, 
                      plate_image: np.ndarray,
                      plate_path: Path,
                      plate_detection: Detection
                      ) -> Optional[CharacterAnnotation]:
        """
        Anotuje znaki na pojedynczej tablicy.
        
        Args:
            plate_image: Obraz tablicy (BGR)
            plate_path: Ścieżka do pliku tablicy
            plate_detection: Detection obiektu tablicy
            
        Returns:
            CharacterAnnotation lub None
        """
        try:
            # Detekcja znaków
            characters = self.detector.detect(plate_image)
            
            if not characters:
                logger.debug(f"Brak znaków na tablicy: {plate_path.name}")
                return None
            
            # Wizualizacja (opcjonalnie)
            if self.save_visualizations:
                self._visualize(plate_image, characters, plate_path)
            
            annotation = CharacterAnnotation(
                plate_path=plate_path,
                plate_detection=plate_detection,
                characters=characters
            )
            
            logger.debug(f"[OK] Anotowano {len(characters)} znaków w {plate_path.name}")
            return annotation
        
        except Exception as e:
            logger.error(f"Błąd anotacji tablicy: {e}")
            return None
    
    def annotate_plates_batch(self,
                             plates: List[Tuple[Path, np.ndarray, Detection]],
                             progress_callback: Optional[Callable] = None
                             ) -> List[CharacterAnnotation]:
        """
        Anotuje wiele tablic.
        
        Args:
            plates: List[(plate_path, plate_image, plate_detection)]
            progress_callback: (current, total, name) -> None
            
        Returns:
            Lista CharacterAnnotation
        """
        annotations = []
        
        for idx, (plate_path, plate_image, plate_detection) in enumerate(plates):
            if progress_callback:
                progress_callback(idx + 1, len(plates), plate_path.name)
            
            ann = self.annotate_plate(plate_image, plate_path, plate_detection)
            if ann:
                annotations.append(ann)
        
        return annotations
    
    def _visualize(self, 
                   plate_image: np.ndarray,
                   characters: List[CharacterDetection],
                   plate_path: Path):
        """Rysuje znaki na tablicy (dla debugu)."""
        if not CV2_AVAILABLE:
            return
        
        try:
            vis = plate_image.copy()
            
            for char_det in characters:
                x1, y1, x2, y2 = char_det.bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Prostokąt
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Tekst
                cv2.putText(
                    vis,
                    char_det.character,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            
            # Zapisz
            vis_path = plate_path.parent / f"{plate_path.stem}_vis.jpg"
            cv2.imwrite(str(vis_path), vis)
            
        except Exception as e:
            logger.debug(f"Błąd wizualizacji: {e}")
