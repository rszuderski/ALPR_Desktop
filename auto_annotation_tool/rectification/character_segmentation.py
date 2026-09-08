#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentacja znaków z tablicy rejestracyjnej.
"""

import cv2
import numpy as np
from typing import List, Tuple
from ..config import logger


class CharacterSegmenter:
    """Segmentuje znaki z wyrównanej tablicy rejestracyjnej."""
    
    @staticmethod
    def segment_characters(
        image: np.ndarray,
        num_segments: int = 8,
        min_width: int = 5,
        threshold_value: int = 127
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Podziel tablicę na segmenty zawierające znaki.
        
        Args:
            image: Obraz tablicy (BGR)
            num_segments: Liczba oczekiwanych segmentów/znaków (domyślnie 8)
            min_width: Minimalna szerokość segmentu
            threshold_value: Próg binaryzacji (0-255)
        
        Returns:
            Tuple zawierający:
            - Lista obrazów segmentów (każdy segment = jeden znak)
            - Lista współrzędnych (x, y, width, height) dla każdego segmentu
        """
        if image is None or image.size == 0:
            logger.warning("Pusty obraz do segmentacji")
            return [], []
        
        # Konwertuj do skali szarości
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Binaryzacja
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Inweruj (znaki to białe obszary)
        binary = cv2.bitwise_not(binary)
        
        # Morfologia - czyść szumy
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Znajdź kontury (znaki)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtruj i sortuj kontury po X
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filtruj zbyt małe obiekty
            if w >= min_width and h >= min_width:
                valid_contours.append((x, y, w, h))
        
        # Sortuj po współrzędnej X (od lewej do prawej)
        valid_contours.sort(key=lambda c: c[0])
        
        # Jeśli znaleziono za dużo/mało segmentów, spróbuj prostej dzielenia
        if len(valid_contours) != num_segments:
            logger.warning(
                f"Znaleziono {len(valid_contours)} segmentów, "
                f"oczekiwano {num_segments}. Używam podziału stałego."
            )
            return CharacterSegmenter._segment_by_width(image, num_segments)
        
        # Ekstrahuj segmenty
        segments = []
        bboxes = []
        
        for x, y, w, h in valid_contours:
            # Dodaj margines
            margin = 2
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            
            segment = image[y1:y2, x1:x2].copy()
            segments.append(segment)
            bboxes.append((x1, y1, x2 - x1, y2 - y1))
        
        logger.info(f"Segmentacja: znaleziono {len(segments)} znaków")
        return segments, bboxes
    
    @staticmethod
    def _segment_by_width(image: np.ndarray, num_segments: int) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Podziel obraz na stałe segmenty (jeśli automatyczna segmentacja nie zadziała).
        
        Args:
            image: Obraz tablicy
            num_segments: Liczba segmentów
        
        Returns:
            Tuple segmentów i ich współrzędnych
        """
        height, width = image.shape[:2]
        segment_width = width // num_segments
        
        segments = []
        bboxes = []
        
        for i in range(num_segments):
            x1 = i * segment_width
            x2 = (i + 1) * segment_width if i < num_segments - 1 else width
            
            segment = image[:, x1:x2].copy()
            segments.append(segment)
            bboxes.append((x1, 0, x2 - x1, height))
        
        logger.info(f"Segmentacja stała: {num_segments} segmentów po {segment_width}px")
        return segments, bboxes
    
    @staticmethod
    def visualize_segments(
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Narysuj prostokąty segmentów na obrazie.
        
        Args:
            image: Oryginalny obraz
            bboxes: Lista współrzędnych (x, y, w, h)
            color: Kolor linii (BGR)
            thickness: Grubość linii
        
        Returns:
            Obraz z narysowanymi segmentami
        """
        vis = image.copy()
        
        for i, (x, y, w, h) in enumerate(bboxes):
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
            # Dodaj numer segmentu
            cv2.putText(vis, str(i + 1), (x + 5, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis