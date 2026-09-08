#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moduł rozpoznawania znaków na tablicach rejestracyjnych.
"""

from .plate_generator import PlateGenerator
from .char_detector import CharacterDetector, CharacterDetection, DetectionMethod
from .char_annotator import CharacterAnnotator, CharacterAnnotation  

__all__ = [
    'PlateGenerator',
    'CharacterDetector',
    'CharacterDetection',
    'DetectionMethod',
    'CharacterAnnotator',
    'CharacterAnnotation', 
]