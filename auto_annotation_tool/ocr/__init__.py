#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moduł OCR dla tablic rejestracyjnych.
"""

from .plate_ocr import PlateOCR
from .validators import LicensePlateValidator, LicensePlateFormat

__all__ = [
    'PlateOCR',
    'LicensePlateValidator',
    'LicensePlateFormat'
]