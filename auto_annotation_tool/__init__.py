#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-Annotation Tool dla CVAT v3.0
"""

from .config import CONFIG, logger
from .data_models import Detection, ImageAnnotation, AnnotationReport, AnnotationStatus

__version__ = CONFIG.VERSION
__all__ = [
    'CONFIG',
    'logger',
    'Detection',
    'ImageAnnotation',
    'AnnotationReport',
    'AnnotationStatus',
]