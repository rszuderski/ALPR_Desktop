#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funkcje pomocnicze.
"""

import gc
import os
import time
from pathlib import Path
from threading import RLock
from typing import Dict, Any, Tuple, List

from .config import (
    CONFIG, logger, 
    YAML_AVAILABLE, yaml,
    PIL_AVAILABLE,
    CV2_AVAILABLE, cv2,
    is_cuda_available, get_torch_module
)

_IMAGE_FILES_CACHE_LOCK = RLock()
_IMAGE_FILES_CACHE: dict[str, dict[str, object]] = {}
_IMAGE_FILES_CACHE_TTL_SECONDS = 5.0


def cleanup_gpu_memory():
    """Zwalnia pamięć GPU."""
    gc.collect()
    torch = get_torch_module()
    if is_cuda_available() and torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
                torch.cuda.synchronize()
        except Exception as e:
            logger.debug(f"Błąd czyszczenia GPU: {e}")


def safe_load_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Bezpiecznie ładuje plik YAML."""
    result = {}
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Błąd odczytu pliku: {e}")
        return result
    
    if YAML_AVAILABLE and yaml is not None:
        try:
            result = yaml.safe_load(content) or {}
            return result
        except Exception as e:
            logger.warning(f"Błąd parsowania YAML: {e}")
    
    # Fallback parser
    for line in content.split('\n'):
        line = line.strip()
        
        if not line or line.startswith('#'):
            continue
        
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.split('#')[0].strip()
            value = value.strip('\'"')
            
            if value.lower() in ('true', 'yes'):
                result[key] = True
            elif value.lower() in ('false', 'no'):
                result[key] = False
            elif value.isdigit():
                result[key] = int(value)
            else:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    
    return result


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Pobiera wymiary obrazu (width, height)."""
    if PIL_AVAILABLE:
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            pass
    
    if CV2_AVAILABLE and cv2 is not None:
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
                return (w, h)
        except Exception:
            pass
    
    return (1920, 1080)


def count_images_in_directory(directory: Path) -> int:
    """Zlicza obrazy w katalogu."""
    if not directory.exists() or not directory.is_dir():
        return 0
    
    return sum(
        1 for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in CONFIG.IMAGE_EXTENSIONS
    )


def get_image_files(directory: Path) -> List[Path]:
    """Zwraca posortowaną listę plików obrazów."""
    if not directory.exists() or not directory.is_dir():
        return []

    try:
        cache_key = os.path.normcase(os.path.abspath(os.fsdecode(os.fspath(directory))))
    except Exception:
        cache_key = str(directory)
    try:
        stat = directory.stat()
        signature = (
            int(getattr(stat, "st_mtime_ns", 0) or 0),
            int(getattr(stat, "st_size", 0) or 0),
        )
    except Exception:
        signature = None
    now = time.monotonic()

    if cache_key and signature is not None:
        with _IMAGE_FILES_CACHE_LOCK:
            cached = _IMAGE_FILES_CACHE.get(cache_key)
            if (
                isinstance(cached, dict)
                and cached.get("signature") == signature
                and (now - float(cached.get("saved_at", 0.0) or 0.0)) <= _IMAGE_FILES_CACHE_TTL_SECONDS
            ):
                return [Path(item) for item in list(cached.get("paths") or [])]

    image_files = sorted([
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in CONFIG.IMAGE_EXTENSIONS
    ])

    if cache_key and signature is not None:
        with _IMAGE_FILES_CACHE_LOCK:
            if len(_IMAGE_FILES_CACHE) > 128:
                _IMAGE_FILES_CACHE.clear()
            _IMAGE_FILES_CACHE[cache_key] = {
                "signature": signature,
                "saved_at": now,
                "paths": [str(path) for path in image_files],
            }

    return image_files


def format_duration(seconds: float) -> str:
    """Formatuje czas trwania."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def calculate_iou(bbox1: Tuple[float, ...], bbox2: Tuple[float, ...]) -> float:
    """Oblicza IoU dwóch bounding boxów."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    inter_area = (x2 - x1) * (y2 - y1)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area
