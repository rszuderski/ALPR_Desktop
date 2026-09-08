#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Porównywanie anotacji.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime

from ..config import CONFIG, logger
from ..utils import calculate_iou


@dataclass
class AnnotationDiff:
    """Różnica między anotacjami."""
    image_name: str
    
    auto_plates: int = 0
    corrected_plates: int = 0
    
    plates_unchanged: int = 0
    plates_minor_fix: int = 0
    plates_major_fix: int = 0
    plates_added: int = 0
    plates_removed: int = 0


class AnnotationComparator:
    """
    Porównuje anotacje automatyczne z poprawionymi.
    
    Metryki:
    - unchanged: IoU >= 0.95 (prawie identyczne)
    - minor_fix: IoU >= 0.8 (małe poprawki)
    - major_fix: IoU >= 0.5 (duże poprawki)
    - added: nowe w poprawionych
    - removed: usunięte z automatycznych
    """
    
    IOU_UNCHANGED = 0.95
    IOU_MINOR = 0.8
    IOU_MAJOR = 0.5
    
    def __init__(self):
        self.diffs: List[AnnotationDiff] = []
        self._parse_cache: Dict[tuple[str, int, int], Dict[str, List[Tuple]]] = {}
    
    def compare(self,
                auto_xml_path: Path,
                corrected_xml_path: Path) -> Dict:
        """
        Porównuje dwa pliki CVAT XML.
        
        Returns:
            Statystyki porównania
        """
        self.diffs = []
        
        stats = {
            "total_images": 0,
            "total_auto_plates": 0,
            "total_corrected_plates": 0,
            "plates_unchanged": 0,
            "plates_minor_fix": 0,
            "plates_major_fix": 0,
            "plates_added": 0,
            "plates_removed": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }
        
        # Parsuj
        auto_data = self._parse_cvat(auto_xml_path)
        corr_data = self._parse_cvat(corrected_xml_path)
        
        # Porównaj
        all_images = set(auto_data.keys()) | set(corr_data.keys())
        stats["total_images"] = len(all_images)
        
        for img_name in all_images:
            auto = auto_data.get(img_name, [])
            corr = corr_data.get(img_name, [])
            
            diff = self._compare_image(img_name, auto, corr)
            self.diffs.append(diff)
            
            stats["total_auto_plates"] += diff.auto_plates
            stats["total_corrected_plates"] += diff.corrected_plates
            stats["plates_unchanged"] += diff.plates_unchanged
            stats["plates_minor_fix"] += diff.plates_minor_fix
            stats["plates_major_fix"] += diff.plates_major_fix
            stats["plates_added"] += diff.plates_added
            stats["plates_removed"] += diff.plates_removed
        
        # Oblicz metryki
        total_good = stats["plates_unchanged"] + stats["plates_minor_fix"]
        total_detected = stats["total_auto_plates"]
        total_real = stats["total_corrected_plates"]
        
        if total_detected > 0:
            stats["precision"] = ((total_detected - stats["plates_removed"]) / total_detected) * 100
        
        if total_real > 0:
            stats["recall"] = ((total_real - stats["plates_added"]) / total_real) * 100
            stats["accuracy"] = (total_good / total_real) * 100
        
        logger.info(f"Porównano {len(all_images)} obrazów")
        
        return stats
    
    def _parse_cvat(self, xml_path: Path) -> Dict[str, List[Tuple]]:
        """Parsuje CVAT XML → {image: [bbox, ...]}"""
        try:
            stat = xml_path.stat()
            cache_key = (
                str(xml_path.resolve()),
                int(getattr(stat, "st_mtime_ns", 0)),
                int(getattr(stat, "st_size", 0)),
            )
        except Exception:
            cache_key = None

        if cache_key is not None and cache_key in self._parse_cache:
            return self._parse_cache[cache_key]

        data = {}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for image in root.findall('.//image'):
                img_name = image.get('name', '')
                data[img_name] = []
                
                for poly in image.findall('polygon'):
                    label = poly.get('label', '').lower()
                    
                    if label in CONFIG.PLATE_LABELS or label == "plate":
                        points_str = poly.get('points', '')
                        points = []
                        
                        for p in points_str.split(';'):
                            if ',' in p:
                                x, y = p.strip().split(',')
                                points.append((float(x), float(y)))
                        
                        if len(points) >= 4:
                            xs = [p[0] for p in points]
                            ys = [p[1] for p in points]
                            bbox = (min(xs), min(ys), max(xs), max(ys))
                            data[img_name].append(bbox)
                            
        except Exception as e:
            logger.error(f"Błąd parsowania {xml_path}: {e}")
        
        if cache_key is not None:
            self._parse_cache[cache_key] = data

        return data
    
    def _compare_image(self, img_name: str, auto: List, corr: List) -> AnnotationDiff:
        """Porównuje anotacje dla obrazu."""
        diff = AnnotationDiff(image_name=img_name)
        diff.auto_plates = len(auto)
        diff.corrected_plates = len(corr)
        
        matched_auto = set()
        matched_corr = set()
        
        for i, auto_bbox in enumerate(auto):
            best_j = None
            best_iou = 0
            
            for j, corr_bbox in enumerate(corr):
                if j in matched_corr:
                    continue
                
                iou = calculate_iou(auto_bbox, corr_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            if best_j is not None and best_iou >= self.IOU_MAJOR:
                matched_auto.add(i)
                matched_corr.add(best_j)
                
                if best_iou >= self.IOU_UNCHANGED:
                    diff.plates_unchanged += 1
                elif best_iou >= self.IOU_MINOR:
                    diff.plates_minor_fix += 1
                else:
                    diff.plates_major_fix += 1
        
        diff.plates_removed = len(auto) - len(matched_auto)
        diff.plates_added = len(corr) - len(matched_corr)
        
        return diff
