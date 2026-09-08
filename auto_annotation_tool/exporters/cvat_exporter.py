#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eksporter anotacji do formatu CVAT XML.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from ..config import CONFIG, logger, CVAT_IMPORT_INFO
from ..data_models import Detection, ImageAnnotation


class CVATExporter:
    """
    Eksporter anotacji do formatu CVAT XML 1.1.
    
    Generuje:
    - <box label="vehicle"> dla pojazdów
    - <polygon label="plate" points="x1,y1;x2,y2;x3,y3;x4,y4"> dla tablic
    """
    
    def __init__(self, 
                 task_name: str = "Auto-annotation",
                 labels: Optional[List[Dict]] = None):
        """
        Args:
            task_name: Nazwa zadania w CVAT
            labels: Lista definicji etykiet
        """
        self.task_name = task_name
        self.labels = labels or [
            {"name": "vehicle", "type": "rectangle", "color": "#00FF00"},
            {"name": "plate", "type": "polygon", "color": "#FF0000"},
        ]
    
    @staticmethod
    def get_import_instructions() -> str:
        """Zwraca instrukcje importu do CVAT."""
        return CVAT_IMPORT_INFO
    
    def export(self, 
               annotations: List[ImageAnnotation],
               output_path: Path,
               include_confidence: bool = True,
               only_successful: bool = True) -> bool:
        """
        Eksportuje anotacje do pliku CVAT XML.
        
        Args:
            annotations: Lista anotacji
            output_path: Ścieżka wyjściowa
            include_confidence: Czy dodać atrybut confidence
            only_successful: Czy eksportować tylko udane anotacje
            
        Returns:
            True jeśli sukces
        """
        try:
            # Filtruj
            if only_successful:
                annotations = [a for a in annotations if a.is_successful]
            
            root = ET.Element("annotations")
            
            # Meta
            self._add_meta(root, include_confidence)
            
            # Obrazy
            for idx, ann in enumerate(annotations):
                self._add_image(root, idx, ann, include_confidence)
            
            # Formatuj i zapisz
            xml_str = ET.tostring(root, encoding='unicode')
            pretty_xml = self._prettify(xml_str)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
            
            # Statystyki
            total_vehicles = sum(ann.num_vehicles for ann in annotations)
            total_plates = sum(ann.num_plates for ann in annotations)
            
            logger.info(f"Zapisano CVAT XML: {output_path}")
            logger.info(f"  Obrazów: {len(annotations)}")
            logger.info(f"  Pojazdów: {total_vehicles}")
            logger.info(f"  Tablic: {total_plates}")
            
            return True
            
        except Exception as e:
            logger.error(f"Błąd eksportu: {e}")
            return False
    
    def _add_meta(self, root: ET.Element, include_confidence: bool):
        """Dodaje sekcję meta."""
        version = ET.SubElement(root, "version")
        version.text = "1.1"
        
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        
        ET.SubElement(task, "name").text = self.task_name
        ET.SubElement(task, "created").text = datetime.now().isoformat()
        ET.SubElement(task, "source").text = f"Auto-Annotation Tool v{CONFIG.VERSION}"
        
        # Labels
        labels_el = ET.SubElement(task, "labels")
        for label_def in self.labels:
            label_el = ET.SubElement(labels_el, "label")
            ET.SubElement(label_el, "name").text = label_def["name"]
            
            if "color" in label_def:
                ET.SubElement(label_el, "color").text = label_def["color"]
            
            if include_confidence:
                attrs = ET.SubElement(label_el, "attributes")
                attr = ET.SubElement(attrs, "attribute")
                ET.SubElement(attr, "name").text = "confidence"
                ET.SubElement(attr, "input_type").text = "number"
                ET.SubElement(attr, "default_value").text = "1.0"
    
    def _add_image(self, root: ET.Element, idx: int, ann: ImageAnnotation, 
                   include_confidence: bool):
        """Dodaje obraz z anotacjami."""
        image_el = ET.SubElement(root, "image")
        image_el.set("id", str(idx))
        image_el.set("name", ann.filename)
        image_el.set("width", str(ann.width))
        image_el.set("height", str(ann.height))
        
        for det in ann.detections:
            normalized_label = str(det.label or "").strip().lower()
            if normalized_label == "vehicle":
                self._add_box(image_el, det, include_confidence)
            elif normalized_label in CONFIG.PLATE_LABELS:
                self._add_polygon(image_el, det, include_confidence)
    
    def _add_box(self, parent: ET.Element, det: Detection, include_confidence: bool):
        """Dodaje <box>."""
        box = ET.SubElement(parent, "box")
        box.set("label", "vehicle")
        box.set("source", "auto")
        box.set("occluded", "0")
        box.set("z_order", "0")
        
        x1, y1, x2, y2 = det.bbox
        box.set("xtl", f"{x1:.2f}")
        box.set("ytl", f"{y1:.2f}")
        box.set("xbr", f"{x2:.2f}")
        box.set("ybr", f"{y2:.2f}")
        
        if include_confidence:
            attr = ET.SubElement(box, "attribute")
            attr.set("name", "confidence")
            attr.text = f"{det.confidence:.3f}"
        self._add_detection_attributes(box, det, include_confidence=include_confidence)
    
    def _add_polygon(self, parent: ET.Element, det: Detection, include_confidence: bool):
        """Dodaje <polygon>."""
        poly = ET.SubElement(parent, "polygon")
        poly.set("label", "plate")
        manual_source = str(det.attributes.get("manual_source", "") or "").strip().lower()
        manually_edited = str(det.attributes.get("manually_edited", "") or "").strip().lower() == "true"
        poly.set("source", "manual" if manually_edited or manual_source else "auto")
        poly.set("occluded", "0")
        poly.set("z_order", "1")
        
        if det.polygon and len(det.polygon) >= 4:
            points_str = ";".join([f"{p[0]:.2f},{p[1]:.2f}" for p in det.polygon[:4]])
        else:
            # Fallback z bbox
            x1, y1, x2, y2 = det.bbox
            points_str = f"{x1:.2f},{y1:.2f};{x2:.2f},{y1:.2f};{x2:.2f},{y2:.2f};{x1:.2f},{y2:.2f}"
        
        poly.set("points", points_str)
        
        if include_confidence:
            attr = ET.SubElement(poly, "attribute")
            attr.set("name", "confidence")
            attr.text = f"{det.confidence:.3f}"
        self._add_detection_attributes(poly, det, include_confidence=include_confidence)

    @staticmethod
    def _add_detection_attributes(parent: ET.Element, det: Detection, *, include_confidence: bool):
        """Zapisuje dodatkowe atrybuty detekcji do XML."""
        for attr_name, attr_value in dict(getattr(det, "attributes", {}) or {}).items():
            name = str(attr_name or "").strip()
            value = str(attr_value or "").strip()
            if not name or not value:
                continue
            if include_confidence and name == "confidence":
                continue
            attr = ET.SubElement(parent, "attribute")
            attr.set("name", name)
            attr.text = value
    
    def _prettify(self, xml_str: str) -> str:
        """Formatuje XML."""
        try:
            pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
            lines = pretty.split('\n')
            if lines[0].startswith('<?xml'):
                lines = lines[1:]
            lines = [l for l in lines if l.strip()]
            return '<?xml version="1.0" encoding="utf-8"?>\n' + '\n'.join(lines)
        except Exception:
            return '<?xml version="1.0" encoding="utf-8"?>\n' + xml_str
