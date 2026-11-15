#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eksporter anotacji do formatu CVAT XML.
"""

import xml.etree.ElementTree as ET
import os
import tempfile
import copy
from xml.dom import minidom
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from ..config import CONFIG, logger, CVAT_IMPORT_INFO
from ..data_models import Detection, ImageAnnotation
from ..plate_ground_truth import ensure_plate_detection_contract
from ..pose_geometry import GEOMETRY_XML_ATTR, geometry_xml_payload, geometry_matches_stored_polygon, attach_pose_geometry
from ..rectification.quad_geometry import analyze_quad


def write_cvat_tree_atomic(root: ET.Element, output_path: Path) -> None:
    """Serialize once and replace only after a complete write succeeds."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=output_path.parent,
                                         prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            ET.ElementTree(root).write(stream, encoding="utf-8", xml_declaration=True)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
            
            # Avoid reparsing the entire document into a second DOM just for
            # indentation; autosave and run completion both pass through here.
            write_cvat_tree_atomic(root, output_path)
            
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
        ensure_plate_detection_contract(det)
        poly = ET.SubElement(parent, "polygon")
        poly.set("label", "plate")
        manual_source = str(det.attributes.get("manual_source", "") or "").strip().lower()
        manually_edited = str(det.attributes.get("manually_edited", "") or "").strip().lower() == "true"
        poly.set("source", "manual" if manually_edited or manual_source else "auto")
        poly.set("occluded", "0")
        poly.set("z_order", "1")
        
        geometry = getattr(det, "geometry", None) or {}
        evidence_det = det
        preserved_xml = geometry.get("stored_polygon_xml")
        matches_stored = geometry_matches_stored_polygon(det)
        unchanged_legacy = (geometry.get("source") == "legacy_xml" and not geometry.get("manual_edit")
                            and matches_stored)
        if det.polygon:
            analysis = analyze_quad(det.polygon)
            if not analysis.is_geometrically_valid and not unchanged_legacy:
                raise ValueError(f"Nieprawidłowa geometria tablicy: {analysis.status}. Sprawdź narożniki.")
            if preserved_xml and not geometry.get("manual_edit") and matches_stored:
                points_str = preserved_xml
            else:
                def coordinate(value):
                    value = float(value)
                    return f"{value:.2f}" if value == round(value, 2) else repr(value)
                points_str = ";".join(f"{coordinate(p[0])},{coordinate(p[1])}" for p in det.polygon)
        else:
            # Fallback z bbox
            if geometry:
                raise ValueError("Brak polygonu roboczego mimo istniejących metadanych geometrii.")
            x1, y1, x2, y2 = det.bbox
            fallback = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            if not analyze_quad(fallback).is_geometrically_valid:
                raise ValueError("Nieprawidłowy bbox zastępczy tablicy.")
            if not geometry:
                evidence_det = copy.deepcopy(det)
                attach_pose_geometry(evidence_det, None)
            points_str = f"{x1:.2f},{y1:.2f};{x2:.2f},{y1:.2f};{x2:.2f},{y2:.2f};{x1:.2f},{y2:.2f}"
        
        poly.set("points", points_str)
        
        if include_confidence:
            attr = ET.SubElement(poly, "attribute")
            attr.set("name", "confidence")
            attr.text = f"{det.confidence:.3f}"
        self._add_detection_attributes(poly, det, include_confidence=include_confidence)
        payload = geometry_xml_payload(evidence_det)
        if payload:
            attr = ET.SubElement(poly, "attribute", name=GEOMETRY_XML_ATTR)
            attr.text = payload

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
            if name == GEOMETRY_XML_ATTR and getattr(det, "geometry", None):
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
