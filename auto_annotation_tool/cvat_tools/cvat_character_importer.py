#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importer znaków z CVAT XML.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ..config import logger
from ..character_recognition import CharacterDetection


class CVATCharacterImporter:
    """
    Importuje poprawione anotacje znaków z CVAT.
    
    Czyta <polygon> lub <box> z atrybutami "text" i "confidence".
    """
    
    def __init__(self):
        self.annotations = {}  # image_name -> list of character detections
    
    def import_annotations(self, xml_path: Path) -> Tuple[bool, str, Dict]:
        """
        Importuje anotacje z CVAT XML.
        
        Args:
            xml_path: Ścieżka do CVAT XML
            
        Returns:
            (success, message, stats)
        """
        stats = {
            "images": 0,
            "characters": 0,
            "errors": []
        }
        
        self.annotations = {}
        
        if not xml_path.exists():
            return False, "Plik XML nie istnieje", stats
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for image in root.findall('.//image'):
                img_name = image.get('name', '')
                
                if not img_name:
                    continue
                
                stats["images"] += 1
                characters = []
                
                # Parsuj <polygon>
                for poly in image.findall('polygon'):
                    label = poly.get('label', '').lower()
                    
                    if label not in ['character', 'char']:
                        continue
                    
                    char_det = self._parse_polygon(poly)
                    if char_det:
                        characters.append(char_det)
                        stats["characters"] += 1
                
                # Parsuj <box>
                for box in image.findall('box'):
                    label = box.get('label', '').lower()
                    
                    if label not in ['char_bbox', 'character']:
                        continue
                    
                    char_det = self._parse_box(box)
                    if char_det:
                        characters.append(char_det)
                        stats["characters"] += 1
                
                if characters:
                    self.annotations[img_name] = characters
            
            msg = f"Importowano {stats['characters']} znaków z {stats['images']} tablic"
            return True, msg, stats
        
        except ET.ParseError as e:
            return False, f"Błąd parsowania XML: {e}", stats
        except Exception as e:
            return False, f"Błąd importu: {e}", stats
    
    def _parse_polygon(self, poly: ET.Element) -> Optional[CharacterDetection]:
        """Parsuje <polygon>."""
        try:
            points_str = poly.get('points', '')
            if not points_str:
                return None
            
            # Parsuj punkty
            points = []
            for p in points_str.split(';'):
                if ',' in p:
                    x, y = p.strip().split(',')
                    points.append((float(x), float(y)))
            
            if len(points) < 4:
                return None
            
            # Konwertuj na bbox
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            
            # Pobierz tekst z atrybutów
            text = "?"
            confidence = 1.0
            
            for attr in poly.findall('attribute'):
                name = attr.get('name', '')
                if name == 'text':
                    text = attr.text or "?"
                elif name == 'confidence':
                    try:
                        confidence = float(attr.text or "1.0")
                    except:
                        pass
            
            return CharacterDetection(
                character=text,
                bbox=bbox,
                confidence=confidence,
                method="cvat_import"
            )
        
        except Exception as e:
            logger.debug(f"Błąd parsowania polygonu: {e}")
            return None
    
    def _parse_box(self, box: ET.Element) -> Optional[CharacterDetection]:
        """Parsuje <box>."""
        try:
            x1 = float(box.get('xtl', 0))
            y1 = float(box.get('ytl', 0))
            x2 = float(box.get('xbr', 0))
            y2 = float(box.get('ybr', 0))
            
            bbox = (x1, y1, x2, y2)
            
            # Pobierz tekst
            text = "?"
            confidence = 1.0
            
            for attr in box.findall('attribute'):
                name = attr.get('name', '')
                if name == 'text':
                    text = attr.text or "?"
                elif name == 'confidence':
                    try:
                        confidence = float(attr.text or "1.0")
                    except:
                        pass
            
            return CharacterDetection(
                character=text,
                bbox=bbox,
                confidence=confidence,
                method="cvat_import"
            )
        
        except Exception as e:
            logger.debug(f"Błąd parsowania boxa: {e}")
            return None
    
    def get_characters_for_image(self, image_name: str) -> List[CharacterDetection]:
        """Zwraca znaki dla danego obrazu."""
        return self.annotations.get(image_name, [])
    
    def get_all_annotations(self) -> Dict[str, List[CharacterDetection]]:
        """Zwraca wszystkie importowane anotacje."""
        return self.annotations