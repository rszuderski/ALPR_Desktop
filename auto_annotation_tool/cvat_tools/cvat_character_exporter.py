#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eksporter znaków do CVAT XML.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import json

from ..config import logger

class CVATCharacterExporter:
    """Eksportuje wycięte znaki z metadata.json do CVAT XML 1.1."""
    
    def __init__(self):
        self.task_name = "Character Annotation"
        self.labels = [
            {"name": "character", "type": "box", "color": "#00FF00"}
        ]
    
    def export(self, metadata_path: Path, output_xml_path: Path) -> bool:
        try:
            if not metadata_path.exists():
                logger.error(f"Brak pliku: {metadata_path}")
                return False
                
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            root = ET.Element("annotations")
            self._add_meta(root)
            
            for idx, (plate_id, data) in enumerate(metadata.items()):
                # Tworzymy tag <image> dla każdej wyciętej tablicy
                image_el = ET.SubElement(root, "image")
                image_el.set("id", str(idx))
                image_el.set("name", f"{plate_id}.jpg")
                
                # Odczytaj rzeczywiste wymiary wyciętej tablicy z obrazu.
                w, h = 256, 64 # Wartości domyślne w razie awarii
                
                # metadata_path to np. Workspace/3_cropped_characters/run_XXX/metadata.json
                # więc zdjęcia są w folderze obok:
                img_path = metadata_path.parent / "images" / f"{plate_id}.jpg"
                
                if img_path.exists():
                    try:
                        from PIL import Image
                        with Image.open(img_path) as img:
                            w, h = img.size
                    except Exception:
                        pass
                
                image_el.set("width", str(w))
                image_el.set("height", str(h))
                
                # Zapisujemy każdy znak jako <box>
                chars = data.get("characters", [])
                for char_data in chars:
                    char_text = str(char_data.get("character", "?"))
                    bbox = char_data.get("bbox", [0, 0, 0, 0])
                    conf = float(char_data.get("confidence", 0.0))
                    
                    box = ET.SubElement(image_el, "box")
                    box.set("label", "character")
                    box.set("source", "auto")
                    box.set("occluded", "0")
                    box.set("xtl", f"{bbox[0]:.2f}")
                    box.set("ytl", f"{bbox[1]:.2f}")
                    box.set("xbr", f"{bbox[2]:.2f}")
                    box.set("ybr", f"{bbox[3]:.2f}")
                    
                    # Tekst znaku zapisujemy w atrybucie
                    attr_txt = ET.SubElement(box, "attribute")
                    attr_txt.set("name", "text")
                    attr_txt.text = char_text
                    
                    # Pewność
                    attr_conf = ET.SubElement(box, "attribute")
                    attr_conf.set("name", "confidence")
                    attr_conf.text = f"{conf:.3f}"
            
            xml_str = ET.tostring(root, encoding='unicode')
            pretty_xml = self._prettify(xml_str)
            
            output_xml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_xml_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
                
            return True
            
        except Exception as e:
            logger.error(f"Błąd eksportu znaków do CVAT: {e}")
            return False

    def _add_meta(self, root: ET.Element):
        version = ET.SubElement(root, "version")
        version.text = "1.1"
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        ET.SubElement(task, "name").text = self.task_name
        
        labels_el = ET.SubElement(task, "labels")
        for label_def in self.labels:
            label_el = ET.SubElement(labels_el, "label")
            ET.SubElement(label_el, "name").text = label_def["name"]
            
            attrs = ET.SubElement(label_el, "attributes")
            
            # Deklaracja atrybutu 'text'
            attr_text = ET.SubElement(attrs, "attribute")
            ET.SubElement(attr_text, "name").text = "text"
            ET.SubElement(attr_text, "input_type").text = "text"
            ET.SubElement(attr_text, "default_value").text = "?"
            
            # Deklaracja atrybutu 'confidence'
            attr_conf = ET.SubElement(attrs, "attribute")
            ET.SubElement(attr_conf, "name").text = "confidence"
            ET.SubElement(attr_conf, "input_type").text = "number"
            ET.SubElement(attr_conf, "default_value").text = "1.0"
            
    def _prettify(self, xml_str: str) -> str:
        try:
            pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
            lines = [l for l in pretty.split('\n') if l.strip()]
            if lines[0].startswith('<?xml'): lines = lines[1:]
            return '<?xml version="1.0" encoding="utf-8"?>\n' + '\n'.join(lines)
        except Exception:
            return '<?xml version="1.0" encoding="utf-8"?>\n' + xml_str
