#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Konwertery formatów danych.
Obsługuje konwersję CVAT XML → YOLO Pose format.
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .config import CONFIG, logger
from .rectification.quad_geometry import validated_pose_label_points, pose_label_identity_confirmation, format_pose_label


@dataclass
class PlateAnnotation:
    """Pojedyncza anotacja tablicy."""
    image_name: str
    image_width: int
    image_height: int
    points: List[Tuple[float, float]]  # 4 punkty (x, y) w pikselach
    attributes: Dict[str, str] = field(default_factory=dict)
    
    def to_yolo_pose(self) -> str:
        """
        Konwertuje do formatu YOLO Pose.
        
        Format: class_id x_center y_center width height kp1_x kp1_y kp2_x kp2_y kp3_x kp3_y kp4_x kp4_y
        Wszystkie wartości znormalizowane [0, 1]
        """
        sorted_points = self._sort_points_clockwise()
        
        return format_pose_label(sorted_points, (self.image_width, self.image_height))
    
    def _sort_points_clockwise(self) -> List[Tuple[float, float]]:
        """Zachowuje zatwierdzone indeksy; nie sortuje położenia na ekranie."""
        return validated_pose_label_points(self.points, (self.image_width, self.image_height),
                                           corner_identity_confirmed=pose_label_identity_confirmation(self.attributes))


class CVATToYOLOPoseConverter:
    """
    Konwerter CVAT XML → YOLO Pose format.
    
    WYMAGANY FORMAT WEJŚCIOWY (CVAT):
    ================================
    cvat_export/
    ├── annotations.xml        <- Eksport z CVAT (format "CVAT for images 1.1")
    └── images/                <- Folder z obrazami (opcjonalnie osobno)
        ├── img001.jpg
        ├── img002.jpg
        └── ...
    
    W pliku annotations.xml tablice muszą być jako <polygon> z 4 punktami:
    <polygon label="plate" points="x1,y1;x2,y2;x3,y3;x4,y4"/>
    
    FORMAT WYJŚCIOWY (YOLO Pose):
    =============================
    dataset_yolo_pose/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """
    
    REQUIRED_INPUT_FORMAT = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    WYMAGANY FORMAT DANYCH Z CVAT                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Eksport z CVAT:                                                            ║
║  ─────────────────                                                          ║
║  W CVAT wybierz: Menu → Export task → Format: "CVAT for images 1.1"         ║
║                                                                              ║
║  Otrzymasz plik ZIP zawierający:                                            ║
║  cvat_export/                                                               ║
║  ├── annotations.xml      ← Główny plik anotacji                           ║
║  └── images/              ← Folder z obrazami                              ║
║      ├── img001.jpg                                                         ║
║      ├── img002.jpg                                                         ║
║      └── ...                                                                ║
║                                                                              ║
║  Format anotacji tablicy:                                                   ║
║  ────────────────────────────                                               ║
║  Tablica MUSI być oznaczona jako POLYGON z dokładnie 4 punktami:           ║
║                                                                              ║
║  <polygon label="plate" points="x1,y1;x2,y2;x3,y3;x4,y4"/>                 ║
║                                                                              ║
║  Gdzie punkty to 4 ROGI tablicy w dowolnej kolejności                      ║
║  (program automatycznie posortuje: TL → TR → BR → BL)                      ║
║                                                                              ║
║  UWAGI:                                                                     ║
║  • Label MUSI być: "plate", "license_plate" lub "numberplate"              ║
║  • Polygon MUSI mieć dokładnie 4 punkty                                    ║
║  • Obrazy muszą być w formacie: JPG, PNG, BMP, WebP, TIFF                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    OUTPUT_FORMAT = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FORMAT WYJŚCIOWY (YOLO POSE)                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  dataset_yolo_pose/                                                         ║
║  ├── data.yaml                ← Konfiguracja datasetu                      ║
║  ├── images/                                                                ║
║  │   ├── train/               ← Obrazy treningowe                          ║
║  │   │   ├── img001.jpg                                                     ║
║  │   │   └── ...                                                            ║
║  │   └── val/                 ← Obrazy walidacyjne                         ║
║  │       ├── img050.jpg                                                     ║
║  │       └── ...                                                            ║
║  └── labels/                                                                ║
║      ├── train/               ← Etykiety treningowe                        ║
║      │   ├── img001.txt                                                     ║
║      │   └── ...                                                            ║
║      └── val/                 ← Etykiety walidacyjne                       ║
║          ├── img050.txt                                                     ║
║          └── ...                                                            ║
║                                                                              ║
║  Format etykiety (plik .txt):                                               ║
║  ─────────────────────────────────                                          ║
║  class x_center y_center width height kp1_x kp1_y kp2_x kp2_y ...          ║
║                                                                              ║
║  Gdzie:                                                                      ║
║  • class = 0 (plate)                                                        ║
║  • x_center, y_center, width, height = znormalizowane [0,1]                ║
║  • kp1..kp4 = 4 rogi tablicy (TL, TR, BR, BL), znormalizowane              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    def __init__(self, 
                 train_split: float = 0.8,
                 random_seed: int = 42):
        """
        Args:
            train_split: Proporcja danych treningowych (0.8 = 80% train, 20% val)
            random_seed: Ziarno dla powtarzalności podziału
        """
        self.train_split = train_split
        self.random_seed = random_seed
    
    def convert(self,
                cvat_xml_path: Path,
                images_dir: Path,
                output_dir: Path,
                progress_callback=None) -> Dict:
        """
        Konwertuje dane z CVAT do formatu YOLO Pose.
        
        Args:
            cvat_xml_path: Ścieżka do pliku annotations.xml
            images_dir: Ścieżka do folderu z obrazami
            output_dir: Ścieżka wyjściowa dla datasetu YOLO
            progress_callback: Funkcja (current, total, message)
            
        Returns:
            Dict ze statystykami konwersji
        """
        stats = {
            "total_images": 0,
            "total_plates": 0,
            "train_images": 0,
            "val_images": 0,
            "train_plates": 0,
            "val_plates": 0,
            "skipped_plates": 0,
            "errors": []
        }
        
        # Parsuj CVAT XML
        logger.info(f"Parsowanie CVAT XML: {cvat_xml_path}")
        annotations = self._parse_cvat_xml(cvat_xml_path, stats)
        if stats["errors"]:
            raise ValueError("Nieprawidłowe etykiety Pose: " + stats["errors"][0])
        
        if not annotations:
            stats["errors"].append("Nie znaleziono anotacji tablic")
            return stats
        prepared_labels = {id(ann): ann.to_yolo_pose() for ann in annotations}
        
        # Grupuj po obrazie
        images_annotations = {}
        for ann in annotations:
            if ann.image_name not in images_annotations:
                images_annotations[ann.image_name] = []
            images_annotations[ann.image_name].append(ann)
        
        stats["total_images"] = len(images_annotations)
        stats["total_plates"] = len(annotations)
        
        # Podziel na train/val
        image_names = list(images_annotations.keys())
        random.seed(self.random_seed)
        random.shuffle(image_names)
        
        split_idx = int(len(image_names) * self.train_split)
        train_images = set(image_names[:split_idx])
        val_images = set(image_names[split_idx:])
        
        # Utwórz strukturę folderów
        output_dir = Path(output_dir)
        for split in ["train", "val"]:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Przetwarzaj obrazy
        total = len(image_names)
        for i, img_name in enumerate(image_names):
            if progress_callback:
                progress_callback(i + 1, total, f"Konwertuję: {img_name}")
            
            split = "train" if img_name in train_images else "val"
            
            # Skopiuj obraz
            src_img = images_dir / img_name
            if not src_img.exists():
                stats["errors"].append(f"Brak obrazu: {img_name}")
                continue
            
            dst_img = output_dir / "images" / split / img_name
            shutil.copy2(src_img, dst_img)
            
            # Zapisz etykiety
            label_name = Path(img_name).stem + ".txt"
            label_path = output_dir / "labels" / split / label_name
            
            yolo_lines = []
            for ann in images_annotations[img_name]:
                yolo_line = prepared_labels[id(ann)]
                if yolo_line:
                    yolo_lines.append(yolo_line)
                    if split == "train":
                        stats["train_plates"] += 1
                    else:
                        stats["val_plates"] += 1
            
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_lines))
            
            if split == "train":
                stats["train_images"] += 1
            else:
                stats["val_images"] += 1
        
        # Utwórz data.yaml
        self._create_data_yaml(output_dir)
        
        logger.info(f"Konwersja zakończona: {stats}")
        return stats
    
    def _parse_cvat_xml(self, xml_path: Path, stats: Dict) -> List[PlateAnnotation]:
        """Parsuje plik CVAT XML."""
        annotations = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for image in root.findall('.//image'):
                img_name = image.get('name', '')
                img_width = int(image.get('width', 0))
                img_height = int(image.get('height', 0))
                
                if not img_name or not img_width or not img_height:
                    if any(poly.get('label', '').lower() in CONFIG.PLATE_LABELS for poly in image.findall('polygon')):
                        stats["errors"].append(f"{img_name}: nieprawidłowe wymiary obrazu dla Pose")
                    continue
                
                for poly in image.findall('polygon'):
                    label = poly.get('label', '').lower()
                    
                    if label not in CONFIG.PLATE_LABELS:
                        continue
                    
                    points_str = poly.get('points', '')
                    if not points_str:
                        stats["skipped_plates"] += 1
                        stats["errors"].append(f"{img_name}: KEYPOINTS_MISSING")
                        continue
                    
                    try:
                        points = []
                        for p in points_str.split(';'):
                            x, y = p.strip().split(',')
                            points.append((float(x), float(y)))
                        
                        if len(points) == 4:
                            ann = PlateAnnotation(
                                image_name=img_name,
                                image_width=img_width,
                                image_height=img_height,
                                points=points,
                                attributes={a.get('name', ''): a.text or '' for a in poly.findall('attribute')},
                            )
                            ann.to_yolo_pose()
                            annotations.append(ann)
                        else:
                            stats["skipped_plates"] += 1
                            stats["errors"].append(f"{img_name}: KEYPOINTS_MISSING ({len(points)})")
                            
                    except (ValueError, AttributeError) as exc:
                        stats["skipped_plates"] += 1
                        stats["errors"].append(f"{img_name}: {exc}")
                        
        except ET.ParseError as e:
            stats["errors"].append(f"Błąd parsowania XML: {e}")
        
        return annotations
    
    def _create_data_yaml(self, output_dir: Path):
        """Tworzy plik data.yaml dla YOLO Pose."""
        yaml_content = f"""# YOLO Pose Dataset - License Plates
# Wygenerowano przez Auto-Annotation Tool

path: {output_dir.absolute()}
train: images/train
val: images/val

# Klasy
nc: 1
names:
  0: plate

# Keypoints: 4 rogi tablicy (x, y dla każdego)
# Kolejność: Top-Left, Top-Right, Bottom-Right, Bottom-Left
kpt_shape: [4, 2]

# Flip indexes dla augmentacji (zamiana lewych z prawymi)
flip_idx: [1, 0, 3, 2]
"""
        
        yaml_path = output_dir / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        logger.info(f"Utworzono: {yaml_path}")


def get_format_requirements() -> str:
    """Zwraca opis wymaganego formatu danych."""
    return CVATToYOLOPoseConverter.REQUIRED_INPUT_FORMAT
