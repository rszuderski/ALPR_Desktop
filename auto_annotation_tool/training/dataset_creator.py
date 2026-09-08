#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tworzenie datasetu YOLO Pose z eksportu CVAT.
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

from ..config import CONFIG, logger
from ..utils import get_image_files


@dataclass
class PlateAnnotation:
    """Anotacja tablicy."""
    image_name: str
    image_width: int
    image_height: int
    points: List[Tuple[float, float]]
    
    def to_yolo_pose(self) -> Optional[str]:
        """Konwertuje do formatu YOLO Pose."""
        if len(self.points) != 4:
            return None
        
        # Bbox z punktów
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Normalizacja
        x_center = ((x_min + x_max) / 2) / self.image_width
        y_center = ((y_min + y_max) / 2) / self.image_height
        width = (x_max - x_min) / self.image_width
        height = (y_max - y_min) / self.image_height
        
        # Keypoints (posortowane)
        sorted_points = self._sort_clockwise()
        
        keypoints = []
        for px, py in sorted_points:
            kp_x = px / self.image_width
            kp_y = py / self.image_height
            keypoints.extend([kp_x, kp_y])
        
        values = [0, x_center, y_center, width, height] + keypoints
        return " ".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in values])
    
    def _sort_clockwise(self) -> List[Tuple[float, float]]:
        """Sortuje punkty: TL, TR, BR, BL."""
        if len(self.points) != 4:
            return self.points
        
        cx = sum(p[0] for p in self.points) / 4
        cy = sum(p[1] for p in self.points) / 4
        
        top = sorted([p for p in self.points if p[1] < cy], key=lambda p: p[0])
        bottom = sorted([p for p in self.points if p[1] >= cy], key=lambda p: p[0], reverse=True)
        
        if len(top) != 2:
            sorted_by_y = sorted(self.points, key=lambda p: p[1])
            top = sorted(sorted_by_y[:2], key=lambda p: p[0])
            bottom = sorted(sorted_by_y[2:], key=lambda p: p[0], reverse=True)
        
        return [top[0], top[1], bottom[0], bottom[1]]


class DatasetCreator:
    """
    Tworzy dataset YOLO Pose z eksportu CVAT.
    
    Wymagany format CVAT:
    - Tablice jako <polygon> z dokładnie 4 punktami
    - Label: "plate", "license_plate" lub "numberplate"
    """
    
    REQUIRED_FORMAT = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    WYMAGANY FORMAT EKSPORTU Z CVAT                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. W CVAT: Menu → Export annotations → Format: "CVAT for images 1.1"       ║
║                                                                              ║
║  2. Struktura eksportu:                                                      ║
║     cvat_export/                                                             ║
║     ├── annotations.xml                                                      ║
║     └── images/                                                              ║
║         ├── img001.jpg                                                       ║
║         └── ...                                                              ║
║                                                                              ║
║  3. Format tablicy w XML:                                                    ║
║     <polygon label="plate" points="x1,y1;x2,y2;x3,y3;x4,y4"/>               ║
║                                                                              ║
║  UWAGI:                                                                      ║
║  • Polygon MUSI mieć dokładnie 4 punkty (4 rogi tablicy)                    ║
║  • Label: "plate", "license_plate" lub "numberplate"                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    def __init__(self):
        self.annotations: List[PlateAnnotation] = []
        self.xml_image_names: set[str] = set()
        self.annotated_image_names: set[str] = set()

    @staticmethod
    def _allocate_split_counts(n_total: int, split_ratios: Dict[str, float]) -> Dict[str, int]:
        ordered_splits = ["train", "val"]
        if "test" in split_ratios:
            ordered_splits.append("test")

        counts = {split: 0 for split in ordered_splits}
        required_splits = [split for split in ("train", "val") if split in counts]

        if n_total < len(required_splits):
            return {}

        weights = {
            split: max(0.0, float(split_ratios.get(split, 0.0) or 0.0))
            for split in ordered_splits
        }
        weight_sum = sum(weights.values())
        if weight_sum <= 0.0:
            weights["train"] = 1.0
            weight_sum = 1.0

        exact = {
            split: (weights[split] / weight_sum) * n_total
            for split in ordered_splits
        }
        fractional_parts = []
        allocated = 0
        for split in ordered_splits:
            extra = int(exact[split])
            counts[split] = extra
            allocated += extra
            fractional_parts.append((exact[split] - extra, split))

        leftover = n_total - allocated
        fractional_parts.sort(key=lambda item: (item[0], -ordered_splits.index(item[1])), reverse=True)
        for _fraction, split in fractional_parts:
            if leftover <= 0:
                break
            counts[split] += 1
            leftover -= 1

        # YOLO potrzebuje niepustych splitów train i val. Jeśli po czystym podziale
        # któryś z nich dostał 0, pożycz 1 obraz z największego splitu mającego zapas.
        for required_split in required_splits:
            if int(counts.get(required_split, 0) or 0) > 0:
                continue
            donor = None
            donor_count = 0
            for split in ordered_splits:
                split_count = int(counts.get(split, 0) or 0)
                if split == required_split or split_count <= 1:
                    continue
                if split_count > donor_count:
                    donor = split
                    donor_count = split_count
            if donor is None:
                return {}
            counts[donor] -= 1
            counts[required_split] += 1

        return counts
    
    @staticmethod
    def get_required_format() -> str:
        """Zwraca opis wymaganego formatu."""
        return DatasetCreator.REQUIRED_FORMAT
    
    def parse_cvat_xml(self, xml_path: Path, allowed_image_names: Optional[set[str]] = None) -> Tuple[bool, str, Dict]:
        """
        Parsuje plik CVAT XML.
        
        Returns:
            (success, message, stats)
        """
        stats = {
            "images": 0,
            "plates": 0,
            "skipped": 0,
            "skipped_unapproved": 0,
            "annotated_images": 0,
            "pending_xml_images": 0,
            "errors": []
        }
        
        self.annotations = []
        self.xml_image_names = set()
        self.annotated_image_names = set()
        
        if not xml_path.exists():
            return False, "Plik nie istnieje", stats

        allowed_lookup = None
        if allowed_image_names is not None:
            allowed_lookup = {
                str(name or "").strip().lower()
                for name in set(allowed_image_names or set())
                if str(name or "").strip()
            }
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for image in root.findall('.//image'):
                img_name = image.get('name', '')
                img_width = int(image.get('width', 0))
                img_height = int(image.get('height', 0))
                
                if not img_name or not img_width or not img_height:
                    continue

                if allowed_lookup is not None and img_name.strip().lower() not in allowed_lookup:
                    stats["skipped_unapproved"] += 1
                    continue
                
                stats["images"] += 1
                self.xml_image_names.add(img_name)
                
                for poly in image.findall('polygon'):
                    label = poly.get('label', '').lower()
                    
                    if label not in CONFIG.PLATE_LABELS:
                        continue
                    
                    points_str = poly.get('points', '')
                    if not points_str:
                        stats["skipped"] += 1
                        continue
                    
                    try:
                        points = []
                        for p in points_str.split(';'):
                            if ',' in p:
                                x, y = p.strip().split(',')
                                points.append((float(x), float(y)))
                        
                        if len(points) == 4:
                            self.annotations.append(PlateAnnotation(
                                image_name=img_name,
                                image_width=img_width,
                                image_height=img_height,
                                points=points
                            ))
                            self.annotated_image_names.add(img_name)
                            stats["plates"] += 1
                        else:
                            stats["skipped"] += 1
                            stats["errors"].append(f"{img_name}: polygon ma {len(points)} punktów (wymagane 4)")
                            
                    except Exception as e:
                        stats["skipped"] += 1
                        stats["errors"].append(f"{img_name}: {e}")
            
            if stats["plates"] == 0:
                return False, "Nie znaleziono tablic z 4 punktami", stats

            stats["annotated_images"] = len(self.annotated_image_names)
            stats["pending_xml_images"] = max(0, len(self.xml_image_names) - len(self.annotated_image_names))
            
            return True, f"Znaleziono {stats['plates']} tablic", stats
            
        except ET.ParseError as e:
            return False, f"Błąd parsowania XML: {e}", stats
    
    def create_dataset(self,
                       images_dir: Path,
                       output_dir: Path,
                       split_ratios: Dict[str, float] = None,
                       progress_callback: Optional[Callable[[int, int, str], None]] = None
                       ) -> Tuple[bool, str, Dict]:
        """
        Tworzy dataset YOLO Pose.
        
        Args:
            images_dir: Folder z obrazami
            output_dir: Folder wyjściowy
            split_ratios: {"train": 0.8, "val": 0.2} lub z "test"
            progress_callback: Callback postępu
            
        Returns:
            (success, message, stats)
        """
        if not self.annotations:
            return False, "Brak anotacji - najpierw sparsuj XML", {}
        
        split_ratios = split_ratios or {"train": 0.8, "val": 0.2}
        
        stats = {
            "total": 0,
            "train": 0,
            "val": 0,
            "test": 0,
            "skipped": 0
        }
        
        # Grupuj po obrazie
        images_data = {}
        for ann in self.annotations:
            if ann.image_name not in images_data:
                images_data[ann.image_name] = []
            images_data[ann.image_name].append(ann)
        
        # Przygotuj podział
        image_names = list(images_data.keys())
        
        import random
        random.seed(42)
        random.shuffle(image_names)
        
        n_total = len(image_names)
        split_counts = self._allocate_split_counts(n_total, split_ratios)
        if not split_counts:
            return False, "Do treningu YOLO Pose potrzebne sa co najmniej 2 oznaczone obrazy, aby wypelnic train i val.", stats

        splits = {}
        offset = 0
        for split_name in split_counts.keys():
            split_count = int(split_counts.get(split_name, 0) or 0)
            splits[split_name] = set(image_names[offset:offset + split_count])
            offset += split_count
        
        # Utwórz foldery
        output_dir = Path(output_dir)
        for split in splits.keys():
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Przetwarzaj
        for i, img_name in enumerate(image_names):
            if progress_callback:
                progress_callback(i + 1, n_total, img_name)
            
            # Znajdź split
            split = None
            for s, names in splits.items():
                if img_name in names:
                    split = s
                    break
            
            if not split:
                continue
            
            # Kopiuj obraz
            src_img = images_dir / img_name
            if not src_img.exists():
                stats["skipped"] += 1
                continue
            
            dst_img = output_dir / "images" / split / img_name
            shutil.copy2(src_img, dst_img)
            
            # Zapisz etykietę
            label_name = Path(img_name).stem + ".txt"
            label_path = output_dir / "labels" / split / label_name
            
            lines = []
            for ann in images_data[img_name]:
                line = ann.to_yolo_pose()
                if line:
                    lines.append(line)
            
            with open(label_path, 'w') as f:
                f.write("\n".join(lines))
            
            stats["total"] += 1
            stats[split] += 1
        
        # Utwórz data.yaml
        self._create_data_yaml(output_dir)
        
        return True, f"Utworzono dataset: {stats['total']} obrazów", stats
    
    def get_annotated_image_names(self) -> set[str]:
        return set(self.annotated_image_names)

    def get_pending_source_images(
        self,
        images_dir: Path,
        source_image_paths: Optional[List[Path]] = None,
    ) -> List[Path]:
        annotated_names = self.get_annotated_image_names()
        if source_image_paths is not None:
            result: List[Path] = []
            seen_names: set[str] = set()
            for image_path in list(source_image_paths or []):
                try:
                    candidate = Path(image_path)
                except Exception:
                    continue
                if not candidate.exists() or not candidate.is_file():
                    continue
                if candidate.suffix.lower() not in CONFIG.IMAGE_EXTENSIONS:
                    continue
                if candidate.name in annotated_names:
                    continue
                safe_name = candidate.name.lower()
                if safe_name in seen_names:
                    continue
                seen_names.add(safe_name)
                result.append(candidate)
            return result

        images_dir = Path(images_dir)
        return [
            image_path
            for image_path in get_image_files(images_dir)
            if image_path.name not in annotated_names
        ]

    @staticmethod
    def _load_stage_manifest(manifest_path: Path) -> Dict:
        if not manifest_path.exists():
            return {}
        try:
            data = json.loads(manifest_path.read_text(encoding='utf-8'))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _save_stage_manifest(manifest_path: Path, payload: Dict) -> None:
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    @staticmethod
    def _path_key(path_like) -> str:
        try:
            return str(Path(path_like).resolve())
        except Exception:
            return str(Path(path_like))

    @staticmethod
    def _paths_equivalent(left, right) -> bool:
        if left is None or right is None:
            return False
        try:
            return Path(left).resolve() == Path(right).resolve()
        except Exception:
            return str(Path(left)) == str(Path(right))

    @staticmethod
    def _find_stage_entry_key_by_name(entries: Dict, stage_name: str) -> Optional[str]:
        normalized = str(stage_name or "").strip()
        if not normalized:
            return None

        for entry_key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("stage_name") or "").strip() == normalized:
                return entry_key
        return None

    @staticmethod
    def _allocate_stage_name(stage_images_dir: Path, preferred_name: str, reserved_names: set[str]) -> str:
        preferred = str(preferred_name or "").strip() or "image.jpg"
        stem = Path(preferred).stem or "image"
        suffix = Path(preferred).suffix or ".jpg"
        candidate = preferred
        counter = 1
        while candidate in reserved_names or (stage_images_dir / candidate).exists():
            candidate = f"{stem}__{counter:03d}{suffix}"
            counter += 1
        return candidate

    def sync_pending_stage(
        self,
        images_dir: Path,
        stage_dir: Path,
        source_image_paths: Optional[List[Path]] = None,
    ) -> Tuple[bool, str, Dict]:
        images_dir = Path(images_dir)
        stage_dir = Path(stage_dir)
        stage_images_dir = stage_dir / "images"
        manifest_path = stage_dir / "stage_manifest.json"

        try:
            stage_images_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return False, f"Nie udało się przygotować stage: {e}", {}

        pending_paths = self.get_pending_source_images(images_dir, source_image_paths=source_image_paths)
        source_dir_resolved = self._path_key(images_dir)
        stage_images_dir_resolved = self._path_key(stage_images_dir)
        source_is_stage = source_dir_resolved == stage_images_dir_resolved

        manifest = self._load_stage_manifest(manifest_path)
        entries = manifest.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}

        pending_by_key = {
            self._path_key(path): path
            for path in pending_paths
        }
        pending_stage_names = {path.name for path in pending_paths}

        removed = 0
        added = 0
        updated = 0

        for source_key, entry in list(entries.items()):
            if not isinstance(entry, dict):
                entries.pop(source_key, None)
                continue

            stage_name = str(entry.get("stage_name") or "").strip()
            if not stage_name:
                entries.pop(source_key, None)
                continue

            target_path = stage_images_dir / stage_name
            entry_source_dir = str(entry.get("source_dir") or "").strip()

            if source_is_stage:
                if stage_name in pending_stage_names:
                    continue
                try:
                    if target_path.exists():
                        target_path.unlink()
                except Exception:
                    pass
                entries.pop(source_key, None)
                removed += 1
                continue

            if entry_source_dir != source_dir_resolved:
                continue
            if source_key in pending_by_key:
                continue

            try:
                if target_path.exists():
                    target_path.unlink()
            except Exception:
                pass
            entries.pop(source_key, None)
            removed += 1

        if source_is_stage:
            for stage_image_path in get_image_files(stage_images_dir):
                if stage_image_path.name in pending_stage_names:
                    continue
                try:
                    stage_image_path.unlink()
                    removed += 1
                except Exception:
                    pass

        reserved_names = {
            str(entry.get("stage_name") or "").strip()
            for entry in entries.values()
            if isinstance(entry, dict) and str(entry.get("stage_name") or "").strip()
        }

        for source_key, source_path in pending_by_key.items():
            source_in_stage = self._paths_equivalent(source_path.parent, stage_images_dir)
            entry_key = source_key
            entry = entries.get(entry_key) if isinstance(entries.get(entry_key), dict) else {}

            if source_in_stage and not entry:
                legacy_entry_key = self._find_stage_entry_key_by_name(entries, source_path.name)
                if legacy_entry_key and legacy_entry_key != source_key:
                    entry_key = legacy_entry_key
                    entry = entries.get(entry_key) if isinstance(entries.get(entry_key), dict) else {}

            previous_stage_name = str(entry.get("stage_name") or "").strip()
            previous_target_path = stage_images_dir / previous_stage_name if previous_stage_name else None

            if source_in_stage:
                stage_name = source_path.name
                if not entry:
                    added += 1
                else:
                    updated += 1

                if (
                    previous_target_path is not None
                    and previous_target_path.exists()
                    and not self._paths_equivalent(previous_target_path, source_path)
                ):
                    try:
                        previous_target_path.unlink()
                    except Exception:
                        pass
            else:
                stage_name = previous_stage_name
                target_path = stage_images_dir / stage_name if stage_name else None

                if not stage_name or target_path is None or not target_path.exists():
                    stage_name = self._allocate_stage_name(stage_images_dir, source_path.name, reserved_names)
                    target_path = stage_images_dir / stage_name
                    added += 1
                else:
                    updated += 1

                if not self._paths_equivalent(source_path, target_path):
                    try:
                        try:
                            os.link(source_path, target_path)
                        except Exception:
                            shutil.copy2(source_path, target_path)
                    except Exception as e:
                        return False, f"Nie udało się skopiować obrazu do stage: {e}", {}

            if entry_key != source_key:
                entries.pop(entry_key, None)

            reserved_names.add(stage_name)
            entries[source_key] = {
                "stage_name": stage_name,
                "image_name": source_path.name,
                "source_dir": stage_images_dir_resolved if source_in_stage else source_dir_resolved,
                "origin": "stage" if source_in_stage else "sync",
                "last_synced_at": datetime.now().isoformat(timespec="seconds"),
            }

        stage_images = get_image_files(stage_images_dir)
        payload = {
            "stage_version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "source_dir": source_dir_resolved,
            "annotated_images": len(self.get_annotated_image_names()),
            "pending_images": len(pending_paths),
            "stage_images_total": len(stage_images),
            "entries": entries,
        }

        try:
            self._save_stage_manifest(manifest_path, payload)
        except Exception as e:
            return False, f"Dataset gotowy, ale nie udało się zapisać manifestu stage: {e}", {}

        stats = {
            "stage_dir": str(stage_dir),
            "stage_images_dir": str(stage_images_dir),
            "pending_count": len(pending_paths),
            "stage_images_total": len(stage_images),
            "added": added,
            "updated": updated,
            "removed": removed,
            "pending_images": [path.name for path in pending_paths],
        }
        return True, f"Zsynchronizowano stage ręcznej anotacji: {len(pending_paths)} oczekujących zdjęć.", stats

    def add_images_to_stage(self, source_dir: Path, stage_dir: Path) -> Tuple[bool, str, Dict]:
        source_dir = Path(source_dir)
        stage_dir = Path(stage_dir)
        stage_images_dir = stage_dir / "images"
        manifest_path = stage_dir / "stage_manifest.json"

        if not source_dir.exists() or not source_dir.is_dir():
            return False, "Wybrany folder ze zdjęciami nie istnieje.", {}

        source_images = get_image_files(source_dir)
        if not source_images:
            return False, "W wybranym folderze nie ma obrazów do dodania.", {}

        try:
            stage_images_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return False, f"Nie udało się przygotować stage: {e}", {}

        manifest = self._load_stage_manifest(manifest_path)
        entries = manifest.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}

        stage_images_dir_resolved = self._path_key(stage_images_dir)
        source_dir_resolved = self._path_key(source_dir)
        reserved_names = {
            str(entry.get("stage_name") or "").strip()
            for entry in entries.values()
            if isinstance(entry, dict) and str(entry.get("stage_name") or "").strip()
        }
        existing_by_imported_from = {}
        for entry_key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            imported_from_path = str(entry.get("imported_from_path") or "").strip()
            if imported_from_path:
                existing_by_imported_from[imported_from_path] = entry_key

        added = 0
        updated = 0

        for source_path in source_images:
            import_key = self._path_key(source_path)
            source_in_stage = self._paths_equivalent(source_path.parent, stage_images_dir)

            if source_in_stage:
                stage_name = source_path.name
                target_path = source_path
                stage_key = self._path_key(target_path)
                entry_key = stage_key
                entry = entries.get(entry_key) if isinstance(entries.get(entry_key), dict) else {}
                if not entry:
                    added += 1
                else:
                    updated += 1
            else:
                entry_key = existing_by_imported_from.get(import_key)
                entry = entries.get(entry_key) if entry_key and isinstance(entries.get(entry_key), dict) else {}
                stage_name = str(entry.get("stage_name") or "").strip()
                target_path = stage_images_dir / stage_name if stage_name else None

                if not stage_name or target_path is None or not target_path.exists():
                    stage_name = self._allocate_stage_name(stage_images_dir, source_path.name, reserved_names)
                    target_path = stage_images_dir / stage_name
                    added += 1
                else:
                    updated += 1

                if not self._paths_equivalent(source_path, target_path):
                    try:
                        try:
                            os.link(source_path, target_path)
                        except Exception:
                            shutil.copy2(source_path, target_path)
                    except Exception as e:
                        return False, f"Nie udało się dodać obrazu do stage: {e}", {}

                stage_key = self._path_key(target_path)

            if entry_key and entry_key != stage_key:
                entries.pop(entry_key, None)

            reserved_names.add(stage_name)
            entries[stage_key] = {
                "stage_name": stage_name,
                "image_name": source_path.name,
                "source_dir": stage_images_dir_resolved,
                "origin": "manual",
                "imported_from_dir": source_dir_resolved if not source_in_stage else stage_images_dir_resolved,
                "imported_from_path": import_key,
                "last_synced_at": datetime.now().isoformat(timespec="seconds"),
            }
            existing_by_imported_from[import_key] = stage_key

        stage_images = get_image_files(stage_images_dir)
        payload = {
            "stage_version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "source_dir": stage_images_dir_resolved,
            "annotated_images": int(manifest.get("annotated_images", 0) or 0),
            "pending_images": len(stage_images),
            "stage_images_total": len(stage_images),
            "entries": entries,
        }

        try:
            self._save_stage_manifest(manifest_path, payload)
        except Exception as e:
            return False, f"Nie udało się zapisać manifestu stage: {e}", {}

        stats = {
            "stage_dir": str(stage_dir),
            "stage_images_dir": str(stage_images_dir),
            "added": added,
            "updated": updated,
            "stage_images_total": len(stage_images),
        }
        return True, f"Dodano obrazy do stage: {len(source_images)} plików.", stats

    def _create_data_yaml(self, output_dir: Path):
        """Tworzy przenośny plik data.yaml (bez ścieżek absolutnych)."""
        content = f"""# YOLO Pose Dataset - License Plates
# Wygenerowano przez {CONFIG.APP_NAME} v{CONFIG.VERSION}
# Brak zmiennej 'path' gwarantuje, że dataset jest w 100% przenośny!
# Ścieżki train/val są relatywne do lokalizacji tego pliku.

train: images/train
val: images/val

# Klasy
nc: 1
names:
  0: plate

# Keypoints: 4 rogi tablicy
kpt_shape: [4, 2]

# Flip indexes
flip_idx: [1, 0, 3, 2]
"""
        
        with open(output_dir / "data.yaml", 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Utworzono przenośny plik: {output_dir / 'data.yaml'}")
