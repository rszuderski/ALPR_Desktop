#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elastyczny podział datasetu.
"""

import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

from ..config import CONFIG, logger
from ..utils import safe_load_yaml


class DatasetSplitter:
    """
    Elastyczny podział datasetu YOLO.
    
    Obsługuje:
    - Podział train/val/test z dowolnymi proporcjami
    - Stratyfikację (równomierny rozkład)
    - Zachowanie istniejącego podziału
    - Scalanie wielu datasetów
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
    
    def split_dataset(self,
                      source_dir: Path,
                      output_dir: Path,
                      ratios: Dict[str, float],
                      progress_callback: Optional[Callable[[int, int, str], None]] = None
                      ) -> Tuple[bool, str, Dict]:
        """
        Dzieli dataset według podanych proporcji.
        
        Args:
            source_dir: Źródłowy dataset (z images/ i labels/)
            output_dir: Folder wyjściowy
            ratios: {"train": 0.7, "val": 0.2, "test": 0.1}
            progress_callback: Callback postępu
            
        Returns:
            (success, message, stats)
        """
        # Walidacja
        total_ratio = sum(ratios.values())
        if abs(total_ratio - 1.0) > 0.01:
            return False, f"Proporcje muszą sumować się do 1.0 (jest {total_ratio})", {}
        
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        
        # Zbierz wszystkie obrazy
        all_images = []
        
        for split in ["train", "val", "test", ""]:
            img_dir = source_dir / "images" / split if split else source_dir / "images"
            lbl_dir = source_dir / "labels" / split if split else source_dir / "labels"
            if split and not img_dir.exists():
                alt_img_dir = source_dir / split / "images"
                alt_lbl_dir = source_dir / split / "labels"
                if alt_img_dir.exists():
                    img_dir = alt_img_dir
                    lbl_dir = alt_lbl_dir
            if img_dir.exists():
                for img_path in img_dir.iterdir():
                    if img_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                        # Znajdź odpowiadającą etykietę
                        lbl_path = lbl_dir / (img_path.stem + ".txt")
                        
                        if lbl_path.exists():
                            all_images.append({
                                "image": img_path,
                                "label": lbl_path,
                                "name": img_path.name
                            })
        
        if not all_images:
            return False, "Nie znaleziono obrazów z etykietami", {}
        
        # Przetasuj
        random.seed(self.random_seed)
        random.shuffle(all_images)
        
        # Podziel
        n_total = len(all_images)
        splits_data = {}
        start_idx = 0
        
        for split_name, ratio in ratios.items():
            n_split = int(n_total * ratio)
            if split_name == list(ratios.keys())[-1]:
                # Ostatni split bierze resztę
                n_split = n_total - start_idx
            
            splits_data[split_name] = all_images[start_idx:start_idx + n_split]
            start_idx += n_split
        
        # Utwórz strukturę
        for split_name in ratios.keys():
            (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        
        # Kopiuj pliki
        stats = {split: 0 for split in ratios.keys()}
        stats["total"] = 0
        
        processed = 0
        for split_name, items in splits_data.items():
            for item in items:
                if progress_callback:
                    processed += 1
                    progress_callback(processed, n_total, item["name"])
                
                # Kopiuj obraz
                dst_img = output_dir / "images" / split_name / item["name"]
                shutil.copy2(item["image"], dst_img)
                
                # Kopiuj etykietę
                dst_lbl = output_dir / "labels" / split_name / item["label"].name
                shutil.copy2(item["label"], dst_lbl)
                
                stats[split_name] += 1
                stats["total"] += 1
        
        # Kopiuj/utwórz data.yaml
        self._create_data_yaml(source_dir, output_dir, list(ratios.keys()))
        
        return True, f"Podzielono {stats['total']} obrazów", stats
    
    def merge_datasets(self,
                       source_dirs: List[Path],
                       output_dir: Path,
                       progress_callback: Optional[Callable[[int, int, str], None]] = None
                       ) -> Tuple[bool, str, Dict]:
        """
        Scala wiele datasetów w jeden.
        
        Args:
            source_dirs: Lista folderów źródłowych
            output_dir: Folder wyjściowy
            progress_callback: Callback postępu
            
        Returns:
            (success, message, stats)
        """
        output_dir = Path(output_dir)
        stats = {"total": 0, "sources": len(source_dirs)}
        
        # Zbierz wszystkie pliki
        all_files = []
        
        for src_dir in source_dirs:
            src_dir = Path(src_dir)
            
            for split in ["train", "val", "test"]:
                img_dir = src_dir / "images" / split
                lbl_dir = src_dir / "labels" / split
                
                if not img_dir.exists():
                    continue
                
                for img_path in img_dir.iterdir():
                    if img_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                        lbl_path = lbl_dir / (img_path.stem + ".txt")
                        
                        if lbl_path.exists():
                            all_files.append({
                                "image": img_path,
                                "label": lbl_path,
                                "split": split,
                                "name": img_path.name
                            })
        
        if not all_files:
            return False, "Nie znaleziono plików", stats
        
        # Utwórz strukturę
        for split in ["train", "val", "test"]:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Kopiuj (z obsługą duplikatów)
        used_names = set()
        
        for i, item in enumerate(all_files):
            if progress_callback:
                progress_callback(i + 1, len(all_files), item["name"])
            
            name = item["name"]
            
            # Obsługa duplikatów
            if name in used_names:
                stem = Path(name).stem
                suffix = Path(name).suffix
                counter = 1
                while f"{stem}_{counter}{suffix}" in used_names:
                    counter += 1
                name = f"{stem}_{counter}{suffix}"
            
            used_names.add(name)
            
            # Kopiuj
            dst_img = output_dir / "images" / item["split"] / name
            dst_lbl = output_dir / "labels" / item["split"] / (Path(name).stem + ".txt")
            
            shutil.copy2(item["image"], dst_img)
            shutil.copy2(item["label"], dst_lbl)
            
            stats["total"] += 1
        
        # Utwórz data.yaml
        self._create_data_yaml(source_dirs[0], output_dir, ["train", "val", "test"])
        
        return True, f"Scalono {stats['total']} obrazów z {stats['sources']} źródeł", stats
    
 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elastyczny podział datasetu.
"""

import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

from ..config import CONFIG, logger
from ..utils import safe_load_yaml


class DatasetSplitter:
    """
    Elastyczny podział datasetu YOLO.
    
    Obsługuje:
    - Podział train/val/test z dowolnymi proporcjami
    - Stratyfikację (równomierny rozkład)
    - Zachowanie istniejącego podziału
    - Scalanie wielu datasetów
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
    
    def split_dataset(self,
                      source_dir: Path,
                      output_dir: Path,
                      ratios: Dict[str, float],
                      progress_callback: Optional[Callable[[int, int, str], None]] = None
                      ) -> Tuple[bool, str, Dict]:
        """
        Dzieli dataset według podanych proporcji.
        
        Args:
            source_dir: Źródłowy dataset (z images/ i labels/)
            output_dir: Folder wyjściowy
            ratios: {"train": 0.7, "val": 0.2, "test": 0.1}
            progress_callback: Callback postępu
            
        Returns:
            (success, message, stats)
        """
        # Walidacja
        total_ratio = sum(ratios.values())
        if abs(total_ratio - 1.0) > 0.01:
            return False, f"Proporcje muszą sumować się do 1.0 (jest {total_ratio})", {}
        
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        
        # Zbierz wszystkie obrazy
        all_images = []
        
        for split in ["train", "val", "test", ""]:
            img_dir = source_dir / "images" / split if split else source_dir / "images"
            lbl_dir = source_dir / "labels" / split if split else source_dir / "labels"
            if split and not img_dir.exists():
                alt_img_dir = source_dir / split / "images"
                alt_lbl_dir = source_dir / split / "labels"
                if alt_img_dir.exists():
                    img_dir = alt_img_dir
                    lbl_dir = alt_lbl_dir
            if img_dir.exists():
                for img_path in img_dir.iterdir():
                    if img_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                        # Znajdź odpowiadającą etykietę
                        lbl_path = lbl_dir / (img_path.stem + ".txt")
                        
                        if lbl_path.exists():
                            all_images.append({
                                "image": img_path,
                                "label": lbl_path,
                                "name": img_path.name
                            })
        
        if not all_images:
            return False, "Nie znaleziono obrazów z etykietami", {}
        
        # Przetasuj
        random.seed(self.random_seed)
        random.shuffle(all_images)
        
        # Podziel
        n_total = len(all_images)
        splits_data = {}
        start_idx = 0
        
        for split_name, ratio in ratios.items():
            n_split = int(n_total * ratio)
            if split_name == list(ratios.keys())[-1]:
                # Ostatni split bierze resztę
                n_split = n_total - start_idx
            
            splits_data[split_name] = all_images[start_idx:start_idx + n_split]
            start_idx += n_split
        
        # Utwórz strukturę
        for split_name in ratios.keys():
            (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        
        # Kopiuj pliki
        stats = {split: 0 for split in ratios.keys()}
        stats["total"] = 0
        
        processed = 0
        for split_name, items in splits_data.items():
            for item in items:
                if progress_callback:
                    processed += 1
                    progress_callback(processed, n_total, item["name"])
                
                # Kopiuj obraz
                dst_img = output_dir / "images" / split_name / item["name"]
                shutil.copy2(item["image"], dst_img)
                
                # Kopiuj etykietę
                dst_lbl = output_dir / "labels" / split_name / item["label"].name
                shutil.copy2(item["label"], dst_lbl)
                
                stats[split_name] += 1
                stats["total"] += 1
        
        # Kopiuj/utwórz data.yaml
        self._create_data_yaml(source_dir, output_dir, list(ratios.keys()))
        
        return True, f"Podzielono {stats['total']} obrazów", stats
    
    def merge_datasets(self,
                       source_dirs: List[Path],
                       output_dir: Path,
                       progress_callback: Optional[Callable[[int, int, str], None]] = None
                       ) -> Tuple[bool, str, Dict]:
        """
        Scala wiele datasetów w jeden.
        
        Args:
            source_dirs: Lista folderów źródłowych
            output_dir: Folder wyjściowy
            progress_callback: Callback postępu
            
        Returns:
            (success, message, stats)
        """
        output_dir = Path(output_dir)
        stats = {"total": 0, "sources": len(source_dirs)}
        
        # Zbierz wszystkie pliki
        all_files = []
        
        for src_dir in source_dirs:
            src_dir = Path(src_dir)
            
            for split in ["train", "val", "test"]:
                img_dir = src_dir / "images" / split
                lbl_dir = src_dir / "labels" / split
                
                if not img_dir.exists():
                    continue
                
                for img_path in img_dir.iterdir():
                    if img_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                        lbl_path = lbl_dir / (img_path.stem + ".txt")
                        
                        if lbl_path.exists():
                            all_files.append({
                                "image": img_path,
                                "label": lbl_path,
                                "split": split,
                                "name": img_path.name
                            })
        
        if not all_files:
            return False, "Nie znaleziono plików", stats
        
        # Utwórz strukturę
        for split in ["train", "val", "test"]:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Kopiuj (z obsługą duplikatów)
        used_names = set()
        
        for i, item in enumerate(all_files):
            if progress_callback:
                progress_callback(i + 1, len(all_files), item["name"])
            
            name = item["name"]
            
            # Obsługa duplikatów
            if name in used_names:
                stem = Path(name).stem
                suffix = Path(name).suffix
                counter = 1
                while f"{stem}_{counter}{suffix}" in used_names:
                    counter += 1
                name = f"{stem}_{counter}{suffix}"
            
            used_names.add(name)
            
            # Kopiuj
            dst_img = output_dir / "images" / item["split"] / name
            dst_lbl = output_dir / "labels" / item["split"] / (Path(name).stem + ".txt")
            
            shutil.copy2(item["image"], dst_img)
            shutil.copy2(item["label"], dst_lbl)
            
            stats["total"] += 1
        
        # Utwórz data.yaml
        self._create_data_yaml(source_dirs[0], output_dir, ["train", "val", "test"])
        
        return True, f"Scalono {stats['total']} obrazów z {stats['sources']} źródeł", stats
    
    def _create_data_yaml(self, source_dir: Path, output_dir: Path, splits: List[str]):
        """Tworzy/kopiuje data.yaml."""
        source_yaml = Path(source_dir) / "data.yaml"
        
        if source_yaml.exists():
            # Wczytaj i zmodyfikuj
            config = safe_load_yaml(source_yaml)
            
            content = f"""# YOLO Pose Dataset
path: {output_dir.absolute()}
train: images/train
val: images/val
"""
            if "test" in splits:
                content += "test: images/test\n"
            
            content += f"""
nc: {config.get('nc', 1)}
names:
"""
            names = config.get('names', {0: 'plate'})
            if isinstance(names, dict):
                for idx, name in names.items():
                    content += f"  {idx}: {name}\n"
            elif isinstance(names, list):
                for idx, name in enumerate(names):
                    content += f"  {idx}: {name}\n"
            
            if 'kpt_shape' in config:
                content += f"\nkpt_shape: {config['kpt_shape']}\n"
            
            if 'flip_idx' in config:
                content += f"flip_idx: {config['flip_idx']}\n"
        else:
            # Domyślny
            content = f"""# YOLO Pose Dataset
path: {output_dir.absolute()}
train: images/train
val: images/val

nc: 1
names:
  0: plate

kpt_shape: [4, 2]
flip_idx: [1, 0, 3, 2]
"""
        
        with open(output_dir / "data.yaml", 'w', encoding='utf-8') as f:
            f.write(content)
