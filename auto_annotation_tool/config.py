#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Konfiguracja i stałe aplikacji.
"""

import logging
import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import FrozenSet
from pathlib import Path

# ============================================================================
# LOGOWANIE
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AutoAnnotationTool")

# ============================================================================
# SPRAWDZENIE DOSTĘPNOŚCI BIBLIOTEK
# ============================================================================

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    TK_AVAILABLE = True
except Exception as e:
    TK_AVAILABLE = False
    tk = None
    ttk = None
    logger.error("Tkinter niedostępny - GUI nie będzie działać")

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    logger.warning("PIL/Pillow niedostępny")

_TORCH_MODULE = None
_TORCH_IMPORT_ERROR: Exception | None = None
_YOLO_CLASS = None
_YOLO_IMPORT_ERROR: Exception | None = None
YOLO = None
torch = None


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


TORCH_AVAILABLE = _has_module("torch")
YOLO_AVAILABLE = bool(TORCH_AVAILABLE and _has_module("ultralytics"))
CUDA_AVAILABLE = False


def get_torch_module():
    """Lazy-load PyTorch dopiero wtedy, gdy faktycznie potrzebujemy GPU/modeli."""
    global _TORCH_MODULE, _TORCH_IMPORT_ERROR, torch
    if _TORCH_MODULE is not None:
        return _TORCH_MODULE
    if not TORCH_AVAILABLE:
        return None
    try:
        _TORCH_MODULE = importlib.import_module("torch")
        torch = _TORCH_MODULE
        return _TORCH_MODULE
    except Exception as exc:
        _TORCH_IMPORT_ERROR = exc
        logger.warning(f"PyTorch niedostępny albo nie może załadować bibliotek systemowych: {exc}")
        return None


def get_yolo_class():
    """Lazy-load Ultralytics YOLO bez obciążania startu GUI."""
    global _YOLO_CLASS, _YOLO_IMPORT_ERROR, YOLO
    if _YOLO_CLASS is not None:
        return _YOLO_CLASS
    if not YOLO_AVAILABLE:
        return None
    try:
        module = importlib.import_module("ultralytics")
        _YOLO_CLASS = getattr(module, "YOLO", None)
        YOLO = _YOLO_CLASS
        return _YOLO_CLASS
    except Exception as exc:
        _YOLO_IMPORT_ERROR = exc
        logger.warning(f"Ultralytics YOLO niedostępny albo nie może załadować bibliotek systemowych: {exc}")
        return None


def is_cuda_available() -> bool:
    torch_mod = get_torch_module()
    if torch_mod is None:
        return False
    try:
        return bool(torch_mod.cuda.is_available())
    except Exception:
        return False


def resolve_runtime_device(device: str | int | None = "auto") -> str | int:
    raw = str(device or "auto").strip().lower()
    if raw == "auto":
        return "cuda" if is_cuda_available() else "cpu"
    if raw.isdigit():
        return int(raw)
    return device or "cpu"
    logger.debug(f"Ultralytics/PyTorch pominięty podczas startu GUI: {e}")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None
    logger.warning("OpenCV niedostępny")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None
    logger.warning("PyYAML niedostępny")


# ============================================================================
# KONFIGURACJA GŁÓWNA I STRUKTURA KATALOGÓW
# ============================================================================

@dataclass
class Config:
    """Centralna konfiguracja aplikacji."""
    
    # Wersja
    VERSION: str = "4.5"
    APP_NAME: str = "ALPR Desktop"

    # Progi kampanii
    CAMPAIGN_MIN_CHAR_IMAGES: int = 10
    CAMPAIGN_MIN_PLATE_ANNOTATIONS: int = 10
    CAMPAIGN_MIN_CHAR_PLATES: int = 10

    # Progi jakości datasetów YOLO. Minimum kampanii powyżej jest techniczne;
    # poniższe wartości opisują sensowność materiału do realnego treningu.
    YOLO_POSE_AVERAGE_PLATES: int = 50
    YOLO_POSE_GOOD_PLATES: int = 200
    YOLO_POSE_VERY_GOOD_PLATES: int = 500
    YOLO_CHAR_AVERAGE_PLATES: int = 10
    YOLO_CHAR_AVERAGE_BOXES: int = 100
    YOLO_CHAR_GOOD_PLATES: int = 50
    YOLO_CHAR_GOOD_BOXES: int = 1000
    YOLO_CHAR_VERY_GOOD_PLATES: int = 150
    YOLO_CHAR_VERY_GOOD_BOXES: int = 5000
    
    # Rozszerzenia plików
    IMAGE_EXTENSIONS: FrozenSet[str] = field(
        default_factory=lambda: frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'})
    )
    
    # Kategorie
    VEHICLE_LABELS: FrozenSet[str] = field(
        default_factory=lambda: frozenset({
            'vehicle', 'car', 'truck', 'bus', 'motorcycle', 
            'motorbike', 'van', 'auto', 'samochod', 'pojazd'
        })
    )
    
    PLATE_LABELS: FrozenSet[str] = field(
        default_factory=lambda: frozenset({
            'plate', 'license_plate', 'numberplate', 'license plate',
            'tablica', 'rejestracja', 'lp', 'license-plate'
        })
    )
    
    # Parametry detekcji
    DEFAULT_CONFIDENCE: float = 0.25
    DEFAULT_IOU: float = 0.45
    DEFAULT_IMG_SIZE: int = 640
    
    # Próg dla sprawdzania czy tablica jest wewnątrz pojazdu
    PLATE_INSIDE_THRESHOLD: float = 0.95
    
    # ==========================================
    # LOGICZNA STRUKTURA KATALOGÓW (WORKSPACE)
    # ==========================================
    WORKSPACE_DIR: Path = Path("Workspace").resolve()
    
    DIR_1_RAW: Path         = WORKSPACE_DIR / "1_raw_images"
    DIR_2_AUTO_ANN: Path    = WORKSPACE_DIR / "2_auto_annotations"
    DIR_3_CHARS: Path       = WORKSPACE_DIR / "3_cropped_characters"
    DIR_4_DATASETS: Path    = WORKSPACE_DIR / "4_training_datasets"
    DIR_5_RUNS: Path        = WORKSPACE_DIR / "5_training_runs"
    DIR_6_MODELS: Path      = WORKSPACE_DIR / "6_models"
    DIR_9_PROJECTS: Path    = WORKSPACE_DIR / "9_projects"

    DIR_2_AUTO_ANN_PLATES: Path = DIR_2_AUTO_ANN / "plates"
    DIR_2_AUTO_ANN_CHARS: Path = DIR_2_AUTO_ANN / "chars"

    DIR_4_DATASETS_PLATES: Path = DIR_4_DATASETS / "plates"
    DIR_4_DATASETS_CHARS: Path = DIR_4_DATASETS / "chars"
    DIR_4_DATASETS_VEHICLES: Path = DIR_4_DATASETS / "vehicles"
    DIR_4_DATASETS_CHAR_CLASSIFICATION: Path = DIR_4_DATASETS / "4_char_classification"

    DIR_5_RUNS_PLATES: Path = DIR_5_RUNS / "plates"
    DIR_5_RUNS_CHARS: Path = DIR_5_RUNS / "chars"
    DIR_5_RUNS_VEHICLES: Path = DIR_5_RUNS / "vehicles"

    DIR_6_MODELS_BASE: Path          = DIR_6_MODELS / "base"
    DIR_6_MODELS_TRAINED: Path       = DIR_6_MODELS / "trained"
    DIR_6_MODELS_PLATES: Path        = DIR_6_MODELS_TRAINED / "plates_pose"
    DIR_6_MODELS_CHARS: Path         = DIR_6_MODELS_TRAINED / "characters_ocr"
    DIR_6_MODELS_BASE_POSE: Path     = DIR_6_MODELS_BASE / "pose"
    DIR_6_MODELS_BASE_DETECT: Path   = DIR_6_MODELS_BASE / "detect"
    DIR_6_MODELS_TRAINED_PLATES: Path = DIR_6_MODELS_TRAINED / "plates"
    DIR_6_MODELS_TRAINED_CHARS: Path = DIR_6_MODELS_TRAINED / "chars"
    DIR_6_MODELS_TRAINED_VEHICLES: Path = DIR_6_MODELS_TRAINED / "vehicles"
    DIR_6_MODELS_MOBILE_PACKAGES: Path = DIR_6_MODELS / "mobile_packages"
    
    DIR_7_RANKINGS: Path    = WORKSPACE_DIR / "7_rankings"
    DIR_7_RANKINGS_PLATES: Path = DIR_7_RANKINGS / "plates"
    DIR_7_RANKINGS_CHARS: Path = DIR_7_RANKINGS / "chars"
    DIR_7_RANKINGS_VEHICLES: Path = DIR_7_RANKINGS / "vehicles"
    DIR_7_RANKINGS_MOBILE_PACKAGES: Path = DIR_7_RANKINGS / "mobile_packages"

    DIR_8_PRESETS: Path = WORKSPACE_DIR / "8_presets"
    DIR_8_PRESETS_OCR: Path = DIR_8_PRESETS / "ocr"
    DIR_8_PRESETS_DETECTION_PIPELINE: Path = DIR_8_PRESETS / "detection_pipeline"
    DIR_8_PRESETS_AUGMENTATION: Path = DIR_8_PRESETS / "augmentation"
    DIR_8_PRESETS_AUGMENTATION_PLATES: Path = DIR_8_PRESETS_AUGMENTATION / "plate"
    DIR_8_PRESETS_AUGMENTATION_CHARS: Path = DIR_8_PRESETS_AUGMENTATION / "char"
    DIR_8_PRESETS_TRAINING: Path = DIR_8_PRESETS / "training"
    DIR_8_PRESETS_TRAINING_PLATES: Path = DIR_8_PRESETS_TRAINING / "plate"
    DIR_8_PRESETS_TRAINING_CHARS: Path = DIR_8_PRESETS_TRAINING / "char"
    DIR_8_PRESETS_RANKING_SCENARIOS: Path = DIR_8_PRESETS / "ranking_scenarios"
    DIR_8_LEGACY_OCR_PRESETS: Path = WORKSPACE_DIR / "8_ocr_presets"

    # Aliasy używane przez warstwę GUI.
    @property
    def DEFAULT_OUTPUT_DIR(self) -> str: return str(self.DIR_2_AUTO_ANN)
    
    @property
    def DEFAULT_MODELS_DIR(self) -> str: return str(self.DIR_6_MODELS)
    
    @property
    def DEFAULT_DATASETS_DIR(self) -> str: return str(self.DIR_4_DATASETS)
    
    @property
    def DEFAULT_TRAINING_DIR(self) -> str: return str(self.DIR_5_RUNS)
    
    @property
    def DEFAULT_RANKING_DIR(self) -> str: return str(self.DIR_7_RANKINGS)

    def normalize_task_target(self, target: str | None = None) -> str:
        raw = str(target or "").strip().lower()
        if raw in {"plate", "plates", "pose", "tablica", "tablice", "lp"}:
            return "plate"
        if raw in {"char", "chars", "character", "characters", "ocr", "znak", "znaki"}:
            return "char"
        if raw in {"vehicle", "vehicles", "detect", "pojazd", "pojazdy", "car", "cars"}:
            return "vehicle"
        return "char"

    def get_auto_annotations_dir(self, target: str | None = None) -> Path:
        normalized = self.normalize_task_target(target)
        if normalized == "char":
            return self.DIR_2_AUTO_ANN_CHARS
        return self.DIR_2_AUTO_ANN_PLATES

    def get_datasets_dir(self, target: str | None = None) -> Path:
        normalized = self.normalize_task_target(target)
        if normalized == "plate":
            return self.DIR_4_DATASETS_PLATES
        if normalized == "vehicle":
            return self.DIR_4_DATASETS_VEHICLES
        return self.DIR_4_DATASETS_CHARS

    def get_char_classification_datasets_dir(self) -> Path:
        return self.DIR_4_DATASETS_CHAR_CLASSIFICATION

    def get_training_runs_dir(self, target: str | None = None) -> Path:
        normalized = self.normalize_task_target(target)
        if normalized == "plate":
            return self.DIR_5_RUNS_PLATES
        if normalized == "vehicle":
            return self.DIR_5_RUNS_VEHICLES
        return self.DIR_5_RUNS_CHARS

    def get_ranking_dir(self, target: str | None = None) -> Path:
        normalized = self.normalize_task_target(target)
        if normalized == "plate":
            return self.DIR_7_RANKINGS_PLATES
        if normalized == "vehicle":
            return self.DIR_7_RANKINGS_VEHICLES
        return self.DIR_7_RANKINGS_CHARS

    def normalize_preset_module(self, module: str | None = None) -> str:
        raw = str(module or "").strip().lower()
        if raw in {"", "root", "preset", "presets"}:
            return ""
        if raw in {"ocr", "char_ocr", "ocr_presets"}:
            return "ocr"
        if raw in {"pipeline", "detection", "detection_pipeline", "detekcja"}:
            return "detection_pipeline"
        if raw in {"augmentation", "augmentacja", "aug"}:
            return "augmentation"
        if raw in {"training", "train", "trening"}:
            return "training"
        if raw in {"ranking", "rank", "ranking_scenarios"}:
            return "ranking_scenarios"

        safe_chars: list[str] = []
        for char in raw:
            if ("a" <= char <= "z") or ("0" <= char <= "9") or char in {"_", "-"}:
                safe_chars.append(char)
            else:
                safe_chars.append("_")
        return "".join(safe_chars).strip("_-") or "misc"

    def get_presets_dir(self, module: str | None = None, target: str | None = None) -> Path:
        normalized = self.normalize_preset_module(module)
        if not normalized:
            return self.DIR_8_PRESETS

        base = self.DIR_8_PRESETS / normalized
        if normalized in {"augmentation", "training"} and target:
            return base / self.normalize_task_target(target)
        return base

    def get_legacy_ocr_presets_dir(self) -> Path:
        return self.DIR_8_LEGACY_OCR_PRESETS

    def get_preset_search_dirs(self, module: str | None = None, target: str | None = None) -> list[Path]:
        normalized = self.normalize_preset_module(module)
        candidates = [self.get_presets_dir(normalized, target)]
        if normalized == "ocr":
            candidates.append(self.get_legacy_ocr_presets_dir())

        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def get_trained_models_dir(self, target: str | None = None) -> Path:
        normalized = self.normalize_task_target(target)
        if normalized == "plate":
            return self.DIR_6_MODELS_TRAINED_PLATES
        if normalized == "vehicle":
            return self.DIR_6_MODELS_TRAINED_VEHICLES
        return self.DIR_6_MODELS_TRAINED_CHARS

    def get_mobile_model_packages_dir(self, target: str | None = None) -> Path:
        normalized = self.normalize_task_target(target)
        return self.DIR_6_MODELS_MOBILE_PACKAGES / normalized

    def get_base_models_dir(self, target: str | None = None) -> Path:
        normalized = self.normalize_task_target(target)
        if normalized == "plate":
            return self.DIR_6_MODELS_BASE_POSE
        return self.DIR_6_MODELS_BASE_DETECT

    @staticmethod
    def _safe_count(value) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def describe_yolo_pose_dataset_quality(self, plate_count: int | None = None) -> dict:
        plates = self._safe_count(plate_count)
        average_min = self._safe_count(self.YOLO_POSE_AVERAGE_PLATES)
        good_min = self._safe_count(self.YOLO_POSE_GOOD_PLATES)
        very_good_min = self._safe_count(self.YOLO_POSE_VERY_GOOD_PLATES)

        if plates >= good_min:
            label = "DOBRY"
            tone = "success"
            next_label = "BARDZO DOBRY"
            missing_next = max(0, very_good_min - plates)
        elif plates >= average_min:
            label = "PRZECIĘTNY"
            tone = "warning"
            next_label = "DOBRY"
            missing_next = max(0, good_min - plates)
        else:
            label = "SŁABY"
            tone = "error"
            next_label = "PRZECIĘTNY"
            missing_next = max(0, average_min - plates)

        return {
            "label": label,
            "tone": tone,
            "score": plates,
            "metric": "tablice",
            "average_min": average_min,
            "good_min": good_min,
            "very_good_min": very_good_min,
            "next_label": next_label,
            "missing_next": missing_next,
            "range_text": (
                f"SŁABY < {average_min} tablic, "
                f"PRZECIĘTNY {average_min}-{max(average_min, good_min - 1)}, "
                f"DOBRY {good_min}+"
            ),
        }

    def describe_yolo_char_dataset_quality(
        self,
        *,
        perfect_plates: int | None = None,
        char_boxes: int | None = None,
    ) -> dict:
        plates = self._safe_count(perfect_plates)
        boxes = self._safe_count(char_boxes)
        average_plates = self._safe_count(self.YOLO_CHAR_AVERAGE_PLATES)
        average_boxes = self._safe_count(self.YOLO_CHAR_AVERAGE_BOXES)
        good_plates = self._safe_count(self.YOLO_CHAR_GOOD_PLATES)
        good_boxes = self._safe_count(self.YOLO_CHAR_GOOD_BOXES)
        very_good_plates = self._safe_count(self.YOLO_CHAR_VERY_GOOD_PLATES)
        very_good_boxes = self._safe_count(self.YOLO_CHAR_VERY_GOOD_BOXES)

        good_ready = plates >= good_plates and boxes >= good_boxes
        average_ready = plates >= average_plates and boxes >= average_boxes
        if good_ready:
            label = "DOBRY"
            tone = "success"
            next_label = "BARDZO DOBRY"
            missing_plates = max(0, very_good_plates - plates)
            missing_boxes = max(0, very_good_boxes - boxes)
        elif average_ready:
            label = "PRZECIĘTNY"
            tone = "warning"
            next_label = "DOBRY"
            missing_plates = max(0, good_plates - plates)
            missing_boxes = max(0, good_boxes - boxes)
        else:
            label = "SŁABY"
            tone = "error"
            next_label = "PRZECIĘTNY"
            missing_plates = max(0, average_plates - plates)
            missing_boxes = max(0, average_boxes - boxes)

        return {
            "label": label,
            "tone": tone,
            "perfect_plates": plates,
            "char_boxes": boxes,
            "average_plates": average_plates,
            "average_boxes": average_boxes,
            "good_plates": good_plates,
            "good_boxes": good_boxes,
            "very_good_plates": very_good_plates,
            "very_good_boxes": very_good_boxes,
            "next_label": next_label,
            "missing_next_plates": missing_plates,
            "missing_next_boxes": missing_boxes,
            "range_text": (
                f"SŁABY < {average_plates} tablic perfect lub < {average_boxes} znaków, "
                f"PRZECIĘTNY {average_plates}+ tablic / {average_boxes}+ znaków, "
                f"DOBRY {good_plates}+ tablic / {good_boxes}+ znaków"
            ),
        }

    def get_model_search_dirs(self, target: str | None = None) -> list[Path]:
        normalized = self.normalize_task_target(target)
        candidates: list[Path] = []

        if normalized == "plate":
            candidates.extend([
                self.DIR_6_MODELS_TRAINED_PLATES,
                self.DIR_6_MODELS_PLATES,
                self.DIR_6_MODELS / "pose",
                self.DIR_6_MODELS_BASE_POSE,
            ])
        elif normalized == "vehicle":
            candidates.extend([
                self.DIR_6_MODELS_TRAINED_VEHICLES,
                self.DIR_6_MODELS / "detect",
                self.DIR_6_MODELS_BASE_DETECT,
            ])
        else:
            candidates.extend([
                self.DIR_6_MODELS_TRAINED_CHARS,
                self.DIR_6_MODELS_CHARS,
                self.DIR_6_MODELS / "chars",
                self.DIR_6_MODELS_BASE_DETECT,
            ])

        candidates.append(self.DIR_6_MODELS)

        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def init_workspace(self):
        """Buduje tylko bazowy szkielet Workspace; głębsze gałęzie powstają leniwie przy użyciu."""
        directories = [
            self.DIR_1_RAW, 
            self.DIR_2_AUTO_ANN, 
            self.DIR_3_CHARS,
            self.DIR_4_DATASETS, 
            self.DIR_4_DATASETS_CHAR_CLASSIFICATION,
            self.DIR_5_RUNS, 
            self.DIR_6_MODELS, 
            self.DIR_6_MODELS_MOBILE_PACKAGES,
            self.DIR_7_RANKINGS, 
            self.DIR_7_RANKINGS_MOBILE_PACKAGES,
            self.DIR_8_PRESETS,
            self.DIR_8_PRESETS_OCR,
            self.DIR_8_PRESETS_DETECTION_PIPELINE,
            self.DIR_8_PRESETS_AUGMENTATION,
            self.DIR_8_PRESETS_AUGMENTATION_PLATES,
            self.DIR_8_PRESETS_AUGMENTATION_CHARS,
            self.DIR_8_PRESETS_TRAINING,
            self.DIR_8_PRESETS_TRAINING_PLATES,
            self.DIR_8_PRESETS_TRAINING_CHARS,
            self.DIR_8_PRESETS_RANKING_SCENARIOS,
            self.DIR_9_PROJECTS,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Generowanie pliku README z instrukcją dla użytkownika
        readme_path = self.WORKSPACE_DIR / "STRUKTURA_PROJEKTU.txt"
        if not readme_path.exists():
            readme_text = (
                "=== PRZEWODNIK PO PRZESTRZENI ROBOCZEJ (WORKSPACE) ===\n\n"
                "1_raw_images         : Wrzuć tutaj swoje surowe, nieopisane zdjęcia pojazdów.\n"
                "2_auto_annotations   : Wyniki autoanotacji, uporządkowane dalej na plates/ oraz chars/.\n"
                "3_cropped_characters : Tu lądują wycięte tablice i wyniki OCR z Zakładki nr 2.\n"
                "4_training_datasets  : Gotowe datasety YOLO, porządkowane na plates/, chars/ i vehicles/.\n"
                "  4_char_classification : Datasety OCR/klasyfikacji znaków (manifest.json), nie wejście Z4 YOLO.\n"
                "5_training_runs      : Logi i artefakty treningu, także rozdzielone na plates/, chars/ i vehicles/.\n"
                "6_models             : Modele bazowe w base/, wytrenowane w trained/, dodatkowo rozdzielone według toru.\n"
                "7_rankings           : Raporty z testów i walidacji, rozdzielone według typu modelu.\n"
                "8_presets            : Presety modułów: ocr/, detection_pipeline/, augmentation/, training/.\n"
            )
            readme_path.write_text(readme_text, encoding="utf-8")


# Singleton konfiguracji
CONFIG = Config()
CONFIG.init_workspace()


# ============================================================================
# DOSTĘPNE MODELE YOLO POSE
# ============================================================================

AVAILABLE_POSE_MODELS = {
    # YOLOv8
    "yolov8n-pose": {
        "name": "YOLOv8 Nano Pose",
        "file": "yolov8n-pose.pt",
        "params": "3.3M",
        "speed": "Najszybszy",
        "version": "v8",
        "description": "Najmniejszy model v8, idealny do testów"
    },
    "yolov8s-pose": {
        "name": "YOLOv8 Small Pose",
        "file": "yolov8s-pose.pt",
        "params": "11.6M",
        "speed": "Szybki",
        "version": "v8",
        "description": "Dobry balans szybkości i dokładności"
    },
    "yolov8m-pose": {
        "name": "YOLOv8 Medium Pose",
        "file": "yolov8m-pose.pt",
        "params": "26.4M",
        "speed": "Średni",
        "version": "v8",
        "description": "Większa dokładność"
    },
    "yolov8l-pose": {
        "name": "YOLOv8 Large Pose",
        "file": "yolov8l-pose.pt",
        "params": "44.4M",
        "speed": "Wolny",
        "version": "v8",
        "description": "Wysoka dokładność"
    },
    "yolov8x-pose": {
        "name": "YOLOv8 XLarge Pose",
        "file": "yolov8x-pose.pt",
        "params": "69.4M",
        "speed": "Najwolniejszy",
        "version": "v8",
        "description": "Najwyższa dokładność v8"
    },
    
    # YOLOv11
    "yolo11n-pose": {
        "name": "YOLO11 Nano Pose",
        "file": "yolo11n-pose.pt",
        "params": "2.9M",
        "speed": "Błyskawiczny",
        "version": "v11",
        "description": "Najnowszy lekki model"
    },
    "yolo11s-pose": {
        "name": "YOLO11 Small Pose",
        "file": "yolo11s-pose.pt",
        "params": "9.9M",
        "speed": "Bardzo szybki",
        "version": "v11",
        "description": "Rekomendowany dla tablic"
    },
    "yolo11m-pose": {
        "name": "YOLO11 Medium Pose",
        "file": "yolo11m-pose.pt",
        "params": "20.9M",
        "speed": "Szybki",
        "version": "v11",
        "description": "Wysoka dokładność"
    },
    "yolo11l-pose": {
        "name": "YOLO11 Large Pose",
        "file": "yolo11l-pose.pt",
        "params": "26.2M",
        "speed": "Średni",
        "version": "v11",
        "description": "Bardzo wysoka dokładność"
    },
    "yolo11x-pose": {
        "name": "YOLO11 XLarge Pose",
        "file": "yolo11x-pose.pt",
        "params": "58.8M",
        "speed": "Wolny",
        "version": "v11",
        "description": "Najwyższa dokładność v11"
    },
    
    # YOLOv26
    "yolo26n-pose": {
        "name": "YOLOv26 Nano Pose",
        "file": "yolo26n-pose.pt",
        "params": "3.0M",
        "speed": "Błyskawiczny",
        "version": "v26",
        "description": "Nano z v26 – ultra-lekki dla mobile/edge"
    },
    "yolo26s-pose": {
        "name": "YOLOv26 Small Pose",
        "file": "yolo26s-pose.pt",
        "params": "10.5M",
        "speed": "Bardzo szybki",
        "version": "v26",
        "description": "Small v26 – rekomendowany dla tablic"
    },
    "yolo26m-pose": {
        "name": "YOLOv26 Medium Pose",
        "file": "yolo26m-pose.pt",
        "params": "26.0M",
        "speed": "Szybki",
        "version": "v26",
        "description": "Medium v26 – idealny balans dla dokładności tablic"
    },
    "yolo26l-pose": {
        "name": "YOLOv26 Large Pose",
        "file": "yolo26l-pose.pt",
        "params": "45.0M",
        "speed": "Średni",
        "version": "v26",
        "description": "Large v26 – dla zaawansowanych zadań"
    },
    "yolo26x-pose": {
        "name": "YOLOv26 XLarge Pose",
        "file": "yolo26x-pose.pt",
        "params": "70.0M",
        "speed": "Wolny",
        "version": "v26",
        "description": "XLarge v26 – najwyższa precyzja na serwery"
    },
}

# ============================================================================
# DOSTĘPNE MODELE DO DETEKCJI POJAZDÓW (COCO)
# ============================================================================

AVAILABLE_DETECT_MODELS = {
    # YOLOv8
    "yolov8n": {
        "name": "YOLOv8 Nano (COCO)",
        "file": "yolov8n.pt",
        "params": "3.2M",
        "description": "Najszybszy, wykrywa pojazdy z COCO"
    },
    "yolov8s": {
        "name": "YOLOv8 Small (COCO)",
        "file": "yolov8s.pt",
        "params": "11.2M",
        "description": "Rekomendowany dla pojazdów"
    },
    "yolov8m": {
        "name": "YOLOv8 Medium (COCO)",
        "file": "yolov8m.pt",
        "params": "25.9M",
        "description": "Wyższa dokładność dla pojazdów"
    },
    "yolov8l": {
        "name": "YOLOv8 Large (COCO)",
        "file": "yolov8l.pt",
        "params": "44.0M",
        "description": "Wysoka dokładność, wolniejszy"
    },
    "yolov8x": {
        "name": "YOLOv8 XLarge (COCO)",
        "file": "yolov8x.pt",
        "params": "68.2M",
        "description": "Najwyższa dokładność v8 dla detekcji"
    },
    
    # YOLOv11
    "yolo11n": {
        "name": "YOLOv11 Nano (COCO)",
        "file": "yolo11n.pt",
        "params": "2.6M",
        "description": "Najnowszy nano, bardzo szybki"
    },
    "yolo11s": {
        "name": "YOLOv11 Small (COCO)",
        "file": "yolo11s.pt",
        "params": "9.4M",
        "description": "Najnowszy small, rekomendowany"
    },
    "yolo11m": {
        "name": "YOLOv11 Medium (COCO)",
        "file": "yolo11m.pt",
        "params": "20.1M",
        "description": "Najnowszy medium, wyższa dokładność"
    },
    "yolo11l": {
        "name": "YOLOv11 Large (COCO)",
        "file": "yolo11l.pt",
        "params": "25.3M",
        "description": "Najnowszy large, wysoka dokładność"
    },
    "yolo11x": {
        "name": "YOLOv11 XLarge (COCO)",
        "file": "yolo11x.pt",
        "params": "56.9M",
        "description": "Najwyższa dokładność v11"
    },

    # YOLOv26
    "yolo26n": {
        "name": "YOLOv26 Nano (COCO)",
        "file": "yolo26n.pt",
        "params": "2.8M",
        "description": "Nano v26, bardzo szybki"
    },
    "yolo26s": {
        "name": "YOLOv26 Small (COCO)",
        "file": "yolo26s.pt",
        "params": "9.8M",
        "description": "Small v26, rekomendowany dla pojazdow"
    },
    "yolo26m": {
        "name": "YOLOv26 Medium (COCO)",
        "file": "yolo26m.pt",
        "params": "21.5M",
        "description": "Medium v26, wyzsza dokladnosc dla pojazdow"
    },
    "yolo26l": {
        "name": "YOLOv26 Large (COCO)",
        "file": "yolo26l.pt",
        "params": "28.4M",
        "description": "Large v26, wysoka dokladnosc"
    },
    "yolo26x": {
        "name": "YOLOv26 XLarge (COCO)",
        "file": "yolo26x.pt",
        "params": "62.1M",
        "description": "XLarge v26, najwyzsza precyzja detect"
    },
}

# ============================================================================
# INFORMACJE O FORMACIE CVAT
# ============================================================================

CVAT_IMPORT_INFO = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                      INSTRUKCJA IMPORTU DO CVAT                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. UTWÓRZ ZADANIE W CVAT                                                   ║
║     • Projects → Create new project (opcjonalnie)                           ║
║     • Tasks → Create new task                                               ║
║     • Załaduj TE SAME obrazy co użyte w auto-anotacji                       ║
║                                                                              ║
║  2. ZDEFINIUJ ETYKIETY (Labels)                                             ║
║     Przed importem musisz utworzyć etykiety:                                ║
║                                                                              ║
║     ┌─────────────┬─────────────┬─────────────────────────────┐             ║
║     │ Nazwa       │ Typ         │ Opis                        │             ║
║     ├─────────────┼─────────────┼─────────────────────────────┤             ║
║     │ vehicle     │ Rectangle   │ Bounding box pojazdu        │             ║
║     │ plate       │ Polygon     │ 4 rogi tablicy              │             ║
║     └─────────────┴─────────────┴─────────────────────────────┘             ║
║                                                                              ║
║  3. IMPORTUJ ANOTACJE                                                       ║
║     • Otwórz zadanie                                                        ║
║     • Menu (3 kropki) → Import annotations                                  ║
║     • Format: "CVAT 1.1"                                                    ║
║     • Wybierz wygenerowany plik .xml                                        ║
║                                                                              ║
║  4. POPRAW ANOTACJE                                                         ║
║     • Sprawdź i popraw niedokładne anotacje                                 ║
║     • Dodaj pominięte tablice/pojazdy                                       ║
║     • Usuń fałszywe detekcje                                                ║
║                                                                              ║
║  5. EKSPORTUJ POPRAWIONE ANOTACJE                                           ║
║     • Menu → Export annotations                                             ║
║     • Format: "CVAT for images 1.1"                                         ║
║     • Użyj do porównania w zakładce "Ranking"                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# SESJA - Zapamiętywanie ostatnich ścieżek
# ============================================================================

try:
    from .session import SessionManager
    SESSION = SessionManager()
except ImportError as e:
    logger.warning(f"SessionManager niedostępny: {e}")
    SESSION = None
