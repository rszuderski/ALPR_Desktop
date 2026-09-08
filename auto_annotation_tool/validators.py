#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funkcje walidacji plików i datasetów.
"""

import datetime
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Set, Tuple

from .config import CONFIG, logger, YOLO_AVAILABLE, get_yolo_class
from .utils import safe_load_yaml, cleanup_gpu_memory


_MODEL_VALIDATION_CACHE: Dict[Tuple[str, int, int], Tuple[bool, str, Dict]] = {}
_MODEL_METADATA_SCHEMA = "auto_annotation_tool.model_metadata.v1"
_EXPORTED_MODEL_METADATA_SCHEMA = "auto_annotation_tool.exported_model.v1"


def _empty_model_info() -> Dict:
    return {
        "type": "unknown",
        "task": "unknown",
        "classes": [],
        "num_classes": 0,
        "keypoints": False,
        "kpt_shape": None,
        "file_size_mb": 0,
        "file_name": "",
        "yolo_family": "",
        "yolo_version": "",
        "yolo_size": "",
        "yolo_variant": "",
        "model_scale": "",
        "yaml_file": "",
        "architecture_label": "",
        "source_model": "",
        "source_model_name": "",
        "source_architecture_label": "",
        "ultralytics_version": "",
        "parameter_count": 0,
        "parameters_millions": 0.0,
    }


def _clone_model_info(info: Dict) -> Dict:
    cloned = dict(info or {})
    cloned["classes"] = list(cloned.get("classes") or [])
    kpt_shape = cloned.get("kpt_shape")
    if isinstance(kpt_shape, list):
        cloned["kpt_shape"] = list(kpt_shape)
    return cloned


def _build_model_cache_key(model_path: Path) -> Tuple[str, int, int] | None:
    try:
        resolved = str(model_path.resolve())
    except Exception:
        resolved = str(model_path)

    try:
        stat = model_path.stat()
    except Exception:
        return None

    return (resolved, int(getattr(stat, "st_mtime_ns", 0) or 0), int(getattr(stat, "st_size", 0) or 0))


def _json_safe_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _model_file_fingerprint(model_path: Path) -> Dict:
    fingerprint: Dict = {
        "file_name": str(model_path.name or "").strip(),
    }
    try:
        fingerprint["path"] = str(model_path.resolve())
    except Exception:
        fingerprint["path"] = str(model_path)
    try:
        stat = model_path.stat()
        fingerprint["size"] = int(getattr(stat, "st_size", 0) or 0)
        fingerprint["mtime_ns"] = int(getattr(stat, "st_mtime_ns", 0) or 0)
    except Exception:
        pass
    return fingerprint


def _model_metadata_sidecar_candidates(model_path: Path) -> list[Path]:
    candidates: list[Path] = []

    def add(candidate: Path | None) -> None:
        if candidate is None:
            return
        try:
            normalized = candidate.resolve()
        except Exception:
            normalized = candidate
        if all(str(existing) != str(normalized) for existing in candidates):
            candidates.append(normalized)

    suffix = str(model_path.suffix or "").strip()
    if suffix:
        add(model_path.with_suffix(f"{suffix}.metadata.json"))
    add(model_path.with_suffix(".metadata.json"))
    add(model_path.with_suffix(".json"))
    add(model_path.with_name(f"{model_path.stem}_metadata.json"))
    add(model_path.with_name("model_metadata.json"))
    add(model_path.with_name("metadata.json"))
    return candidates


def _metadata_payload_matches_model(model_path: Path, payload: Dict) -> bool:
    if not isinstance(payload, dict):
        return False

    model_payload = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    if not isinstance(model_payload, dict):
        model_payload = {}

    expected_name = str(model_path.name or "").strip()
    candidate_names = set()
    for key in ("file_name", "name"):
        value = str(model_payload.get(key) or "").strip()
        if value:
            candidate_names.add(value)
    path_value = str(model_payload.get("path") or "").strip()
    if path_value:
        try:
            candidate_names.add(Path(path_value).name)
        except Exception:
            pass

    if candidate_names and expected_name not in candidate_names:
        return False

    fingerprint = model_payload.get("fingerprint")
    if not isinstance(fingerprint, dict):
        fingerprint = payload.get("fingerprint") if isinstance(payload.get("fingerprint"), dict) else {}
    if not isinstance(fingerprint, dict):
        fingerprint = {}

    if not candidate_names and not fingerprint:
        return False

    try:
        stat = model_path.stat()
    except Exception:
        stat = None

    if stat is not None:
        for size_key in ("size", "file_size", "st_size"):
            if size_key in fingerprint:
                try:
                    if int(fingerprint.get(size_key) or 0) != int(getattr(stat, "st_size", 0) or 0):
                        return False
                except Exception:
                    pass
                break
        for mtime_key in ("mtime_ns", "st_mtime_ns"):
            if mtime_key in fingerprint:
                try:
                    if int(fingerprint.get(mtime_key) or 0) != int(getattr(stat, "st_mtime_ns", 0) or 0):
                        return False
                except Exception:
                    pass
                break

    return True


def _normalize_model_metadata_payload(model_path: Path, metadata_path: Path, payload: Dict):
    if not _metadata_payload_matches_model(model_path, payload):
        return None

    schema = str(payload.get("schema") or "").strip()
    model_payload = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    if not isinstance(model_payload, dict):
        model_payload = {}

    raw_info = {}
    validation_ok = True
    validation_message = "Model metadata sidecar OK"

    if schema == _EXPORTED_MODEL_METADATA_SCHEMA:
        raw_info = model_payload.get("info") if isinstance(model_payload.get("info"), dict) else {}
        validation_ok = bool(model_payload.get("validation_ok", True))
        validation_message = str(model_payload.get("validation_message") or validation_message)
    elif schema == _MODEL_METADATA_SCHEMA:
        raw_info = model_payload.get("info") if isinstance(model_payload.get("info"), dict) else payload.get("info")
        if not isinstance(raw_info, dict):
            raw_info = {}
        validation_ok = bool(model_payload.get("validation_ok", payload.get("validation_ok", True)))
        validation_message = str(model_payload.get("validation_message") or payload.get("validation_message") or validation_message)
    elif isinstance(model_payload.get("info"), dict):
        raw_info = model_payload.get("info") or {}
        validation_ok = bool(model_payload.get("validation_ok", True))
        validation_message = str(model_payload.get("validation_message") or validation_message)
    elif isinstance(payload.get("info"), dict):
        raw_info = payload.get("info") or {}
        validation_ok = bool(payload.get("validation_ok", True))
        validation_message = str(payload.get("validation_message") or validation_message)
    else:
        return None

    info = _empty_model_info()
    info["file_name"] = str(model_path.name or "").strip()
    info.update(_infer_yolo_identity_from_text(model_path.name))
    try:
        info["file_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 2)
    except Exception:
        pass
    info.update({str(key): value for key, value in dict(raw_info or {}).items()})
    info["file_name"] = str(model_path.name or "").strip()
    info["metadata_schema"] = schema or "legacy"
    info["metadata_json"] = str(metadata_path)
    info["metadata_source"] = "sidecar"
    return bool(validation_ok), validation_message, _clone_model_info(info)


def read_model_metadata_sidecar(model_path: Path):
    """Return lightweight model metadata saved next to a .pt file, if present."""
    safe_path = Path(model_path)
    for metadata_path in _model_metadata_sidecar_candidates(safe_path):
        if not metadata_path.exists():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except Exception as e:
            logger.debug(f"Could not read model metadata sidecar {metadata_path}: {e}")
            continue
        if not isinstance(payload, dict):
            continue
        normalized = _normalize_model_metadata_payload(safe_path, metadata_path, payload)
        if normalized is not None:
            return normalized
    return None


def write_model_metadata_sidecar(
    model_path: Path,
    info: Dict,
    *,
    validation_ok: bool = True,
    validation_message: str = "",
    extra: Dict | None = None,
    sidecar_path: Path | None = None,
) -> Path | None:
    """Write a small JSON sidecar so UI code can avoid loading heavy .pt files."""
    safe_path = Path(model_path)
    target_path = Path(sidecar_path) if sidecar_path is not None else safe_path.with_suffix(f"{safe_path.suffix}.metadata.json")
    payload = {
        "schema": _MODEL_METADATA_SCHEMA,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": {
            "file_name": str(safe_path.name or "").strip(),
            "path": str(safe_path),
            "fingerprint": _model_file_fingerprint(safe_path),
            "validation_ok": bool(validation_ok),
            "validation_message": str(validation_message or "").strip(),
            "info": _json_safe_value(_clone_model_info(info or {})),
        },
    }
    if isinstance(extra, dict) and extra:
        payload["extra"] = _json_safe_value(extra)
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return target_path
    except Exception as e:
        logger.debug(f"Could not write model metadata sidecar {target_path}: {e}")
        return None


def _infer_yolo_identity_from_text(raw_text: str | Path | None, *, task_hint: str = "") -> Dict:
    text = str(raw_text or "").strip()
    if not text:
        return {}

    match = re.search(r"yolo(?:v)?(8|11|26)([nsmlx])(?:[-_ ]?(pose))?", text.lower())
    if not match:
        return {}

    version = str(match.group(1) or "").strip()
    size = str(match.group(2) or "").strip().lower()
    pose_token = bool(match.group(3))
    normalized_task = str(task_hint or "").strip().lower()
    task_label = ""
    if pose_token or normalized_task == "pose":
        task_label = "Pose"
    elif normalized_task:
        task_label = "Detect"

    family = f"YOLOv{version}" if version == "8" else f"YOLO{version}"
    variant = f"{family}{size}"
    architecture_label = f"{variant} {task_label}".strip()
    return {
        "yolo_family": family,
        "yolo_version": version,
        "yolo_size": size,
        "yolo_variant": variant,
        "architecture_label": architecture_label,
    }


def format_yolo_model_identity(info: Dict | None, *, include_ultralytics_version: bool = False) -> str:
    if not isinstance(info, dict):
        return ""

    architecture_label = str(info.get("architecture_label") or "").strip()
    if not architecture_label:
        task = str(info.get("task") or info.get("type") or "").strip().lower()
        if task == "pose":
            architecture_label = "YOLO Pose"
        elif task:
            architecture_label = "YOLO Detect"

    parts = [architecture_label] if architecture_label else []
    if include_ultralytics_version:
        runtime_version = str(info.get("ultralytics_version") or "").strip()
        if runtime_version:
            parts.append(f"Ultralytics {runtime_version}")
    return " | ".join(parts)


def _resolve_nested_source_model_identity(
    source_model_value: str | Path | None,
    *,
    task_hint: str = "",
    visited: set[str] | None = None,
) -> Dict:
    text = str(source_model_value or "").strip()
    if not text:
        return {}

    direct_identity = _infer_yolo_identity_from_text(text, task_hint=task_hint)
    if direct_identity:
        return direct_identity

    try:
        source_path = Path(text)
    except Exception:
        return {}

    if source_path.suffix.lower() != ".pt" or not source_path.exists():
        return {}

    try:
        normalized_path = str(source_path.resolve())
    except Exception:
        normalized_path = str(source_path)

    visited_set = set(visited or set())
    if normalized_path in visited_set:
        return {}
    visited_set.add(normalized_path)

    try:
        ok, _msg, nested_info = validate_model_file(source_path, _visited=visited_set)
    except Exception:
        return {}
    if not ok or not isinstance(nested_info, dict):
        return {}

    resolved: Dict = {}
    for key_name in (
        "yolo_family",
        "yolo_version",
        "yolo_size",
        "yolo_variant",
        "model_scale",
        "architecture_label",
    ):
        value = nested_info.get(key_name)
        if value:
            resolved[key_name] = value

    if not resolved:
        nested_source_arch = str(nested_info.get("source_architecture_label") or "").strip()
        if nested_source_arch:
            resolved["architecture_label"] = nested_source_arch
    return resolved


def validate_yolo_dataset(dataset_path: Path) -> Tuple[bool, str, Dict]:
    """Waliduje dataset YOLO."""
    stats = {
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "train_labels": 0,
        "val_labels": 0,
        "test_labels": 0,
        "total_images": 0,
        "total_labels": 0,
        "config": {},
        "warnings": []
    }
    
    if not dataset_path.exists():
        return False, "Folder nie istnieje", stats
    
    if not dataset_path.is_dir():
        return False, "Ścieżka nie jest folderem", stats
    
    # Znajdź data.yaml
    yaml_file = dataset_path / "data.yaml"
    if not yaml_file.exists():
        return False, "Brak pliku data.yaml", stats
    
    # Parsuj YAML
    try:
        config = safe_load_yaml(yaml_file)
        stats["config"] = config
    except Exception as e:
        stats["warnings"].append(f"Błąd parsowania data.yaml: {e}")
    
    # Sprawdź foldery
    for split in ["train", "val", "test"]:
        images_dir = dataset_path / "images" / split
        labels_dir = dataset_path / "labels" / split
        
        if images_dir.exists():
            img_count = sum(1 for f in images_dir.iterdir() 
                          if f.suffix.lower() in CONFIG.IMAGE_EXTENSIONS)
            stats[f"{split}_images"] = img_count
            stats["total_images"] += img_count
        
        if labels_dir.exists():
            lbl_count = sum(1 for f in labels_dir.iterdir() 
                          if f.suffix.lower() == '.txt')
            stats[f"{split}_labels"] = lbl_count
            stats["total_labels"] += lbl_count
    
    if stats["total_images"] == 0:
        return False, "Brak obrazów w dataset", stats
    
    if stats["total_labels"] == 0:
        return False, "Brak plików etykiet", stats
    
    return True, "Dataset OK", stats


def validate_model_file(
    model_path: Path,
    _visited: set[str] | None = None,
    *,
    prefer_sidecar: bool = True,
    allow_heavy_load: bool = True,
    write_sidecar: bool = True,
) -> Tuple[bool, str, Dict]:
    """Waliduje plik modelu .pt."""
    info = _empty_model_info()

    if not model_path.exists():
        return False, "Plik nie istnieje", info

    if model_path.suffix.lower() != ".pt":
        return False, "Plik musi mieć rozszerzenie .pt", info

    info["file_name"] = str(model_path.name or "").strip()
    info.update(_infer_yolo_identity_from_text(model_path.name))

    try:
        info["file_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 2)
    except Exception:
        pass

    cache_key = _build_model_cache_key(model_path)
    if cache_key is not None:
        cached = _MODEL_VALIDATION_CACHE.get(cache_key)
        if cached is not None:
            ok, message, cached_info = cached
            return ok, message, _clone_model_info(cached_info)

    if prefer_sidecar:
        sidecar_result = read_model_metadata_sidecar(model_path)
        if sidecar_result is not None:
            ok, message, sidecar_info = sidecar_result
            if ok and cache_key is not None:
                _MODEL_VALIDATION_CACHE[cache_key] = (ok, message, _clone_model_info(sidecar_info))
            if ok or not allow_heavy_load:
                return ok, message, _clone_model_info(sidecar_info)
            info.update(_clone_model_info(sidecar_info))

    if not allow_heavy_load:
        return False, "No lightweight model metadata sidecar", info

    if not YOLO_AVAILABLE:
        return False, "YOLO niedostępny", info
    YoloClass = get_yolo_class()
    if YoloClass is None:
        return False, "YOLO niedostępny", info

    model = None
    try:
        model = YoloClass(str(model_path))

        if hasattr(model, "task"):
            info["task"] = str(model.task)
            info["type"] = str(model.task)

        if hasattr(model, "model") and hasattr(model.model, "kpt_shape"):
            info["type"] = "pose"
            info["keypoints"] = True
            info["kpt_shape"] = list(model.model.kpt_shape)

        inner_model = getattr(model, "model", None)
        yaml_meta = getattr(inner_model, "yaml", None)
        if isinstance(yaml_meta, dict):
            info["yaml_file"] = str(yaml_meta.get("yaml_file") or "").strip()
            info["model_scale"] = str(yaml_meta.get("scale") or "").strip().lower()

        ckpt = getattr(model, "ckpt", None)
        if isinstance(ckpt, dict):
            info["ultralytics_version"] = str(ckpt.get("version") or "").strip()
            train_args = ckpt.get("train_args")
            if isinstance(train_args, dict):
                info["source_model"] = str(train_args.get("model") or "").strip()
                if info["source_model"]:
                    try:
                        info["source_model_name"] = Path(info["source_model"]).name
                    except Exception:
                        info["source_model_name"] = info["source_model"]
                    source_identity = _infer_yolo_identity_from_text(
                        info["source_model_name"] or info["source_model"],
                        task_hint=str(info.get("task") or info.get("type") or ""),
                    )
                    if not source_identity:
                        source_identity = _resolve_nested_source_model_identity(
                            info["source_model"],
                            task_hint=str(info.get("task") or info.get("type") or ""),
                            visited=_visited,
                        )
                    info["source_architecture_label"] = str(source_identity.get("architecture_label") or "").strip()

        task_hint = str(info.get("task") or info.get("type") or "").strip()
        for candidate in (
            info.get("yaml_file"),
            info.get("source_model_name"),
            info.get("source_model"),
            model_path.name,
        ):
            inferred = _infer_yolo_identity_from_text(candidate, task_hint=task_hint)
            if inferred:
                info.update({key: value for key, value in inferred.items() if value})
                break

        if not str(info.get("architecture_label") or "").strip() and info.get("source_model"):
            nested_identity = _resolve_nested_source_model_identity(
                info.get("source_model"),
                task_hint=task_hint,
                visited=_visited,
            )
            if nested_identity:
                info.update({key: value for key, value in nested_identity.items() if value})

        if hasattr(model, "names"):
            if isinstance(model.names, dict):
                info["classes"] = list(model.names.values())
            else:
                info["classes"] = list(model.names)
            info["num_classes"] = len(info["classes"])

        try:
            model_obj = getattr(model, "model", None)
            for source in (model_obj, model):
                parameters = getattr(source, "parameters", None)
                if not callable(parameters):
                    continue
                parameter_count = int(sum(int(param.numel()) for param in parameters()))
                if parameter_count > 0:
                    info["parameter_count"] = parameter_count
                    info["parameters_millions"] = round(parameter_count / 1_000_000.0, 3)
                    break
        except Exception:
            pass

        ok, message = True, f"Model {info['type'].upper()} OK"
        if write_sidecar:
            write_model_metadata_sidecar(
                model_path,
                info,
                validation_ok=ok,
                validation_message=message,
            )
        if cache_key is not None:
            _MODEL_VALIDATION_CACHE[cache_key] = (ok, message, _clone_model_info(info))
        return ok, message, info

    except Exception as e:
        message = f"Błąd ładowania: {e}"
        if cache_key is not None:
            _MODEL_VALIDATION_CACHE[cache_key] = (False, message, _clone_model_info(info))
        return False, message, info

    finally:
        if model:
            del model
        cleanup_gpu_memory()


def validate_cvat_xml(xml_path: Path) -> Tuple[bool, str, Dict]:
    """Waliduje plik CVAT XML."""
    stats = {
        "images": 0,
        "vehicles": 0,
        "plates": 0,
        "plates_with_4_points": 0,
        "boxes": 0,
        "polygons": 0,
        "labels": []
    }
    
    if not xml_path.exists():
        return False, "Plik nie istnieje", stats
    
    labels_seen: Set[str] = set()
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for image in root.findall('.//image'):
            stats["images"] += 1
            
            for box in image.findall('box'):
                stats["boxes"] += 1
                label = box.get('label', '').lower()
                labels_seen.add(label)
                if label in CONFIG.VEHICLE_LABELS:
                    stats["vehicles"] += 1
            
            for poly in image.findall('polygon'):
                stats["polygons"] += 1
                label = poly.get('label', '').lower()
                labels_seen.add(label)
                
                if label in CONFIG.PLATE_LABELS:
                    points = poly.get('points', '').split(';')
                    if len(points) == 4:
                        stats["plates"] += 1
                        stats["plates_with_4_points"] += 1
        
        stats["labels"] = sorted(labels_seen)
        
        if stats["images"] == 0:
            return False, "Brak obrazów w pliku", stats
        
        return True, "Plik CVAT XML OK", stats
        
    except ET.ParseError as e:
        return False, f"Błąd parsowania: {e}", stats


def validate_coco_file(coco_path: Path) -> Tuple[bool, str, Dict]:
    """Waliduje plik COCO JSON."""
    stats = {
        "images": 0,
        "annotations": 0,
        "categories": {}
    }
    
    if not coco_path.exists():
        return False, "Plik nie istnieje", stats
    
    try:
        with open(coco_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats["images"] = len(data.get("images", []))
        stats["annotations"] = len(data.get("annotations", []))
        
        for cat in data.get("categories", []):
            name = cat.get("name", "unknown")
            stats["categories"][name] = 0
        
        for ann in data.get("annotations", []):
            cat_id = ann.get("category_id")
            for cat in data.get("categories", []):
                if cat.get("id") == cat_id:
                    name = cat.get("name", "unknown")
                    stats["categories"][name] = stats["categories"].get(name, 0) + 1
        
        if stats["images"] == 0:
            return False, "Brak obrazów", stats
        
        return True, "Plik COCO OK", stats
        
    except Exception as e:
        return False, f"Błąd: {e}", stats
