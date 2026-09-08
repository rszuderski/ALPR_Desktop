#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Android ALPR ``.alprmodel`` packages from trained YOLO checkpoints."""

from __future__ import annotations

from contextlib import contextmanager
import datetime as _dt
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, Mapping

try:
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover - fallback for lean Python environments with pip only.
    from pip._vendor.packaging.requirements import Requirement  # type: ignore


MobileRole = Literal["vehicle", "plate", "character"]
MobileFormat = Literal["litert", "onnx", "ncnn"]

MOBILE_MODEL_SCHEMA = "alpr.model.v1"
MOBILE_ALPR_PACKAGE_SCHEMA = "alpr.package.v1"
COMPLETE_ALPR_MODELS_REQUIRED = (
    "Kompletny pakiet ALPR wymaga modeli MT i MZ. "
    "Jeżeli chcesz wyeksportować pojedynczy model, użyj eksportu modelu mobilnego."
)
MAX_PACKAGE_ENTRIES = 256
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_ALPR_PACKAGE_ENTRIES = 640
MAX_ALPR_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
SAFE_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
CALIBRATION_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ONNX_INT8_CALIBRATION_MAX_IMAGES = 300

MOBILE_EXPORT_CORE_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("ultralytics", "ultralytics"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
)
MOBILE_EXPORT_ONNX_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("onnx", "onnx>=1.12.0,<2.0.0"),
    ("onnxruntime", "onnxruntime"),
    ("onnxslim", "onnxslim>=0.1.71"),
)
MOBILE_EXPORT_ONNX_INT8_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("PIL", "Pillow"),
    ("yaml", "PyYAML"),
)
MOBILE_EXPORT_TFLITE_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("tensorflow", "tensorflow>=2.0.0,<=2.19.0"),
    ("tf_keras", "tf_keras<=2.19.0"),
    ("sng4onnx", "sng4onnx>=1.0.1"),
    ("onnx_graphsurgeon", "onnx_graphsurgeon>=0.3.26"),
    ("ai_edge_litert", "ai-edge-litert>=1.2.0"),
    ("onnx", "onnx>=1.12.0,<2.0.0"),
    ("onnx2tf", "onnx2tf>=1.26.3,<1.29.0"),
    ("onnxslim", "onnxslim>=0.1.71"),
    ("onnxruntime", "onnxruntime"),
    ("google.protobuf", "protobuf>=5"),
)
MOBILE_EXPORT_NCNN_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("ncnn", "ncnn"),
    ("pnnx", "pnnx"),
)


class MobileExportError(RuntimeError):
    """Raised when a mobile model package cannot be built or validated."""


@dataclass(frozen=True)
class MobileExportRequest:
    checkpoint: Path
    destination: Path
    role: MobileRole
    formats: tuple[MobileFormat, ...]
    image_size: int | tuple[int, int]
    quantizations: tuple[str, ...] = ("fp32",)
    format_quantizations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    calibration_data: Path | None = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    model_id: str = ""
    name: str = ""
    version: str = "1"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MobileAlprPackageRequest:
    destination: Path
    vehicle_package: Path | None = None
    plate_package: Path | None = None
    character_package: Path | None = None
    vehicle_request: MobileExportRequest | None = None
    plate_request: MobileExportRequest | None = None
    character_request: MobileExportRequest | None = None
    package_id: str = ""
    name: str = ""
    version: str = "1"
    ranking_dataset: dict[str, Any] = field(default_factory=dict)
    calibration_dataset: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportedVariant:
    id: str
    runtime: str
    precision: str
    files: tuple[Path, ...]
    relative_files: tuple[str, ...]
    input_spec: dict
    output_spec: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


_YOLO_EXPORT_RUNTIME_CACHE: tuple[tuple[tuple[str, str], ...], str | None] | None = None


def _module_version(name: str) -> str:
    try:
        return str(importlib_metadata.version(name))
    except Exception:
        return "?"


def mobile_export_requirement_status(spec: str, import_name: str = "") -> tuple[bool, str]:
    text = str(spec or "").strip()
    module_name = str(import_name or "").strip()
    if not text:
        return False, "Puste wymaganie eksportu."
    try:
        requirement = Requirement(text)
    except Exception:
        available = _module_available(module_name or text)
        return (
            bool(available),
            f"{module_name or text}: {'dostępne' if available else 'brak'}",
        )

    dist_name = requirement.name
    try:
        version = importlib_metadata.version(dist_name)
    except Exception:
        if module_name and _module_available(module_name):
            return True, f"{dist_name}: moduł dostępny, wersja nie została odczytana."
        return False, f"Brak pakietu {dist_name}."

    if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
        return False, f"{dist_name}=={version} nie spełnia wymagania {requirement.specifier}."
    if module_name and not _module_available(module_name):
        return False, f"{dist_name}=={version}, ale nie można zaimportować modułu {module_name}."
    return True, f"{dist_name}=={version}"


def mobile_export_required_specs(
    formats: tuple[str, ...] | list[str] | set[str],
    *,
    quantizations: tuple[str, ...] | list[str] | set[str] | None = None,
    format_quantizations: dict[str, tuple[str, ...] | list[str] | set[str]] | None = None,
) -> list[dict[str, str]]:
    selected = set(str(item or "").strip().lower() for item in (formats or ()))
    rows: dict[str, dict[str, str]] = {}

    def add(scope: str, items: tuple[tuple[str, str], ...]) -> None:
        for module_name, spec in items:
            row = rows.setdefault(spec, {"module": module_name, "spec": spec, "scope": ""})
            scopes = {part.strip() for part in str(row.get("scope") or "").split(",") if part.strip()}
            scopes.add(scope)
            row["scope"] = ", ".join(sorted(scopes))

    add("YOLO", MOBILE_EXPORT_CORE_REQUIREMENTS)
    if "onnx" in selected:
        add("ONNX", MOBILE_EXPORT_ONNX_REQUIREMENTS)
        raw_map = format_quantizations if isinstance(format_quantizations, dict) else {}
        onnx_precisions = set(
            str(item or "").strip().lower()
            for item in (raw_map.get("onnx") or raw_map.get("ONNX") or ())
            if str(item or "").strip()
        )
        if not onnx_precisions and quantizations is not None:
            onnx_precisions = {
                str(item or "").strip().lower()
                for item in (quantizations or ())
                if str(item or "").strip()
            }
        if "int8" in onnx_precisions:
            add("ONNX INT8", MOBILE_EXPORT_ONNX_INT8_REQUIREMENTS)
    if "litert" in selected:
        add("LiteRT/TFLite", MOBILE_EXPORT_TFLITE_REQUIREMENTS)
    if "ncnn" in selected:
        add("NCNN", MOBILE_EXPORT_NCNN_REQUIREMENTS)
    return list(rows.values())


def check_mobile_yolo_export_runtime() -> str | None:
    """Return a human-readable problem when YOLO export dependencies are broken."""
    global _YOLO_EXPORT_RUNTIME_CACHE
    module_names = ("torch", "torchvision", "ultralytics")
    signature = tuple((name, _module_version(name)) for name in module_names)
    if _YOLO_EXPORT_RUNTIME_CACHE and _YOLO_EXPORT_RUNTIME_CACHE[0] == signature:
        return _YOLO_EXPORT_RUNTIME_CACHE[1]

    versions = ", ".join(f"{name}={version}" for name, version in signature)
    problem: str | None = None
    try:
        importlib.import_module("torch")
        importlib.import_module("torchvision")
        ultralytics = importlib.import_module("ultralytics")
        yolo_class = getattr(ultralytics, "YOLO", None)
        if yolo_class is None:
            problem = f"Runtime YOLO jest niekompletny: pakiet ultralytics nie udostępnia klasy YOLO ({versions})."
    except Exception as exc:
        problem = (
            "Niespójne zależności YOLO: nie można zaimportować ultralytics/torch/torchvision. "
            f"Zainstalowane wersje: {versions}. Szczegóły: {exc}"
        )

    _YOLO_EXPORT_RUNTIME_CACHE = (signature, problem)
    return problem


def _tflite_interpreter_class():
    tensorflow_error = None
    try:
        tensorflow = importlib.import_module("tensorflow")
        lite_module = getattr(tensorflow, "lite", None)
        interpreter_cls = getattr(lite_module, "Interpreter", None)
        if interpreter_cls is not None:
            return interpreter_cls
    except Exception as exc:
        tensorflow_error = exc

    try:
        module = importlib.import_module("tensorflow.lite.python.interpreter")
        interpreter_cls = getattr(module, "Interpreter", None)
        if interpreter_cls is not None:
            return interpreter_cls
    except Exception as exc:
        tensorflow_error = tensorflow_error or exc

    try:
        module = importlib.import_module("tflite_runtime.interpreter")
        interpreter_cls = getattr(module, "Interpreter", None)
        if interpreter_cls is not None:
            return interpreter_cls
    except Exception as exc:
        detail = f"tflite_runtime: {exc}"
        if tensorflow_error is not None:
            detail = f"tensorflow: {tensorflow_error}; {detail}"
        raise MobileExportError(f"Brak interpretera TFLite do inspekcji wariantu: {detail}") from exc

    raise MobileExportError("Brak interpretera TFLite do inspekcji wariantu.")


def check_mobile_tflite_inspection_runtime() -> str | None:
    try:
        _tflite_interpreter_class()
    except Exception as exc:
        return str(exc)
    return None


def check_mobile_onnx_int8_runtime() -> str | None:
    missing = []
    required = (
        ("onnx", "onnx"),
        ("onnxruntime.quantization", "onnxruntime.quantization"),
        ("numpy", "numpy"),
        ("PIL", "Pillow/PIL"),
        ("yaml", "PyYAML"),
    )
    for module_name, label in required:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            missing.append(f"{label}: {exc}")
    if missing:
        return "Brak runtime kwantyzacji ONNX INT8: " + "; ".join(missing)
    return None


@contextmanager
def _controlled_ultralytics_export_runtime():
    """Keep Ultralytics export deterministic: no hidden pip AutoUpdate, CPU checks only."""
    previous_skip = os.environ.get("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS")
    os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"
    patched_cuda = None
    original_is_available = None
    try:
        try:
            torch_mod = importlib.import_module("torch")
            cuda_obj = getattr(torch_mod, "cuda", None)
            original_is_available = getattr(cuda_obj, "is_available", None)
            if cuda_obj is not None and callable(original_is_available):
                setattr(cuda_obj, "is_available", lambda: False)
                patched_cuda = cuda_obj
        except Exception:
            patched_cuda = None
        yield
    finally:
        if patched_cuda is not None and original_is_available is not None:
            try:
                setattr(patched_cuda, "is_available", original_is_available)
            except Exception:
                pass
        if previous_skip is None:
            os.environ.pop("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", None)
        else:
            os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = previous_skip


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (Path, PurePosixPath)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _path_reproducibility_payload(path: Path | str | None, *, include_hash: bool = True) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        safe_path = Path(path).resolve()
    except Exception:
        text = str(path or "").strip()
        if not text:
            return None
        return {"path": text, "exists": False}
    payload: dict[str, Any] = {
        "path": str(safe_path),
        "name": safe_path.name,
        "exists": safe_path.exists(),
        "is_file": safe_path.is_file(),
        "is_dir": safe_path.is_dir(),
    }
    if include_hash and safe_path.exists() and safe_path.is_file():
        try:
            payload["sha256"] = sha256_file(safe_path)
        except Exception as exc:
            payload["sha256_error"] = str(exc)
    return payload


def _runtime_toolchain_payload() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "ultralytics": _module_version("ultralytics"),
        "torch": _module_version("torch"),
        "torchvision": _module_version("torchvision"),
        "onnx": _module_version("onnx"),
        "onnxruntime": _module_version("onnxruntime"),
        "tensorflow": _module_version("tensorflow"),
        "ai_edge_litert": _module_version("ai-edge-litert"),
        "onnx2tf": _module_version("onnx2tf"),
        "ncnn": _module_version("ncnn"),
    }


def _normalized_export_formats(formats: tuple[Any, ...] | list[Any] | set[Any] | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item or "").strip().lower() for item in (formats or ()) if str(item or "").strip()))


def _normalized_export_quantizations(
    quantizations: tuple[Any, ...] | list[Any] | set[Any] | None,
    *,
    default: tuple[str, ...] = ("fp32",),
) -> tuple[str, ...]:
    normalized = tuple(
        dict.fromkeys(str(item or "").strip().lower() for item in (quantizations or ()) if str(item or "").strip())
    )
    return normalized or default


def _request_format_quantization_map(request: MobileExportRequest) -> dict[str, tuple[str, ...]]:
    formats = _normalized_export_formats(request.formats)
    global_quantizations = _normalized_export_quantizations(request.quantizations)
    raw_map = request.format_quantizations if isinstance(getattr(request, "format_quantizations", None), dict) else {}
    supported: dict[str, tuple[str, ...]] = {
        "litert": ("fp32", "int8"),
        "onnx": ("fp32", "int8"),
        "ncnn": ("fp32",),
    }
    result: dict[str, tuple[str, ...]] = {}
    for fmt in formats:
        aliases = [fmt]
        if fmt == "litert":
            aliases.append("tflite")
        raw_values = None
        for alias in aliases:
            if alias in raw_map:
                raw_values = raw_map.get(alias)
                break
        requested = _normalized_export_quantizations(raw_values, default=global_quantizations) if raw_values is not None else global_quantizations
        allowed = supported.get(fmt, ("fp32",))
        precisions = tuple(item for item in requested if item in allowed)
        result[fmt] = precisions or ("fp32",)
    return result


def _request_int8_calibration_targets(request: MobileExportRequest) -> list[str]:
    labels = {"litert": "litert:int8", "onnx": "onnx:int8"}
    targets: list[str] = []
    for fmt, precisions in _request_format_quantization_map(request).items():
        if "int8" in precisions and fmt in labels:
            targets.append(labels[fmt])
    return targets


def _export_request_payload(request: MobileExportRequest) -> dict[str, Any]:
    formats = list(_normalized_export_formats(request.formats))
    quantizations = list(
        _normalized_export_quantizations(request.quantizations, default=())
    )
    format_quantizations = _request_format_quantization_map(request)
    calibration_payload = _path_reproducibility_payload(request.calibration_data)
    return {
        "formats": formats,
        "quantizations": quantizations,
        "format_quantizations": {key: list(value) for key, value in format_quantizations.items()},
        "image_size": _json_safe_value(request.image_size),
        "confidence_threshold": float(request.confidence_threshold),
        "iou_threshold": float(request.iou_threshold),
        "calibration_data": calibration_payload,
        "calibration_required_for": _request_int8_calibration_targets(request),
    }


def _single_model_reproducibility_payload(
    *,
    request: MobileExportRequest,
    model_id: str,
    role: str,
    task: str,
    labels: list[str],
    checkpoint: Path,
    model_info: dict,
    metadata: dict[str, Any],
    source_meta: dict[str, Any],
    variants: list[dict[str, Any]],
) -> dict[str, Any]:
    model_meta = dict(metadata.get("model") or {})
    candidate = dict(metadata.get("candidate") or {})
    return _json_safe_value(
        {
            "schema": "alpr.export.reproducibility.v1",
            "kind": "single_model_export",
            "created_at": source_meta.get("exported_at") or _utc_now_iso(),
            "model": {
                "model_id": model_id,
                "name": request.name or model_id,
                "version": str(request.version or "1"),
                "role": role,
                "task": task,
                "display_id": model_meta.get("display_id") or candidate.get("model_label") or "",
                "architecture_label": model_meta.get("architecture_label") or candidate.get("model_version") or "",
            },
            "request": _export_request_payload(request),
            "ui_export_settings": dict(metadata.get("export_settings") or {}),
            "source_checkpoint": _path_reproducibility_payload(checkpoint),
            "candidate": candidate,
            "training": dict(metadata.get("training") or {}),
            "metrics": dict(metadata.get("metrics") or {}),
            "model_metadata": model_meta,
            "model_info": dict(model_info or {}),
            "labels": {
                "count": len(labels),
                "items": list(labels),
            },
            "variants": variants,
            "toolchain": _runtime_toolchain_payload(),
        }
    )


def _child_request_reproducibility_payload(request: MobileExportRequest | None) -> dict[str, Any] | None:
    if request is None:
        return None
    return _json_safe_value(
        {
            "model_id": request.model_id,
            "name": request.name,
            "version": request.version,
            "role": request.role,
            "checkpoint": _path_reproducibility_payload(request.checkpoint),
            "destination": _path_reproducibility_payload(request.destination, include_hash=False),
            "request": _export_request_payload(request),
            "candidate": dict((request.metadata or {}).get("candidate") or {}),
            "training": dict((request.metadata or {}).get("training") or {}),
            "metrics": dict((request.metadata or {}).get("metrics") or {}),
            "model": dict((request.metadata or {}).get("model") or {}),
            "vehicle_detection": dict((request.metadata or {}).get("vehicle_detection") or {}),
        }
    )


def _package_model_source_payload(
    *,
    role: str,
    item: dict[str, Any] | None,
    request: MobileExportRequest | None,
    source_package: Path | None,
) -> dict[str, Any]:
    model_item = dict(item or {})
    payload: dict[str, Any] = {
        "role": role,
        "source_kind": "fresh_export" if request is not None else "existing_package",
        "model_id": str(model_item.get("model_id") or getattr(request, "model_id", "") or ""),
        "name": str(model_item.get("name") or getattr(request, "name", "") or ""),
        "version": str(model_item.get("version") or getattr(request, "version", "") or ""),
        "task": str(model_item.get("task") or ""),
        "package_file": str(model_item.get("package_file") or ""),
        "manifest_file": str(model_item.get("manifest_file") or ""),
        "variants": list(model_item.get("variants") or []),
        "variant_count": int(model_item.get("variant_count") or 0),
        "labels": {
            "count": int(model_item.get("label_count") or len(list(model_item.get("labels") or []))),
            "items": list(model_item.get("labels") or []),
        },
        "training": dict(model_item.get("training") or {}),
        "metrics": dict(model_item.get("metrics") or {}),
        "source": dict(model_item.get("source") or {}),
        "model": dict(model_item.get("model") or {}),
    }
    child_repro = model_item.get("export_reproducibility")
    if isinstance(child_repro, dict) and child_repro:
        payload["child_export_reproducibility"] = child_repro
    request_payload = _child_request_reproducibility_payload(request)
    if request_payload:
        payload["requested_export"] = request_payload
    package_payload = _path_reproducibility_payload(source_package)
    if package_payload:
        payload["source_package"] = package_payload
    if role == "vehicle":
        vehicle_detection = dict(model_item.get("vehicle_detection") or {})
        if vehicle_detection:
            payload["vehicle_detection"] = vehicle_detection
    return _json_safe_value(payload)


def _alpr_package_reproducibility_payload(
    *,
    request: MobileAlprPackageRequest,
    package_id: str,
    created_at: str,
    models: dict[str, dict[str, Any]],
    pipeline: list[dict[str, Any]],
) -> dict[str, Any]:
    role_requests = {
        "vehicle": request.vehicle_request,
        "plate": request.plate_request,
        "character": request.character_request,
    }
    role_packages = {
        "vehicle": request.vehicle_package,
        "plate": request.plate_package,
        "character": request.character_package,
    }
    role_order = [role for role in ("vehicle", "plate", "character") if role in models]
    return _json_safe_value(
        {
            "schema": "alpr.package.reproducibility.v1",
            "kind": "alpr_mobile_package_export",
            "created_at": created_at,
            "package": {
                "package_id": package_id,
                "name": request.name or package_id,
                "version": str(request.version or "1"),
                "destination": _path_reproducibility_payload(request.destination, include_hash=False),
                "roles": role_order,
                "pipeline_variant": "+".join({"vehicle": "MP", "plate": "MT", "character": "MZ"}[role] for role in role_order),
            },
            "pipeline": pipeline,
            "models": {
                role: _package_model_source_payload(
                    role=role,
                    item=models.get(role),
                    request=role_requests.get(role),
                    source_package=role_packages.get(role),
                )
                for role in role_order
            },
            "ranking_dataset": dict(request.ranking_dataset or {}),
            "calibration_dataset": dict(request.calibration_dataset or {}),
            "package_metadata": dict(request.metadata or {}),
            "toolchain": _runtime_toolchain_payload(),
        }
    )


def _safe_id(value: str, fallback: str = "alpr-model") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip(".-_")
    if not text:
        text = fallback
    if not re.match(r"^[A-Za-z0-9]", text):
        text = f"m-{text}"
    return text[:80]


def _normalize_role(role: str) -> MobileRole:
    raw = str(role or "").strip().lower()
    if raw in {"plate", "plates", "pose", "tablica", "tablice"}:
        return "plate"
    if raw in {"char", "chars", "character", "characters", "ocr", "znak", "znaki"}:
        return "character"
    if raw in {"vehicle", "vehicles", "pojazd", "pojazdy", "car", "cars"}:
        return "vehicle"
    raise MobileExportError(f"Nieobsługiwana rola modelu mobilnego: {role}")


_DEFAULT_VEHICLE_LABELS = ("car", "motorcycle", "bus", "truck", "van", "vehicle")
_DEFAULT_COCO_VEHICLE_CLASS_INDICES = (2, 3, 5, 7)

def _normalize_class_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())

def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items: Any = value
    if isinstance(value, str):
        raw_items = re.split(r"[,;\n]+", value)
    elif isinstance(value, dict):
        raw_items = value.values()
    elif not isinstance(value, (list, tuple, set)):
        raw_items = [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item or "").strip()
        key = _normalize_class_token(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result

def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    raw_items: Any = value
    if isinstance(value, str):
        raw_items = re.split(r"[,;\s]+", value)
    elif not isinstance(value, (list, tuple, set)):
        raw_items = [value]
    result: list[int] = []
    seen: set[int] = set()
    for item in raw_items:
        try:
            number = int(item)
        except Exception:
            continue
        if number < 0 or number in seen:
            continue
        seen.add(number)
        result.append(number)
    return result

def _build_vehicle_detection_contract(metadata: dict, labels: list[str]) -> dict[str, Any]:
    vehicle_meta = dict(metadata.get("vehicle_detection") or {}) if isinstance(metadata, dict) else {}
    labels_list = [str(label or "").strip() for label in list(labels or []) if str(label or "").strip()]
    include_labels = _coerce_string_list(
        vehicle_meta.get("include_labels")
        or vehicle_meta.get("class_labels")
        or vehicle_meta.get("labels")
        or vehicle_meta.get("vehicle_labels")
    )
    if not include_labels:
        include_labels = list(_DEFAULT_VEHICLE_LABELS)
    explicit_indices = set(
        _coerce_int_list(
            vehicle_meta.get("include_class_indices")
            or vehicle_meta.get("class_indices")
            or vehicle_meta.get("class_ids")
        )
    )
    allowed = {_normalize_class_token(label) for label in include_labels}
    matched_indices = {
        index
        for index, label in enumerate(labels_list)
        if _normalize_class_token(label) in allowed
    }
    if labels_list:
        include_indices = sorted(index for index in (explicit_indices | matched_indices) if index < len(labels_list))
        if not include_indices and len(labels_list) <= 3 and not explicit_indices:
            include_indices = list(range(len(labels_list)))
    else:
        include_indices = sorted(explicit_indices)
    fallback_coco = _coerce_int_list(vehicle_meta.get("fallback_coco_class_indices"))
    if not fallback_coco:
        fallback_coco = list(_DEFAULT_COCO_VEHICLE_CLASS_INDICES)
    contract = dict(vehicle_meta)
    contract.update(
        {
            "filter_mode": "include",
            "include_labels": include_labels,
            "include_class_indices": include_indices,
            "fallback_coco_class_indices": fallback_coco,
            "label_matching": "case_insensitive_normalized",
            "next_stage": "plate_detection",
            "available_labels": labels_list,
            "available_label_count": len(labels_list),
        }
    )
    return contract


def _role_task(role: str) -> str:
    return "pose" if _normalize_role(role) == "plate" else "detect"


def _decoder_for_task(task: str, *, end2end_output: bool = False) -> str:
    task_name = "pose" if str(task).lower() == "pose" else "detect"
    suffix = "end2end_v1" if end2end_output else "raw_v1"
    return f"ultralytics_{task_name}_{suffix}"


def expected_yolo_output_attributes(
    *,
    output_format: str,
    class_count: int,
    keypoint_count: int,
    keypoint_dimensions: int,
    has_objectness: bool = False,
) -> int:
    """Number of attributes per anchor/detection, independent of tensor layout."""
    keypoints = int(keypoint_count) * int(keypoint_dimensions)
    if output_format == "raw_yolo":
        return 4 + int(has_objectness) + int(class_count) + keypoints
    if output_format == "end2end_detections":
        return 6 + keypoints
    raise MobileExportError(f"Nieobsługiwany format wyjścia YOLO: {output_format}")


def _resolve_ncnn_output_spec(
    *, task: str, class_count: int, keypoint_count: int, keypoint_dimensions: int,
    confidence_threshold: float, iou_threshold: float,
) -> dict:
    # Ultralytics/PNNX disables the end-to-end branch for NCNN (no TopK).
    # This describes the exported artifact, not the original checkpoint head.
    return {
        "decoder": _decoder_for_task(task),
        "output_format": "raw_yolo",
        "class_count": int(class_count),
        "keypoint_count": int(keypoint_count),
        "keypoint_dimensions": int(keypoint_dimensions),
        "end2end_output": False,
        "has_objectness": False,
        "tensor_layout": "channels_first",
        "box_format": "xywh",
        "normalized_coordinates": False,
        "nms_in_graph": False,
        "nms_required": True,
        "confidence_threshold": float(confidence_threshold),
        "iou_threshold": float(iou_threshold),
    }


def _validate_variant_output_contract(runtime: str, output: dict, *, task: str, class_count: int) -> None:
    """Validate the runtime's complete output contract."""
    output_format = output.get("output_format")
    if output_format not in {"raw_yolo", "end2end_detections"}:
        raise MobileExportError(f"Wariant {runtime}: nieprawidłowy output_format={output_format!r}.")
    end2end = output_format == "end2end_detections"
    if output.get("decoder") != _decoder_for_task(task, end2end_output=end2end):
        raise MobileExportError(f"Wariant {runtime}: decoder jest niezgodny z task={task} i output_format={output_format}.")
    try:
        counts = [output.get(name, 0) for name in ("class_count", "keypoint_count", "keypoint_dimensions")]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise ValueError("dimensions must be integers")
        classes, keypoints, dimensions = counts
        if classes != class_count or classes <= 0:
            raise ValueError("class_count differs from labels")
        if task == "pose" and (keypoints < 4 or dimensions not in {2, 3}):
            raise ValueError("pose needs at least 4 keypoints with 2 or 3 dimensions")
        if task == "detect" and (keypoints != 0 or dimensions != 0):
            raise ValueError("detect cannot have keypoints")
    except (TypeError, ValueError) as exc:
        raise MobileExportError(f"Wariant {runtime}: nieprawidłowe wymiary wyjścia YOLO ({exc}).") from exc
    expected = {
        "box_format": "xyxy" if end2end else "xywh",
        "nms_in_graph": False,
        "nms_required": not end2end,
    }
    if "end2end_output" in output:
        expected["end2end_output"] = end2end
    if end2end:
        expected.update(has_objectness=False, tensor_layout="detections_first", score_index=4, class_index=5)
    elif output.get("tensor_layout") not in {"channels_first", "anchors_first"}:
        raise MobileExportError(f"Wariant {runtime}: nieprawidłowy tensor_layout dla RAW YOLO.")
    if runtime == "ncnn":
        expected.update(
            decoder=_decoder_for_task(task), output_format="raw_yolo", box_format="xywh",
            nms_required=True, has_objectness=False, tensor_layout="channels_first", normalized_coordinates=False,
        )
        if "end2end_output" in output:
            expected["end2end_output"] = False
    for name, value in expected.items():
        actual = output.get(name)
        if actual != value or (isinstance(value, bool) and not isinstance(actual, bool)):
            raise MobileExportError(f"Wariant {runtime}: {name}={actual!r}, wymagane {value!r}.")


def _normalize_image_size(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise MobileExportError("imgsz musi być liczbą albo parą (width, height).")
        width, height = int(value[0]), int(value[1])
    else:
        width = height = int(value)
    if width <= 0 or height <= 0:
        raise MobileExportError("imgsz musi mieć dodatnie wymiary.")
    return width, height


def _ultralytics_imgsz(value: int | tuple[int, int]) -> int | tuple[int, int]:
    width, height = _normalize_image_size(value)
    if width == height:
        return width
    return (height, width)


def _resolve_calibration_yaml_root(data_yaml: Path, payload: dict[str, Any]) -> Path:
    raw_root = payload.get("path") if isinstance(payload, dict) else None
    if raw_root:
        root = Path(str(raw_root))
        if not root.is_absolute():
            root = data_yaml.parent / root
        return root.resolve()
    return data_yaml.parent.resolve()


def _as_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "").strip()]
    return [str(value)] if str(value or "").strip() else []


def _resolve_calibration_sources(data_yaml: Path) -> list[Path]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise MobileExportError(f"Brak PyYAML do odczytu data.yaml kalibracji ONNX INT8: {exc}") from exc

    try:
        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise MobileExportError(f"Nie można odczytać data.yaml kalibracji ONNX INT8: {data_yaml} ({exc})") from exc
    if not isinstance(payload, dict):
        raise MobileExportError(f"Nieprawidłowy data.yaml kalibracji ONNX INT8: {data_yaml}")

    root = _resolve_calibration_yaml_root(data_yaml, payload)
    result: list[Path] = []
    for split in ("val", "train", "test"):
        for item in _as_path_list(payload.get(split)):
            path = Path(item)
            if not path.is_absolute():
                path = root / path
            result.append(path.resolve())
    return result or [data_yaml.parent.resolve()]


def _collect_calibration_images(data_yaml: Path, *, limit: int = ONNX_INT8_CALIBRATION_MAX_IMAGES) -> list[Path]:
    images: list[Path] = []
    seen: set[str] = set()
    for source in _resolve_calibration_sources(data_yaml):
        candidates: list[Path] = []
        if source.is_file():
            if source.suffix.lower() in CALIBRATION_IMAGE_SUFFIXES:
                candidates = [source]
            else:
                try:
                    candidates = [
                        Path(line.strip())
                        for line in source.read_text(encoding="utf-8", errors="ignore").splitlines()
                        if line.strip() and not line.strip().startswith("#")
                    ]
                except Exception:
                    candidates = []
        elif source.is_dir():
            candidates = [path for path in source.rglob("*") if path.suffix.lower() in CALIBRATION_IMAGE_SUFFIXES]
        for candidate in candidates:
            path = candidate if candidate.is_absolute() else (source.parent / candidate)
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            key = str(resolved).lower()
            if key in seen or not resolved.exists() or not resolved.is_file():
                continue
            seen.add(key)
            images.append(resolved)
            if len(images) >= limit:
                return images
    return images


def _onnx_input_name_and_size(onnx_path: Path, fallback_size: int | tuple[int, int]) -> tuple[str, tuple[int, int]]:
    try:
        import onnx  # type: ignore
    except Exception as exc:
        raise MobileExportError(f"Brak pakietu onnx do przygotowania kwantyzacji ONNX INT8: {exc}") from exc
    model = onnx.load(str(onnx_path))
    if not model.graph.input:
        raise MobileExportError("Model ONNX nie ma wejścia do kalibracji INT8.")
    input_value = model.graph.input[0]
    shape = [_shape_dim_value(dim) for dim in input_value.type.tensor_type.shape.dim]
    fallback_width, fallback_height = _normalize_image_size(fallback_size)
    width, height = fallback_width, fallback_height
    if len(shape) >= 4:
        try:
            if isinstance(shape[2], int) and int(shape[2]) > 0:
                height = int(shape[2])
            if isinstance(shape[3], int) and int(shape[3]) > 0:
                width = int(shape[3])
        except Exception:
            pass
    return str(input_value.name or "images"), (width, height)


class _OnnxImageCalibrationReader:
    def __init__(self, *, input_name: str, image_paths: list[Path], image_size: tuple[int, int]):
        self.input_name = input_name
        self.image_paths = list(image_paths)
        self.image_size = image_size
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None
        path = self.image_paths[self.index]
        self.index += 1
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise MobileExportError(f"Brak bibliotek obrazu do kalibracji ONNX INT8: {exc}") from exc
        width, height = self.image_size
        try:
            image = Image.open(path).convert("RGB").resize((width, height))
        except Exception as exc:
            raise MobileExportError(f"Nie można odczytać obrazu kalibracyjnego ONNX INT8: {path} ({exc})") from exc
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))[None, ...]
        return {self.input_name: array}

    def rewind(self) -> None:
        self.index = 0


def _dtype_name_from_onnx(elem_type: int) -> str:
    try:
        import onnx  # type: ignore

        mapping = {
            onnx.TensorProto.FLOAT: "FLOAT32",
            onnx.TensorProto.UINT8: "UINT8",
            onnx.TensorProto.INT8: "INT8",
        }
        return mapping.get(int(elem_type), str(elem_type))
    except Exception:
        return str(elem_type)


def _dtype_name_from_tflite(dtype_value) -> str:
    text = str(dtype_value or "").upper()
    if "FLOAT32" in text or "FLOAT" in text:
        return "FLOAT32"
    if "UINT8" in text:
        return "UINT8"
    if "INT8" in text:
        return "INT8"
    return text.replace("<CLASS 'NUMPY.", "").replace("'>", "") or "FLOAT32"


def _tflite_shape_list(detail: dict, key: str = "shape") -> list[int | str]:
    raw = detail.get(key) if isinstance(detail, dict) else None
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        try:
            raw = raw.tolist()
        except Exception:
            pass
    if isinstance(raw, (list, tuple)):
        items = raw
    else:
        try:
            items = list(raw)
        except Exception:
            items = [raw]
    result: list[int | str] = []
    for item in items:
        try:
            result.append(int(item))
        except Exception:
            result.append(str(item))
    return result


def _shape_dim_value(dim) -> int | str:
    value = getattr(dim, "dim_value", 0) or 0
    if value:
        return int(value)
    param = str(getattr(dim, "dim_param", "") or "").strip()
    return param or 0


def _shape_is_static(shape: list[int | str]) -> bool:
    return bool(shape) and all(isinstance(item, int) and int(item) > 0 for item in shape)


def _build_input_spec_from_shape(shape: list[int | str], dtype: str) -> dict:
    if len(shape) != 4 or not _shape_is_static(shape):
        raise MobileExportError(f"Wejście modelu musi mieć statyczny kształt 4D, otrzymano: {shape}")
    dims = [int(item) for item in shape]
    if dims[0] != 1:
        raise MobileExportError(f"Android obsługuje batch=1, otrzymano wejście: {shape}")
    if dims[3] == 3:
        layout = "NHWC"
        height, width = dims[1], dims[2]
    elif dims[1] == 3:
        layout = "NCHW"
        height, width = dims[2], dims[3]
    else:
        raise MobileExportError(f"Nie rozpoznano układu wejścia modelu: {shape}")
    if str(dtype).upper() not in {"FLOAT32", "UINT8", "INT8"}:
        raise MobileExportError(f"Nieobsługiwany typ wejścia modelu: {dtype}")
    return {
        "width": int(width),
        "height": int(height),
        "channels": 3,
        "layout": layout,
        "color": "RGB",
        "data_type": str(dtype).upper(),
        "scale": 0.0039215686,
        "offset": 0.0,
    }


def _infer_output_spec(
    shape: list[int | str],
    *,
    task: str,
    class_count: int,
    keypoint_count: int,
    keypoint_dimensions: int = 0,
    end2end_output: bool = False,
    normalized_coordinates: bool = False,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict:
    if len(shape) < 3 or not _shape_is_static(shape):
        raise MobileExportError(f"Wyjście YOLO musi mieć statyczny kształt co najmniej 3D, otrzymano: {shape}")
    dims = [int(item) for item in shape]
    if dims[0] != 1:
        raise MobileExportError(f"Android obsługuje batch=1, otrzymano wyjście: {shape}")

    normalized_keypoint_count = int(keypoint_count)
    preferred_keypoint_dimensions: list[int] = []
    if normalized_keypoint_count > 0:
        try:
            detected_dimensions = int(keypoint_dimensions)
        except Exception:
            detected_dimensions = 0
        if detected_dimensions > 0:
            preferred_keypoint_dimensions.append(detected_dimensions)
        for fallback_dimensions in (3, 2):
            if fallback_dimensions not in preferred_keypoint_dimensions:
                preferred_keypoint_dimensions.append(fallback_dimensions)
    else:
        preferred_keypoint_dimensions.append(0)

    def _end2end_spec(dimensions: int) -> dict:
        return {
            "decoder": _decoder_for_task(task, end2end_output=True),
            "output_format": "end2end_detections",
            "end2end_output": True,
            "class_count": int(class_count),
            "keypoint_count": normalized_keypoint_count,
            "keypoint_dimensions": int(dimensions),
            "has_objectness": False,
            "score_index": 4,
            "class_index": 5,
            "tensor_layout": "detections_first",
            "box_format": "xyxy",
            "normalized_coordinates": bool(normalized_coordinates),
            "nms_in_graph": False,
            "nms_required": False,
            "confidence_threshold": float(confidence_threshold),
            "iou_threshold": float(iou_threshold),
        }

    end2end_candidates: list[tuple[int, int]] = []
    for dimensions in preferred_keypoint_dimensions:
        expected = expected_yolo_output_attributes(
            output_format="end2end_detections", class_count=class_count,
            keypoint_count=normalized_keypoint_count, keypoint_dimensions=dimensions,
        )
        if (expected, dimensions) not in end2end_candidates:
            end2end_candidates.append((expected, dimensions))
    if end2end_output:
        for expected, dimensions in end2end_candidates:
            if len(dims) >= 1 and dims[-1] == expected:
                return _end2end_spec(dimensions)

    candidates: list[tuple[int, bool, int]] = []
    expected_text_items: list[str] = []
    seen_candidates: set[tuple[int, bool]] = set()
    for dimensions in preferred_keypoint_dimensions:
        base_channels = expected_yolo_output_attributes(
            output_format="raw_yolo", class_count=class_count,
            keypoint_count=normalized_keypoint_count, keypoint_dimensions=dimensions,
        )
        for expected, objectness in ((base_channels, False), (base_channels + 1, True)):
            key = (expected, objectness)
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            candidates.append((expected, objectness, dimensions))
            expected_text_items.append(f"{expected}{' + objectness' if objectness else ''} (kpt_dim={dimensions})")

    tensor_layout = ""
    has_objectness = False
    resolved_keypoint_dimensions = int(keypoint_dimensions or 0)
    for expected, objectness, dimensions in candidates:
        if len(dims) >= 2 and dims[-2] == expected:
            tensor_layout = "channels_first"
            has_objectness = objectness
            resolved_keypoint_dimensions = dimensions
            break
        if len(dims) >= 1 and dims[-1] == expected:
            tensor_layout = "anchors_first"
            has_objectness = objectness
            resolved_keypoint_dimensions = dimensions
            break
    if not tensor_layout:
        for expected, dimensions in end2end_candidates:
            if len(dims) >= 1 and dims[-1] == expected:
                return _end2end_spec(dimensions)
        raise MobileExportError(
            "Nie rozpoznano układu surowego wyjścia YOLO. "
            f"Kształt: {shape}, oczekiwane kanały: {', '.join(expected_text_items)}; "
            f"end-to-end: {', '.join(f'{expected} (kpt_dim={dimensions})' for expected, dimensions in end2end_candidates)}."
        )

    return {
        "decoder": _decoder_for_task(task),
        "output_format": "raw_yolo",
        "end2end_output": False,
        "class_count": int(class_count),
        "keypoint_count": normalized_keypoint_count,
        "keypoint_dimensions": int(resolved_keypoint_dimensions),
        "has_objectness": bool(has_objectness),
        "tensor_layout": tensor_layout,
        "box_format": "xywh",
        "normalized_coordinates": bool(normalized_coordinates),
        "nms_in_graph": False,
        "nms_required": True,
        "confidence_threshold": float(confidence_threshold),
        "iou_threshold": float(iou_threshold),
    }


def validate_checkpoint_metadata(checkpoint: Path, metadata: dict | None) -> None:
    """Reject a manifest assembled from different weights or epoch profiles."""
    metadata = metadata if isinstance(metadata, dict) else {}
    training = metadata.get("training") or {}
    metrics = metadata.get("metrics") or {}
    candidate = metadata.get("candidate") or {}
    candidate_training = candidate.get("training_provenance") or {}
    output_snapshots = [(payload.get("output_checkpoint") or {}) for payload in (training, candidate_training)]
    expected_hashes = [candidate.get("checkpoint_sha256"), (metadata.get("source") or {}).get("checkpoint_sha256")]
    for payload in (training, candidate_training):
        output = payload.get("output_checkpoint") or {}
        expected_hashes.extend([payload.get("best_checkpoint_sha256"), output.get("best_checkpoint_sha256"),
                                (output.get("best") or {}).get("sha256")])
    expected_hashes = {str(value).lower() for value in expected_hashes if value}
    if expected_hashes and expected_hashes != {sha256_file(checkpoint).lower()}:
        raise MobileExportError("Metadane treningu nie odpowiadają wybranemu checkpointowi (SHA-256). Odśwież listę modeli.")
    epochs = set()
    for payload in (training, metrics, candidate, candidate_training, *output_snapshots):
        value = payload.get("best_epoch")
        if value is None or value == "":
            continue
        try:
            epoch = int(value)
            if epoch <= 0 or float(value) != epoch:
                raise ValueError()
        except (TypeError, ValueError, OverflowError):
            raise MobileExportError("Nieprawidłowa najlepsza epoka w metadanych modelu.")
        epochs.add(epoch)
    if len(epochs) > 1:
        raise MobileExportError("Profil kandydata i manifest wskazują różne najlepsze epoki. Odśwież listę modeli.")


class MobileModelExporter:
    """Export trained YOLO weights into the Android ALPR model-package contract."""

    def preflight(self, request: MobileExportRequest) -> list[str]:
        problems: list[str] = []
        try:
            checkpoint = Path(request.checkpoint)
            if not checkpoint.exists() or not checkpoint.is_file():
                problems.append(f"Brak checkpointu: {checkpoint}")
            else:
                validate_checkpoint_metadata(checkpoint, request.metadata)
        except Exception as exc:
            problems.append(f"Nieprawidłowa ścieżka checkpointu: {exc}")

        try:
            destination = Path(request.destination)
            if destination.suffix.lower() != ".alprmodel":
                problems.append("Plik docelowy musi mieć rozszerzenie .alprmodel.")
            parent = destination.parent
            if not parent.exists() and not parent.parent.exists():
                problems.append(f"Brak katalogu nadrzędnego dla eksportu: {parent.parent}")
        except Exception as exc:
            problems.append(f"Nieprawidłowa ścieżka docelowa: {exc}")

        try:
            _normalize_role(request.role)
        except Exception as exc:
            problems.append(str(exc))

        formats = _normalized_export_formats(request.formats)
        if not formats:
            problems.append("Wybierz co najmniej jeden format: LiteRT/TFLite, ONNX albo NCNN.")
        for item in formats:
            if item not in {"litert", "onnx", "ncnn"}:
                problems.append(f"Nieobsługiwany format eksportu: {item}")

        quantizations = _normalized_export_quantizations(request.quantizations)
        for precision in quantizations:
            if precision not in {"fp32", "int8"}:
                problems.append(f"Nieobsługiwana precyzja: {precision}")

        raw_format_quantizations = request.format_quantizations if isinstance(request.format_quantizations, dict) else {}
        supported_by_format = {
            "litert": {"fp32", "int8"},
            "onnx": {"fp32", "int8"},
            "ncnn": {"fp32"},
        }
        for raw_format, raw_precisions in raw_format_quantizations.items():
            fmt = str(raw_format or "").strip().lower()
            if fmt == "tflite":
                fmt = "litert"
            if fmt not in supported_by_format:
                problems.append(f"Nieobsługiwany format eksportu w mapie precyzji: {raw_format}")
                continue
            requested_precisions = _normalized_export_quantizations(raw_precisions, default=())
            for precision in requested_precisions:
                if precision not in supported_by_format[fmt]:
                    if fmt == "ncnn" and precision == "int8":
                        problems.append("NCNN nie obsługuje eksportu INT8 w ścieżce Ultralytics; użyj NCNN FP32 albo ONNX/LiteRT INT8.")
                    else:
                        problems.append(f"Format {fmt.upper()} nie obsługuje precyzji {precision.upper()}.")

        try:
            _normalize_image_size(request.image_size)
        except Exception as exc:
            problems.append(str(exc))

        for row in mobile_export_required_specs(formats):
            ok, detail = mobile_export_requirement_status(str(row.get("spec") or ""), str(row.get("module") or ""))
            if not ok:
                problems.append(f"Brak zależności {row.get('scope') or 'Eksport'}: {detail}")

        if all(
            mobile_export_requirement_status(spec, module)[0]
            for module, spec in MOBILE_EXPORT_CORE_REQUIREMENTS
        ):
            runtime_problem = check_mobile_yolo_export_runtime()
            if runtime_problem:
                problems.append(runtime_problem)

        if "litert" in formats:
            tflite_problem = check_mobile_tflite_inspection_runtime()
            if tflite_problem:
                problems.append(tflite_problem)

        calibration_targets = _request_int8_calibration_targets(request)
        if "onnx:int8" in calibration_targets:
            onnx_int8_problem = check_mobile_onnx_int8_runtime()
            if onnx_int8_problem:
                problems.append(onnx_int8_problem)
        if calibration_targets and request.calibration_data is None:
            target_labels = {"litert:int8": "LiteRT/TFLite INT8", "onnx:int8": "ONNX INT8"}
            labels = ", ".join(target_labels.get(target, target) for target in calibration_targets)
            problems.append(f"{labels} wymaga datasetu kalibracyjnego data.yaml.")

        unique: list[str] = []
        seen: set[str] = set()
        for problem in problems:
            key = str(problem)
            if key in seen:
                continue
            seen.add(key)
            unique.append(key)
        return unique

    def export(
        self,
        request: MobileExportRequest,
        *,
        progress: Callable[[float | None, str], None] | None = None,
    ) -> Path:
        problems = self.preflight(request)
        if problems:
            raise MobileExportError("Preflight eksportu mobilnego nie przeszedł:\n" + "\n".join(f"- {item}" for item in problems))

        checkpoint = Path(request.checkpoint).resolve()
        destination = Path(request.destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        role = _normalize_role(request.role)
        task = _role_task(role)
        width, height = _normalize_image_size(request.image_size)
        model_id = _safe_id(request.model_id or checkpoint.stem, fallback=f"{role}-model")
        if not SAFE_MODEL_ID_RE.match(model_id):
            raise MobileExportError(f"Nieprawidłowy model_id: {model_id}")

        def notify(value: float | None, message: str) -> None:
            if progress is not None:
                try:
                    progress(value, message)
                except Exception:
                    pass

        notify(4.0, "Wczytuję checkpoint YOLO i sprawdzam rolę modelu.")
        with tempfile.TemporaryDirectory(prefix="alpr_mobile_export_", dir=str(destination.parent)) as temp_name:
            temp_root = Path(temp_name)
            work_dir = temp_root / "work"
            package_root = temp_root / "package"
            work_dir.mkdir(parents=True, exist_ok=True)
            package_root.mkdir(parents=True, exist_ok=True)
            temp_checkpoint = work_dir / "best.pt"
            shutil.copy2(checkpoint, temp_checkpoint)

            model = self._load_yolo_model(temp_checkpoint)
            model_info = self._inspect_yolo_checkpoint(model, role=role, task=task)
            labels = list(model_info["labels"])
            class_count = int(model_info["class_count"])
            keypoint_count = int(model_info["keypoint_count"])
            keypoint_dimensions = int(model_info.get("keypoint_dimensions", 0) or 0)
            end2end_output = bool(model_info.get("end2end_output", False))

            variants: list[ExportedVariant] = []
            variant_specs = self._requested_variant_specs(request)
            total_specs = max(1, len(variant_specs))
            for index, (runtime, precision) in enumerate(variant_specs, start=1):
                slot_start = 8.0 + (index - 1) * (62.0 / total_specs)
                slot_span = 62.0 / total_specs
                notify(slot_start, f"Eksportuję wariant {runtime.upper()} {precision.upper()}.")

                def variant_progress(fraction: float, message: str, *, start=slot_start, span=slot_span) -> None:
                    try:
                        value = max(0.0, min(1.0, float(fraction)))
                    except Exception:
                        value = 0.0
                    notify(start + (value * span), message)

                variant = self._export_variant(
                    model,
                    runtime=runtime,
                    precision=precision,
                    request=request,
                    package_root=package_root,
                    task=task,
                    class_count=class_count,
                    keypoint_count=keypoint_count,
                    keypoint_dimensions=keypoint_dimensions,
                    end2end_output=end2end_output,
                    progress=variant_progress,
                )
                variants.append(variant)

            if not variants:
                raise MobileExportError("Eksport nie utworzył żadnego wariantu modelu.")

            notify(76.0, "Buduję manifest i liczę sumy SHA-256.")
            manifest = self._build_manifest(
                request=request,
                model_id=model_id,
                role=role,
                task=task,
                width=width,
                height=height,
                labels=labels,
                variants=variants,
                checkpoint=checkpoint,
                model_info=model_info,
            )
            self._validate_manifest_basic(manifest)
            manifest_path = package_root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

            notify(86.0, "Pakuję archiwum .alprmodel.")
            temp_package = temp_root / destination.name
            self._write_zip(package_root, temp_package)
            notify(94.0, "Sprawdzam paczkę po ponownym otwarciu ZIP.")
            self.validate_package(temp_package)

            os.replace(str(temp_package), str(destination))
            notify(100.0, f"Model mobilny gotowy: {destination.name}")
            return destination

    def inspect_variant(self, variant: ExportedVariant, *, role: MobileRole) -> ExportedVariant:
        task = _role_task(role)
        if variant.runtime == "ncnn":
            output_spec = _resolve_ncnn_output_spec(
                task=task,
                class_count=variant.output_spec["class_count"],
                keypoint_count=variant.output_spec["keypoint_count"],
                keypoint_dimensions=variant.output_spec["keypoint_dimensions"],
                confidence_threshold=variant.output_spec["confidence_threshold"],
                iou_threshold=variant.output_spec["iou_threshold"],
            )
            return replace(variant, output_spec=output_spec)
        if variant.runtime == "onnx":
            input_spec, output_spec = self._inspect_onnx_variant(
                variant.files[0],
                task=task,
                class_count=int(variant.output_spec.get("class_count", 0) or 0),
                keypoint_count=int(variant.output_spec.get("keypoint_count", 0) or 0),
                keypoint_dimensions=int(variant.output_spec.get("keypoint_dimensions", 0) or 0),
                end2end_output=bool(variant.output_spec.get("end2end_output", False))
                or "end2end" in str(variant.output_spec.get("decoder", "")),
                confidence_threshold=float(variant.output_spec.get("confidence_threshold", 0.25)),
                iou_threshold=float(variant.output_spec.get("iou_threshold", 0.45)),
            )
            return ExportedVariant(
                id=variant.id,
                runtime=variant.runtime,
                precision=variant.precision,
                files=variant.files,
                relative_files=variant.relative_files,
                input_spec=input_spec,
                output_spec=output_spec,
            )
        if variant.runtime == "tflite":
            input_spec, output_spec = self._inspect_tflite_variant(
                variant.files[0],
                task=task,
                class_count=int(variant.output_spec.get("class_count", 0) or 0),
                keypoint_count=int(variant.output_spec.get("keypoint_count", 0) or 0),
                keypoint_dimensions=int(variant.output_spec.get("keypoint_dimensions", 0) or 0),
                end2end_output=bool(variant.output_spec.get("end2end_output", False))
                or "end2end" in str(variant.output_spec.get("decoder", "")),
                confidence_threshold=float(variant.output_spec.get("confidence_threshold", 0.25)),
                iou_threshold=float(variant.output_spec.get("iou_threshold", 0.45)),
            )
            return ExportedVariant(
                id=variant.id,
                runtime=variant.runtime,
                precision=variant.precision,
                files=variant.files,
                relative_files=variant.relative_files,
                input_spec=input_spec,
                output_spec=output_spec,
            )
        return variant

    def validate_package(self, package_path: Path) -> None:
        package = Path(package_path)
        if not package.exists() or not package.is_file():
            raise MobileExportError(f"Brak pakietu: {package}")
        seen: set[str] = set()
        total_uncompressed = 0
        with zipfile.ZipFile(package, "r") as archive:
            entries = archive.infolist()
            if len(entries) > MAX_PACKAGE_ENTRIES:
                raise MobileExportError("Pakiet zawiera zbyt wiele plików.")
            names = [entry.filename.replace("\\", "/") for entry in entries]
            if "manifest.json" not in names:
                raise MobileExportError("Pakiet nie zawiera manifest.json w katalogu głównym.")
            for entry in entries:
                name = entry.filename.replace("\\", "/")
                if name in seen:
                    raise MobileExportError(f"Powtórzony wpis ZIP: {name}")
                seen.add(name)
                posix = PurePosixPath(name)
                if posix.is_absolute() or ".." in posix.parts:
                    raise MobileExportError(f"Niedozwolona ścieżka w ZIP: {name}")
                total_uncompressed += int(entry.file_size or 0)
                if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                    raise MobileExportError("Rozpakowany pakiet przekracza limit 512 MiB.")

            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            self._validate_manifest_basic(manifest)
            name_set = set(names)
            for variant in manifest.get("variants", []):
                files = list(variant.get("files") or ([variant.get("file")] if variant.get("file") else []))
                runtime = str(variant.get("runtime") or "").lower()
                has_param = False
                has_bin = False
                for relative in files:
                    relative = str(relative or "").replace("\\", "/")
                    if relative not in name_set:
                        raise MobileExportError(f"Brak pliku wariantu {variant.get('id')}: {relative}")
                    checksum = str((variant.get("sha256") or {}).get(relative) or "").lower()
                    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
                        raise MobileExportError(f"Brak prawidłowej sumy SHA-256 dla: {relative}")
                    actual = hashlib.sha256(archive.read(relative)).hexdigest()
                    if actual != checksum:
                        raise MobileExportError(f"Niezgodna suma SHA-256 pliku: {relative}")
                    lower = relative.lower()
                    has_param = has_param or lower.endswith(".param")
                    has_bin = has_bin or lower.endswith(".bin")
                primary = str(files[0] if files else "").lower()
                if runtime == "tflite" and not primary.endswith(".tflite"):
                    raise MobileExportError("Wariant TFLite musi wskazywać plik .tflite.")
                if runtime == "onnx" and not primary.endswith(".onnx"):
                    raise MobileExportError("Wariant ONNX musi wskazywać plik .onnx.")
                if runtime == "ncnn" and not (has_param and has_bin):
                    raise MobileExportError("Wariant NCNN wymaga plików .param i .bin.")

    def _load_yolo_model(self, checkpoint: Path):
        runtime_problem = check_mobile_yolo_export_runtime()
        if runtime_problem:
            raise MobileExportError(runtime_problem)
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise MobileExportError(
                "Nie można uruchomić Ultralytics YOLO mimo obecności pakietu. "
                f"Najpewniej środowisko torch/torchvision/ultralytics jest niespójne: {exc}"
            ) from exc
        try:
            return YOLO(str(checkpoint))
        except Exception as exc:
            raise MobileExportError(f"Nie udało się wczytać checkpointu YOLO: {exc}") from exc

    def _inspect_yolo_checkpoint(self, model, *, role: str, task: str) -> dict:
        model_task = str(getattr(model, "task", "") or "").strip().lower()
        if model_task and model_task != task:
            raise MobileExportError(f"Checkpoint ma task={model_task}, a rola {role} wymaga task={task}.")

        labels = self._labels_from_model(model)
        if not labels:
            raise MobileExportError("Model nie zawiera etykiet klas.")

        keypoint_count = 0
        keypoint_dimensions = 0
        if task == "pose":
            keypoint_count, keypoint_dimensions = self._keypoint_shape_from_model(model)
            if keypoint_count < 4:
                raise MobileExportError("Model tablic musi zwracać co najmniej 4 keypointy narożników.")
        elif keypoint_count != 0:
            raise MobileExportError("Model detect nie powinien mieć keypointów.")

        parameter_count = self._parameter_count_from_model(model)
        end2end_output = self._end2end_output_from_model(model)
        return {
            "task": task,
            "labels": labels,
            "class_count": len(labels),
            "keypoint_count": int(keypoint_count),
            "keypoint_dimensions": int(keypoint_dimensions),
            "end2end_output": bool(end2end_output),
            "model_task": model_task or task,
            "parameter_count": parameter_count,
            "parameters_millions": round(parameter_count / 1_000_000.0, 3) if parameter_count > 0 else 0.0,
        }

    def _parameter_count_from_model(self, model) -> int:
        model_obj = getattr(model, "model", None)
        for source in (model_obj, model):
            if source is None:
                continue
            parameters = getattr(source, "parameters", None)
            if not callable(parameters):
                continue
            try:
                return int(sum(int(param.numel()) for param in parameters()))
            except Exception:
                continue
        return 0

    def _labels_from_model(self, model) -> list[str]:
        raw_names = getattr(model, "names", None)
        if raw_names is None and getattr(model, "model", None) is not None:
            raw_names = getattr(model.model, "names", None)
        if isinstance(raw_names, dict):
            parsed: dict[int, str] = {}
            for key, value in raw_names.items():
                try:
                    index = int(key)
                except Exception:
                    raise MobileExportError(f"Nieprawidłowy indeks klasy w model.names: {key}")
                parsed[index] = str(value)
            indices = sorted(parsed)
            expected = list(range(len(indices)))
            if indices != expected:
                raise MobileExportError(f"Klasy modelu nie tworzą zakresu 0..N-1: {indices}")
            return [parsed[index] for index in indices]
        if isinstance(raw_names, (list, tuple)):
            return [str(item) for item in raw_names]
        raise MobileExportError("Nie udało się odczytać model.names.")

    def _keypoint_count_from_model(self, model) -> int:
        return self._keypoint_shape_from_model(model)[0]

    def _keypoint_shape_from_model(self, model) -> tuple[int, int]:
        candidates = []
        model_obj = getattr(model, "model", None)
        for source in (model, model_obj):
            if source is None:
                continue
            candidates.append(getattr(source, "kpt_shape", None))
            yaml_data = getattr(source, "yaml", None)
            if isinstance(yaml_data, dict):
                candidates.append(yaml_data.get("kpt_shape"))
        for item in candidates:
            if isinstance(item, (list, tuple)) and item:
                try:
                    count = int(item[0])
                    dimensions = int(item[1]) if len(item) > 1 else 0
                    return count, dimensions
                except Exception:
                    continue
            try:
                number = int(item)
                if number > 0:
                    return number, 0
            except Exception:
                continue
        return 0, 0

    def _end2end_output_from_model(self, model) -> bool:
        model_obj = getattr(model, "model", None)
        sources = [model, model_obj]
        nested = getattr(model_obj, "model", None)
        if isinstance(nested, (list, tuple)) and nested:
            sources.append(nested[-1])
        elif nested is not None:
            sources.append(nested)
        for source in sources:
            if source is None:
                continue
            try:
                if bool(getattr(source, "end2end", False)):
                    return True
            except Exception:
                pass
            try:
                yaml_data = getattr(source, "yaml", None)
                if isinstance(yaml_data, dict) and bool(yaml_data.get("end2end", False)):
                    return True
            except Exception:
                pass
        return False

    def _requested_variant_specs(self, request: MobileExportRequest) -> list[tuple[str, str]]:
        format_quantizations = _request_format_quantization_map(request)
        specs: list[tuple[str, str]] = []
        if "litert" in format_quantizations:
            if "fp32" in format_quantizations["litert"]:
                specs.append(("tflite", "fp32"))
            if "int8" in format_quantizations["litert"]:
                specs.append(("tflite", "int8"))
        if "onnx" in format_quantizations:
            if "fp32" in format_quantizations["onnx"]:
                specs.append(("onnx", "fp32"))
            if "int8" in format_quantizations["onnx"]:
                specs.append(("onnx", "int8"))
        if "ncnn" in format_quantizations:
            specs.append(("ncnn", "fp32"))
        return specs

    def _export_variant(
        self,
        model,
        *,
        runtime: str,
        precision: str,
        request: MobileExportRequest,
        package_root: Path,
        task: str,
        class_count: int,
        keypoint_count: int,
        keypoint_dimensions: int = 0,
        end2end_output: bool = False,
        progress: Callable[[float, str], None] | None = None,
    ) -> ExportedVariant:
        def notify(value: float, message: str) -> None:
            if progress is not None:
                try:
                    progress(value, message)
                except Exception:
                    pass

        notify(0.08, f"Buduję plik {runtime.upper()} {precision.upper()}.")
        export_result = self._run_ultralytics_export(model, runtime=runtime, precision=precision, request=request)
        if runtime == "onnx" and precision == "int8":
            notify(0.42, "ONNX FP32 gotowy. Przygotowuję kalibrację INT8.")
            fp32_files = self._locate_export_outputs(export_result, runtime="onnx", precision="fp32")
            source_onnx = fp32_files[0]
            export_result = self._quantize_onnx_int8(
                source_onnx,
                request,
                output_path=source_onnx.with_name(f"{source_onnx.stem}_int8{source_onnx.suffix}"),
                progress=lambda value, message: notify(0.42 + (max(0.0, min(1.0, float(value))) * 0.38), message),
            )
        notify(0.82, "Kopiuję pliki wariantu do paczki.")
        exported_files = self._locate_export_outputs(export_result, runtime=runtime, precision=precision)

        variant_dir_name = {
            ("tflite", "fp32"): "tflite",
            ("tflite", "int8"): "tflite_int8",
            ("onnx", "fp32"): "onnx",
            ("onnx", "int8"): "onnx_int8",
            ("ncnn", "fp32"): "ncnn",
        }.get((runtime, precision), f"{runtime}_{precision}")
        variant_dir = package_root / "variants" / variant_dir_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        copied: list[Path] = []
        rels: list[str] = []
        for source in exported_files:
            if runtime == "tflite":
                destination = variant_dir / "model.tflite"
            elif runtime == "onnx":
                destination = variant_dir / "model.onnx"
            else:
                destination = variant_dir / source.name
            if destination.exists():
                destination.unlink()
            shutil.copy2(source, destination)
            copied.append(destination)
            rels.append(destination.relative_to(package_root).as_posix())

        default_input = {
            "width": _normalize_image_size(request.image_size)[0],
            "height": _normalize_image_size(request.image_size)[1],
            "channels": 3,
            "layout": "NCHW" if runtime in {"onnx", "ncnn"} else "NHWC",
            "color": "RGB",
            "data_type": "FLOAT32" if precision == "fp32" else "INT8",
            "scale": 0.0039215686,
            "offset": 0.0,
        }
        default_output = {
            "decoder": _decoder_for_task(task, end2end_output=end2end_output),
            "output_format": "end2end_detections" if end2end_output else "raw_yolo",
            "class_count": int(class_count),
            "keypoint_count": int(keypoint_count),
            "keypoint_dimensions": int(keypoint_dimensions or 0),
            "end2end_output": bool(end2end_output),
            "has_objectness": False,
            "tensor_layout": "channels_first",
            "box_format": "xyxy" if end2end_output else "xywh",
            "normalized_coordinates": runtime == "tflite",
            "nms_in_graph": False,
            "nms_required": not bool(end2end_output),
            "confidence_threshold": float(request.confidence_threshold),
            "iou_threshold": float(request.iou_threshold),
        }

        variant = ExportedVariant(
            id=f"{runtime}-{precision}",
            runtime=runtime,
            precision=precision,
            files=tuple(copied),
            relative_files=tuple(rels),
            input_spec=default_input,
            output_spec=default_output,
        )
        notify(0.94, "Ustalam kontrakt RAW YOLO dla NCNN." if runtime == "ncnn"
               else "Sprawdzam wejście i wyjście wyeksportowanego modelu.")
        variant = self.inspect_variant(variant, role=request.role)
        notify(1.0, f"Wariant {runtime.upper()} {precision.upper()} gotowy.")
        return variant

    def _run_ultralytics_export(self, model, *, runtime: str, precision: str, request: MobileExportRequest):
        imgsz = _ultralytics_imgsz(request.image_size)
        common = {
            "imgsz": imgsz,
            "batch": 1,
            "nms": False,
            "device": "cpu",
        }
        with _controlled_ultralytics_export_runtime():
            if runtime == "onnx":
                args = dict(common)
                args.update({"dynamic": False, "simplify": True})
                return model.export(format="onnx", **args)
            if runtime == "ncnn":
                return model.export(format="ncnn", **common)
            if runtime == "tflite":
                args = dict(common)
                if precision == "int8":
                    calibration_data = str(Path(request.calibration_data).resolve()) if request.calibration_data else ""
                    if not calibration_data:
                        raise MobileExportError("Eksport INT8 wymaga datasetu kalibracyjnego data.yaml.")
                    args.update({"data": calibration_data, "fraction": 1.0})
                    try:
                        return model.export(format="tflite", int8=True, **args)
                    except TypeError:
                        return model.export(format="tflite", quantize=8, **args)
                return model.export(format="tflite", **args)
        raise MobileExportError(f"Nieobsługiwany runtime eksportu: {runtime}")

    def _quantize_onnx_int8(
        self,
        onnx_path: Path,
        request: MobileExportRequest,
        *,
        output_path: Path,
        progress: Callable[[float, str], None] | None = None,
    ) -> Path:
        def notify(value: float, message: str) -> None:
            if progress is not None:
                try:
                    progress(value, message)
                except Exception:
                    pass

        if request.calibration_data is None:
            raise MobileExportError("Eksport ONNX INT8 wymaga datasetu kalibracyjnego data.yaml.")
        data_yaml = Path(request.calibration_data).resolve()
        if not data_yaml.exists() or not data_yaml.is_file():
            raise MobileExportError(f"Dataset kalibracyjny ONNX INT8 nie istnieje: {data_yaml}")

        notify(0.05, "Czytam data.yaml i wybieram obrazy kalibracyjne ONNX INT8.")
        images = _collect_calibration_images(data_yaml)
        if not images:
            raise MobileExportError(f"Dataset kalibracyjny ONNX INT8 nie zawiera obrazów: {data_yaml}")
        notify(0.18, f"Kalibracja ONNX INT8 użyje {len(images)} obrazów.")

        try:
            import onnx  # type: ignore
            from onnxruntime.quantization import QuantFormat, QuantType, quantize_static  # type: ignore
        except Exception as exc:
            raise MobileExportError(f"Brak bibliotek do kwantyzacji ONNX INT8: {exc}") from exc

        input_name, image_size = _onnx_input_name_and_size(onnx_path, request.image_size)
        reader = _OnnxImageCalibrationReader(input_name=input_name, image_paths=images, image_size=image_size)
        notify(0.28, f"Przygotowałem wejście kalibracji: {input_name}, rozmiar {image_size[0]}x{image_size[1]}.")

        excluded_nodes: list[str] = []
        try:
            graph = onnx.load(str(onnx_path)).graph
            excluded_nodes = [
                str(node.name)
                for node in graph.node
                if str(getattr(node, "name", "") or "").strip() and node.op_type not in {"Conv", "Gemm", "MatMul"}
            ]
            del graph
        except Exception:
            excluded_nodes = []
        if excluded_nodes:
            notify(0.38, f"Kwantyzuję warstwy wagowe, pomijam {len(excluded_nodes)} węzłów głowicy i pomocniczych.")
        else:
            notify(0.38, "Kwantyzuję ONNX INT8 przez ONNX Runtime.")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            notify(0.48, "Trwa statyczna kwantyzacja ONNX INT8. To może chwilę potrwać.")
            quantize_static(
                str(onnx_path),
                str(output_path),
                reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=False,
                nodes_to_exclude=excluded_nodes or None,
            )
            notify(0.92, "Kwantyzacja ONNX INT8 zakończona, sprawdzam plik wynikowy.")
        except Exception as exc:
            raise MobileExportError(f"Nie udało się wykonać kwantyzacji ONNX INT8: {exc}") from exc

        if not output_path.exists() or not output_path.is_file():
            raise MobileExportError("Kwantyzacja ONNX INT8 nie utworzyła pliku wynikowego.")
        return output_path

    def _locate_export_outputs(self, export_result, *, runtime: str, precision: str) -> tuple[Path, ...]:
        candidates: list[Path] = []
        raw_items = export_result if isinstance(export_result, (list, tuple, set)) else [export_result]
        for item in raw_items:
            if item is None:
                continue
            try:
                candidates.append(Path(str(item)))
            except Exception:
                continue

        files: list[Path] = []
        for candidate in candidates:
            if candidate.is_file():
                files.append(candidate)
            elif candidate.is_dir():
                if runtime == "tflite":
                    files.extend(candidate.rglob("*.tflite"))
                elif runtime == "onnx":
                    files.extend(candidate.rglob("*.onnx"))
                elif runtime == "ncnn":
                    files.extend(candidate.rglob("*.param"))
                    files.extend(candidate.rglob("*.bin"))

        if runtime == "tflite":
            filtered = []
            for file_path in files:
                lower = file_path.name.lower()
                is_int8 = any(token in lower for token in ("int8", "integer", "quant"))
                if precision == "int8" and is_int8:
                    filtered.append(file_path)
                elif precision == "fp32" and not is_int8:
                    filtered.append(file_path)
            files = filtered or files
            if not files:
                raise MobileExportError("Eksport LiteRT/TFLite nie zwrócił pliku .tflite.")
            return (sorted(files, key=lambda item: len(str(item)))[0],)

        if runtime == "onnx":
            files = [item for item in files if item.suffix.lower() == ".onnx"]
            if precision == "int8":
                filtered = [item for item in files if any(token in item.name.lower() for token in ("int8", "integer", "quant"))]
                files = filtered or files
            elif precision == "fp32":
                filtered = [item for item in files if not any(token in item.name.lower() for token in ("int8", "integer", "quant"))]
                files = filtered or files
            if not files:
                raise MobileExportError("Eksport ONNX nie zwrócił pliku .onnx.")
            return (sorted(files, key=lambda item: len(str(item)))[0],)

        if runtime == "ncnn":
            param_files = sorted([item for item in files if item.suffix.lower() == ".param"])
            bin_files = sorted([item for item in files if item.suffix.lower() == ".bin"])
            if not param_files or not bin_files:
                raise MobileExportError("Eksport NCNN musi zwrócić pliki .param i .bin.")
            return (param_files[0], bin_files[0])

        raise MobileExportError(f"Nieobsługiwany runtime eksportu: {runtime}")

    def _inspect_onnx_variant(
        self,
        path: Path,
        *,
        task: str,
        class_count: int,
        keypoint_count: int,
        keypoint_dimensions: int = 0,
        end2end_output: bool = False,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> tuple[dict, dict]:
        try:
            import onnx  # type: ignore
        except Exception as exc:
            raise MobileExportError(f"Brak pakietu onnx do inspekcji wariantu: {exc}") from exc
        model = onnx.load(str(path))
        if not model.graph.input or not model.graph.output:
            raise MobileExportError("Model ONNX nie ma wejścia albo wyjścia.")
        input_value = model.graph.input[0]
        output_value = model.graph.output[0]
        input_type = input_value.type.tensor_type
        output_type = output_value.type.tensor_type
        input_shape = [_shape_dim_value(dim) for dim in input_type.shape.dim]
        output_shape = [_shape_dim_value(dim) for dim in output_type.shape.dim]
        input_spec = _build_input_spec_from_shape(input_shape, _dtype_name_from_onnx(input_type.elem_type))
        if _dtype_name_from_onnx(input_type.elem_type) != "FLOAT32":
            raise MobileExportError("Androidowy backend ONNX wymaga wejścia FLOAT32.")
        output_spec = _infer_output_spec(
            output_shape,
            task=task,
            class_count=class_count,
            keypoint_count=keypoint_count,
            keypoint_dimensions=keypoint_dimensions,
            end2end_output=end2end_output,
            normalized_coordinates=False,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        return input_spec, output_spec

    def _inspect_tflite_variant(
        self,
        path: Path,
        *,
        task: str,
        class_count: int,
        keypoint_count: int,
        keypoint_dimensions: int = 0,
        end2end_output: bool = False,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> tuple[dict, dict]:
        interpreter_cls = _tflite_interpreter_class()
        interpreter = interpreter_cls(model_path=str(path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        if not input_details or not output_details:
            raise MobileExportError("Model TFLite nie ma wejścia albo wyjścia.")
        input_detail = input_details[0]
        output_detail = output_details[0]
        input_shape = _tflite_shape_list(input_detail)
        output_shape = _tflite_shape_list(output_detail)
        input_spec = _build_input_spec_from_shape(input_shape, _dtype_name_from_tflite(input_detail.get("dtype")))
        quant = input_detail.get("quantization")
        if quant and isinstance(quant, tuple) and len(quant) == 2:
            scale, zero_point = quant
            if scale:
                input_spec["quantization"] = {"scale": float(scale), "zero_point": int(zero_point)}
        output_spec = _infer_output_spec(
            output_shape,
            task=task,
            class_count=class_count,
            keypoint_count=keypoint_count,
            keypoint_dimensions=keypoint_dimensions,
            end2end_output=end2end_output,
            normalized_coordinates=True,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        return input_spec, output_spec

    def _build_manifest(
        self,
        *,
        request: MobileExportRequest,
        model_id: str,
        role: str,
        task: str,
        width: int,
        height: int,
        labels: list[str],
        variants: list[ExportedVariant],
        checkpoint: Path,
        model_info: dict,
    ) -> dict:
        default_input = dict(variants[0].input_spec) if variants else {
            "width": width,
            "height": height,
            "channels": 3,
            "layout": "NHWC",
            "color": "RGB",
            "data_type": "FLOAT32",
            "scale": 0.0039215686,
            "offset": 0.0,
        }
        default_output = dict(variants[0].output_spec) if variants else {
            "decoder": _decoder_for_task(task, end2end_output=bool(model_info.get("end2end_output", False))),
            "output_format": "end2end_detections" if bool(model_info.get("end2end_output", False)) else "raw_yolo",
            "class_count": len(labels),
            "keypoint_count": int(model_info.get("keypoint_count", 0) or 0),
            "keypoint_dimensions": int(model_info.get("keypoint_dimensions", 0) or 0),
            "end2end_output": bool(model_info.get("end2end_output", False)),
            "has_objectness": False,
            "tensor_layout": "channels_first",
            "box_format": "xyxy" if bool(model_info.get("end2end_output", False)) else "xywh",
            "normalized_coordinates": False,
            "nms_in_graph": False,
            "nms_required": not bool(model_info.get("end2end_output", False)),
            "confidence_threshold": float(request.confidence_threshold),
            "iou_threshold": float(request.iou_threshold),
        }

        variant_items: list[dict] = []
        for variant in variants:
            sha_map = {relative: sha256_file(file_path) for relative, file_path in zip(variant.relative_files, variant.files)}
            item = {
                "id": variant.id,
                "runtime": variant.runtime,
                "precision": variant.precision,
                "sha256": sha_map,
            }
            if len(variant.relative_files) == 1:
                item["file"] = variant.relative_files[0]
            else:
                item["files"] = list(variant.relative_files)
            if variant.input_spec != default_input:
                item["input"] = dict(variant.input_spec)
            if variant.output_spec != default_output:
                item["output"] = dict(variant.output_spec)
            variant_items.append(item)

        metadata = _json_safe_value(dict(request.metadata or {}))
        if not isinstance(metadata, dict):
            metadata = {}
        validate_checkpoint_metadata(checkpoint, metadata)
        source_meta = dict(metadata.get("source") or {})
        source_meta.update(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "exported_at": _utc_now_iso(),
            }
        )
        parameter_count = int(model_info.get("parameter_count", 0) or 0)
        if parameter_count > 0:
            source_meta.setdefault("parameter_count", parameter_count)
            source_meta.setdefault("parameters_millions", round(parameter_count / 1_000_000.0, 3))
        try:
            import ultralytics  # type: ignore

            source_meta.setdefault("ultralytics_version", str(getattr(ultralytics, "__version__", "")))
        except Exception:
            pass
        try:
            import torch  # type: ignore

            source_meta.setdefault("torch_version", str(getattr(torch, "__version__", "")))
        except Exception:
            pass

        manifest = {
            "schema": MOBILE_MODEL_SCHEMA,
            "model_id": model_id,
            "name": request.name or model_id,
            "version": str(request.version or "1"),
            "role": role,
            "task": task,
            "input": default_input,
            "output": default_output,
            "labels": labels,
            "variants": variant_items,
            "source": source_meta,
            "training": dict(metadata.get("training") or {}),
            "metrics": dict(metadata.get("metrics") or {}),
            "exporter": {
                "name": "auto_annotation_tool.mobile_model_exporter",
                "schema": MOBILE_MODEL_SCHEMA,
            },
            "export_reproducibility": _single_model_reproducibility_payload(
                request=request,
                model_id=model_id,
                role=role,
                task=task,
                labels=labels,
                checkpoint=checkpoint,
                model_info=model_info,
                metadata=metadata,
                source_meta=source_meta,
                variants=variant_items,
            ),
        }
        for key, value in metadata.items():
            if key not in {"source", "training", "metrics"}:
                manifest[key] = value
        if role == "vehicle":
            manifest["vehicle_detection"] = _build_vehicle_detection_contract(metadata, labels)
        return manifest

    def _validate_manifest_basic(self, manifest: dict) -> None:
        if not isinstance(manifest, dict):
            raise MobileExportError("Manifest nie jest obiektem JSON.")
        if manifest.get("schema") != MOBILE_MODEL_SCHEMA:
            raise MobileExportError(f"Nieobsługiwany schemat manifestu: {manifest.get('schema')}")
        model_id = str(manifest.get("model_id") or "").strip()
        if not SAFE_MODEL_ID_RE.match(model_id):
            raise MobileExportError(f"Nieprawidłowy model_id: {model_id}")
        role = _normalize_role(str(manifest.get("role") or ""))
        task = str(manifest.get("task") or "").strip().lower()
        if task not in {"detect", "pose"}:
            raise MobileExportError(f"Nieobsługiwany task: {task}")
        if role == "plate" and task != "pose":
            raise MobileExportError("Model tablic musi być modelem pose.")
        if role in {"vehicle", "character"} and task != "detect":
            raise MobileExportError("Model pojazdu i znaków musi być modelem detect.")

        labels = list(manifest.get("labels") or [])
        output = dict(manifest.get("output") or {})
        class_count = int(output.get("class_count", 0) or 0)
        if not labels or len(labels) != class_count:
            raise MobileExportError(f"labels ({len(labels)}) != class_count ({class_count}).")
        if bool(output.get("nms_in_graph")):
            raise MobileExportError("Manifest nie może deklarować NMS w grafie.")
        if role == "plate" and int(output.get("keypoint_count", 0) or 0) < 4:
            raise MobileExportError("Model tablic pose musi mieć co najmniej 4 keypointy.")

        variants = list(manifest.get("variants") or [])
        if not variants:
            raise MobileExportError("Manifest nie zawiera wariantów.")
        for variant in variants:
            runtime = str(variant.get("runtime") or "").strip().lower()
            precision = str(variant.get("precision") or "").strip().lower()
            if runtime not in {"tflite", "onnx", "ncnn"}:
                raise MobileExportError(f"Nieobsługiwany runtime wariantu: {runtime}")
            variant_output = variant.get("output", output)
            if not isinstance(variant_output, dict):
                raise MobileExportError(f"Wariant {runtime}: output nie jest obiektem JSON.")
            # Android treats output as a complete override, not a field merge.
            _validate_variant_output_contract(runtime, variant_output, task=task, class_count=len(labels))
            if precision not in {"fp32", "fp16", "int8", "uint8"}:
                raise MobileExportError(f"Nieobsługiwana precyzja wariantu: {precision}")
            files = list(variant.get("files") or ([variant.get("file")] if variant.get("file") else []))
            if not files:
                raise MobileExportError(f"Wariant {variant.get('id')} nie wskazuje pliku.")
            if not isinstance(variant.get("sha256"), dict) or not variant.get("sha256"):
                raise MobileExportError(f"Wariant {variant.get('id')} nie ma mapy SHA-256.")

        try:
            import jsonschema  # type: ignore

            jsonschema.validate(manifest, self.schema())
        except ImportError:
            pass
        except Exception as exc:
            raise MobileExportError(f"Manifest nie przechodzi JSON Schema: {exc}") from exc

    def _write_zip(self, package_root: Path, destination: Path) -> None:
        written: set[str] = set()
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package_root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(package_root).as_posix()
                posix = PurePosixPath(relative)
                if posix.is_absolute() or ".." in posix.parts:
                    raise MobileExportError(f"Niedozwolona ścieżka pakietu: {relative}")
                if relative in written:
                    raise MobileExportError(f"Powtórzony wpis pakietu: {relative}")
                written.add(relative)
                archive.write(path, relative)

    @staticmethod
    def schema() -> dict:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": True,
            "required": ["schema", "model_id", "role", "task", "input", "output", "labels", "variants"],
            "properties": {
                "schema": {"const": MOBILE_MODEL_SCHEMA},
                "model_id": {"type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$"},
                "role": {"enum": ["vehicle", "plate", "character"]},
                "task": {"enum": ["detect", "pose"]},
                "labels": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "variants": {"type": "array", "minItems": 1},
            },
        }


def _package_model_ref_payload(role: str, item: Mapping[str, Any]) -> dict[str, Any]:
    """Compact model reference intended for lightweight Android reports."""

    payload = dict(item or {})
    source = dict(payload.get("source") or {})
    training = dict(payload.get("training") or {})
    training_dataset = training.get("dataset") if isinstance(training.get("dataset"), Mapping) else {}
    model = dict(payload.get("model") or {})
    variants = [dict(variant) for variant in list(payload.get("variants") or []) if isinstance(variant, Mapping)]
    primary_variant = variants[0] if variants else {}
    package_file = str(payload.get("package_file") or "").strip()
    package_hashes = dict(payload.get("sha256") or {})
    package_sha = str(package_hashes.get(package_file) or "").strip()
    if not package_sha:
        for relative, digest in package_hashes.items():
            if str(relative).lower().endswith(".alprmodel"):
                package_sha = str(digest or "").strip()
                break
    variant_sha_values: list[str] = []
    variant_sha_map = primary_variant.get("sha256") if isinstance(primary_variant.get("sha256"), Mapping) else {}
    for digest in dict(variant_sha_map or {}).values():
        text = str(digest or "").strip()
        if text and text not in variant_sha_values:
            variant_sha_values.append(text)
    checkpoint_sha = str(source.get("checkpoint_sha256") or "").strip()
    return _json_safe_value(
        {
            "role": role,
            "model_id": str(payload.get("model_id") or "").strip(),
            "model_display_id": str(model.get("display_id") or "").strip(),
            "installed_model_fingerprint": str(
                source.get("installed_model_fingerprint")
                or package_sha
                or checkpoint_sha
                or payload.get("model_id")
                or ""
            ).strip(),
            "checkpoint_sha256": checkpoint_sha,
            "package_sha256": package_sha,
            "package_file": package_file,
            "manifest_file": str(payload.get("manifest_file") or "").strip(),
            "variant_id": str(primary_variant.get("id") or "").strip(),
            "variant_ids": [str(variant.get("id") or "").strip() for variant in variants if str(variant.get("id") or "").strip()],
            "variant_artifact_sha256": variant_sha_values,
            "runtime": str(primary_variant.get("runtime") or "").strip(),
            "precision": str(primary_variant.get("precision") or "").strip(),
            "task": str(payload.get("task") or "").strip(),
            "training": {
                "run_id": str(training.get("run_id") or "").strip(),
                "run_epochs_completed": training.get("run_epochs_completed"),
                "total_epochs": training.get("total_epochs"),
                "total_epochs_known": training.get("total_epochs_known"),
                "known_epochs_minimum": training.get("known_epochs_minimum"),
                "total_epochs_scope": str(training.get("total_epochs_scope") or "").strip(),
                "lineage_total_epochs": training.get("lineage_total_epochs"),
                "lineage_total_epochs_known": training.get("lineage_total_epochs_known"),
                "lineage_stage_count": training.get("lineage_stage_count"),
                "lineage_stage_count_known": training.get("lineage_stage_count_known"),
                "known_stage_count_minimum": training.get("known_stage_count_minimum"),
                "run_train_images": training.get("run_train_images"),
                "run_nominal_sample_presentations": training.get("run_nominal_sample_presentations"),
                "lineage_nominal_sample_presentations": training.get("lineage_nominal_sample_presentations"),
                "sample_presentations_known": training.get("sample_presentations_known"),
                "known_sample_presentations_minimum": training.get("known_sample_presentations_minimum"),
                "best_epoch_source": str(training.get("best_epoch_source") or "").strip(),
                "provenance_capture": str(training.get("provenance_capture") or "").strip(),
                "dataset_id": str(training_dataset.get("dataset_id") or training.get("dataset_id") or "").strip(),
                "provenance_status": str(training.get("provenance_status") or "").strip(),
            },
        }
    )


class MobileAlprPackageExporter:
    """Build a complete Android ALPR package from vehicle, plate and character models.

    The existing ``MobileModelExporter`` remains the authority for one logical
    model.  This class composes already-valid single-model packages into a
    deployment candidate: ``MT + MZ`` or ``MP + MT + MZ``.
    """

    def __init__(self, single_model_exporter: MobileModelExporter | None = None):
        self.single_model_exporter = single_model_exporter or MobileModelExporter()

    def preflight(self, request: MobileAlprPackageRequest) -> list[str]:
        problems: list[str] = []
        try:
            destination = Path(request.destination)
            if destination.suffix.lower() != ".alprmodel":
                problems.append("Pakiet ALPR musi miec rozszerzenie .alprmodel.")
            parent = destination.parent
            if not parent.exists() and not parent.parent.exists():
                problems.append(f"Brak katalogu nadrzednego dla eksportu: {parent.parent}")
        except Exception as exc:
            problems.append(f"Nieprawidlowa sciezka docelowa: {exc}")

        problems.extend(self._preflight_model_source(request, "vehicle", required=False))
        problems.extend(self._preflight_model_source(request, "plate", required=True))
        problems.extend(self._preflight_model_source(request, "character", required=True))

        unique: list[str] = []
        seen: set[str] = set()
        for problem in problems:
            key = str(problem)
            if key in seen:
                continue
            seen.add(key)
            unique.append(key)
        return unique

    def export(
        self,
        request: MobileAlprPackageRequest,
        *,
        progress: Callable[[float | None, str], None] | None = None,
    ) -> Path:
        problems = self.preflight(request)
        if problems:
            raise MobileExportError(
                "Preflight kompletnego pakietu ALPR nie przeszedl:\n"
                + "\n".join(f"- {item}" for item in problems)
            )

        destination = Path(request.destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        def notify(value: float | None, message: str) -> None:
            if progress is not None:
                try:
                    progress(value, message)
                except Exception:
                    pass

        has_child_requests = any(
            child is not None
            for child in (
                request.vehicle_request,
                request.plate_request,
                request.character_request,
            )
        )
        if not has_child_requests:
            return self.bundle_existing(request, progress=progress, _skip_preflight=True)

        package_label = "MP, MT i MZ" if (request.vehicle_package or request.vehicle_request) else "MT i MZ"
        notify(4.0, f"Przygotowuje modele {package_label} do kompletnego pakietu ALPR.")
        with tempfile.TemporaryDirectory(prefix="alpr_complete_export_", dir=str(destination.parent)) as temp_name:
            temp_root = Path(temp_name)
            single_root = temp_root / "single_models"
            single_root.mkdir(parents=True, exist_ok=True)
            vehicle_package = Path(request.vehicle_package) if request.vehicle_package else None
            plate_package = Path(request.plate_package) if request.plate_package else single_root / "plate.alprmodel"
            character_package = (
                Path(request.character_package)
                if request.character_package
                else single_root / "character.alprmodel"
            )
            if request.vehicle_request is not None and vehicle_package is None:
                vehicle_package = single_root / "vehicle.alprmodel"

            export_jobs: list[tuple[MobileExportRequest, str, Path, str]] = []
            if request.vehicle_request is not None:
                if vehicle_package is None:
                    raise MobileExportError("Brak miejsca docelowego dla modelu pojazdow MP.")
                export_jobs.append((request.vehicle_request, "vehicle", vehicle_package, "MP"))
            if request.plate_request is not None:
                export_jobs.append((request.plate_request, "plate", plate_package, "MT"))
            if request.character_request is not None:
                export_jobs.append((request.character_request, "character", character_package, "MZ"))

            span = 68.0 / max(1, len(export_jobs))
            for index, (child_request, role, package_path, label) in enumerate(export_jobs):
                self._export_single_child(
                    child_request,
                    role=role,
                    destination=package_path,
                    label=label,
                    offset=6.0 + index * span,
                    span=span,
                    notify=notify,
                )

            bundle_request = replace(
                request,
                vehicle_package=vehicle_package,
                plate_package=plate_package,
                character_package=character_package,
                vehicle_request=None,
                plate_request=None,
                character_request=None,
            )
            return self.bundle_existing(
                bundle_request,
                progress=lambda value, message: notify(
                    None if value is None else 76.0 + float(value) * 0.24,
                    message,
                ),
                _skip_preflight=True,
            )

    def bundle_existing(
        self,
        request: MobileAlprPackageRequest,
        *,
        progress: Callable[[float | None, str], None] | None = None,
        _skip_preflight: bool = False,
    ) -> Path:
        if not _skip_preflight:
            problems = self.preflight(request)
            if problems:
                raise MobileExportError(
                    "Preflight kompletnego pakietu ALPR nie przeszedl:\n"
                    + "\n".join(f"- {item}" for item in problems)
                )
        if not request.plate_package or not request.character_package:
            raise MobileExportError(COMPLETE_ALPR_MODELS_REQUIRED)

        destination = Path(request.destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        def notify(value: float | None, message: str) -> None:
            if progress is not None:
                try:
                    progress(value, message)
                except Exception:
                    pass

        package_label = "MP+MT+MZ" if request.vehicle_package else "MT+MZ"
        notify(6.0, f"Sprawdzam modele mobilne {package_label}.")
        with tempfile.TemporaryDirectory(prefix="alpr_package_bundle_", dir=str(destination.parent)) as temp_name:
            temp_root = Path(temp_name)
            package_root = temp_root / "package"
            package_root.mkdir(parents=True, exist_ok=True)

            vehicle_item = None
            if request.vehicle_package:
                vehicle_item = self._stage_single_model_package(
                    package_root,
                    role="vehicle",
                    source_package=Path(request.vehicle_package),
                )
                notify(24.0, "Pakiet MP zweryfikowany i dolaczony.")
            plate_item = self._stage_single_model_package(
                package_root,
                role="plate",
                source_package=Path(request.plate_package),
            )
            notify(44.0 if vehicle_item else 32.0, "Pakiet MT zweryfikowany i dolaczony.")
            character_item = self._stage_single_model_package(
                package_root,
                role="character",
                source_package=Path(request.character_package),
            )
            notify(66.0 if vehicle_item else 58.0, "Pakiet MZ zweryfikowany i dolaczony.")

            manifest = self._build_manifest(
                request=request,
                vehicle_item=vehicle_item,
                plate_item=plate_item,
                character_item=character_item,
            )
            self._validate_manifest_basic(manifest)
            manifest_path = package_root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

            notify(76.0, f"Pakuje kompletny zestaw ALPR {package_label}.")
            temp_package = temp_root / destination.name
            self._write_zip(package_root, temp_package)
            notify(90.0, "Sprawdzam kompletny pakiet po ponownym otwarciu.")
            self.validate_package(temp_package)

            os.replace(str(temp_package), str(destination))
            notify(100.0, f"Kompletny pakiet ALPR gotowy: {destination.name}")
            return destination

    def validate_package(self, package_path: Path) -> None:
        package = Path(package_path)
        if not package.exists() or not package.is_file():
            raise MobileExportError(f"Brak pakietu ALPR: {package}")

        seen: set[str] = set()
        total_uncompressed = 0
        with zipfile.ZipFile(package, "r") as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ALPR_PACKAGE_ENTRIES:
                raise MobileExportError("Pakiet ALPR zawiera zbyt wiele plikow.")
            names = [entry.filename.replace("\\", "/") for entry in entries]
            if "manifest.json" not in names:
                raise MobileExportError("Pakiet ALPR nie zawiera manifest.json.")
            for entry in entries:
                name = entry.filename.replace("\\", "/")
                if name in seen:
                    raise MobileExportError(f"Powtorzony wpis ZIP: {name}")
                seen.add(name)
                posix = PurePosixPath(name)
                if posix.is_absolute() or ".." in posix.parts:
                    raise MobileExportError(f"Niedozwolona sciezka w ZIP: {name}")
                total_uncompressed += int(entry.file_size or 0)
                if total_uncompressed > MAX_ALPR_UNCOMPRESSED_BYTES:
                    raise MobileExportError("Rozpakowany pakiet ALPR przekracza limit 1024 MiB.")

            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            self._validate_manifest_basic(manifest)
            name_set = set(names)
            models = dict(manifest.get("models") or {})
            roles_to_validate = ["plate", "character"]
            if isinstance(models.get("vehicle"), dict) and models.get("vehicle"):
                roles_to_validate.insert(0, "vehicle")
            for role in roles_to_validate:
                item = dict(models.get(role) or {})
                package_file = str(item.get("package_file") or "").replace("\\", "/")
                manifest_file = str(item.get("manifest_file") or "").replace("\\", "/")
                for relative in (package_file, manifest_file):
                    if relative not in name_set:
                        raise MobileExportError(f"Brak pliku modelu {role}: {relative}")
                    expected = str((item.get("sha256") or {}).get(relative) or "").lower()
                    if not re.fullmatch(r"[0-9a-f]{64}", expected):
                        raise MobileExportError(f"Brak sumy SHA-256 dla {relative}.")
                    actual = hashlib.sha256(archive.read(relative)).hexdigest()
                    if actual != expected:
                        raise MobileExportError(f"Niezgodna suma SHA-256 pliku: {relative}")

            with tempfile.TemporaryDirectory(prefix="alpr_nested_validate_") as temp_name:
                temp_root = Path(temp_name)
                for role in roles_to_validate:
                    item = dict(models.get(role) or {})
                    package_file = str(item.get("package_file") or "").replace("\\", "/")
                    nested_path = temp_root / f"{role}.alprmodel"
                    nested_path.write_bytes(archive.read(package_file))
                    self.single_model_exporter.validate_package(nested_path)
                    nested_manifest = self._read_single_manifest(nested_path)
                    if _normalize_role(str(nested_manifest.get("role") or "")) != role:
                        raise MobileExportError(f"Zagniezdzony pakiet {role} ma niewlasciwa role.")

    def _preflight_model_source(self, request: MobileAlprPackageRequest, role: str, *, required: bool) -> list[str]:
        if role == "vehicle":
            package = request.vehicle_package
            child_request = request.vehicle_request
            label = "MP"
        elif role == "plate":
            package = request.plate_package
            child_request = request.plate_request
            label = "MT"
        else:
            package = request.character_package
            child_request = request.character_request
            label = "MZ"

        has_package = package is not None and str(package).strip() != ""
        has_request = child_request is not None
        if not required and not has_package and not has_request:
            return []
        if required and not has_package and not has_request:
            return [COMPLETE_ALPR_MODELS_REQUIRED]
        if has_package == has_request:
            return [f"{label}: wskaz dokladnie jedno zrodlo: gotowy .alprmodel albo checkpoint do eksportu."]

        problems: list[str] = []
        if has_package:
            try:
                self._validate_single_model_package(Path(package), role=role)
            except Exception as exc:
                problems.append(f"{label}: {exc}")
            return problems

        assert child_request is not None
        try:
            request_role = _normalize_role(str(child_request.role or ""))
            if request_role != role:
                problems.append(f"{label}: rola requestu to {request_role}, oczekiwano {role}.")
        except Exception as exc:
            problems.append(f"{label}: {exc}")
        for problem in self.single_model_exporter.preflight(child_request):
            problems.append(f"{label}: {problem}")
        return problems

    def _export_single_child(
        self,
        request: MobileExportRequest,
        *,
        role: str,
        destination: Path,
        label: str,
        offset: float,
        span: float,
        notify: Callable[[float | None, str], None],
    ) -> Path:
        safe_request = replace(request, role=role, destination=destination)

        def child_progress(value: float | None, message: str) -> None:
            if value is None:
                notify(None, f"{label}: {message}")
            else:
                notify(offset + (max(0.0, min(100.0, float(value))) / 100.0) * span, f"{label}: {message}")

        return self.single_model_exporter.export(safe_request, progress=child_progress)

    def _stage_single_model_package(self, package_root: Path, *, role: str, source_package: Path) -> dict[str, Any]:
        source = Path(source_package).resolve()
        self._validate_single_model_package(source, role=role)
        child_manifest = self._read_single_manifest(source)

        model_dir = package_root / "models" / role
        model_dir.mkdir(parents=True, exist_ok=True)
        package_name = "model.alprmodel"
        package_target = model_dir / package_name
        manifest_target = model_dir / "manifest.json"
        shutil.copy2(source, package_target)
        manifest_target.write_text(json.dumps(child_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        package_rel = package_target.relative_to(package_root).as_posix()
        manifest_rel = manifest_target.relative_to(package_root).as_posix()
        training = dict(child_manifest.get("training") or {})
        metrics = dict(child_manifest.get("metrics") or {})
        source_meta = dict(child_manifest.get("source") or {})
        model_meta = dict(child_manifest.get("model") or {})
        item = {
            "role": role,
            "task": str(child_manifest.get("task") or ""),
            "model_id": str(child_manifest.get("model_id") or ""),
            "name": str(child_manifest.get("name") or ""),
            "version": str(child_manifest.get("version") or ""),
            "schema": str(child_manifest.get("schema") or ""),
            "package_file": package_rel,
            "manifest_file": manifest_rel,
            "sha256": {
                package_rel: sha256_file(package_target),
                manifest_rel: sha256_file(manifest_target),
            },
            "input": dict(child_manifest.get("input") or {}),
            "output": dict(child_manifest.get("output") or {}),
            "labels": list(child_manifest.get("labels") or []),
            "label_count": len(list(child_manifest.get("labels") or [])),
            "variants": list(child_manifest.get("variants") or []),
            "variant_count": len(list(child_manifest.get("variants") or [])),
            "training": training,
            "metrics": metrics,
            "source": source_meta,
            "model": model_meta,
            "export_reproducibility": dict(child_manifest.get("export_reproducibility") or {}),
        }
        if role == "vehicle":
            vehicle_detection = dict(child_manifest.get("vehicle_detection") or {})
            if vehicle_detection:
                item["vehicle_detection"] = vehicle_detection
        return item

    def _validate_single_model_package(self, package_path: Path, *, role: str) -> None:
        safe_path = Path(package_path)
        self.single_model_exporter.validate_package(safe_path)
        manifest = self._read_single_manifest(safe_path)
        actual_role = _normalize_role(str(manifest.get("role") or ""))
        if actual_role != role:
            raise MobileExportError(f"Pakiet ma role {actual_role}, oczekiwano {role}.")

    def _read_single_manifest(self, package_path: Path) -> dict[str, Any]:
        with zipfile.ZipFile(Path(package_path), "r") as archive:
            return json.loads(archive.read("manifest.json").decode("utf-8"))

    def _build_manifest(
        self,
        *,
        request: MobileAlprPackageRequest,
        vehicle_item: dict[str, Any] | None = None,
        plate_item: dict[str, Any],
        character_item: dict[str, Any],
    ) -> dict[str, Any]:
        models = {
            "plate": plate_item,
            "character": character_item,
        }
        pipeline = [
            {"stage": "plate_detection", "model": "plate", "role": "plate", "task": "pose"},
            {"stage": "plate_rectification", "implementation": "android_alpr_rectifier"},
            {"stage": "character_detection", "model": "character", "role": "character", "task": "detect"},
            {"stage": "sequence_assembly", "implementation": "android_alpr_sequence_decoder"},
        ]
        package_seed = f"ALPR-{plate_item.get('model_id')}-{character_item.get('model_id')}"
        if vehicle_item:
            vehicle_detection = dict(vehicle_item.get("vehicle_detection") or {})
            vehicle_stage = {"stage": "vehicle_detection", "model": "vehicle", "role": "vehicle", "task": "detect"}
            if vehicle_detection:
                vehicle_stage["filter"] = {
                    "filter_mode": str(vehicle_detection.get("filter_mode") or "include"),
                    "include_labels": list(vehicle_detection.get("include_labels") or []),
                    "include_class_indices": list(vehicle_detection.get("include_class_indices") or []),
                    "fallback_coco_class_indices": list(vehicle_detection.get("fallback_coco_class_indices") or []),
                }
            models = {
                "vehicle": vehicle_item,
                **models,
            }
            pipeline = [
                vehicle_stage,
                *pipeline,
            ]
            package_seed = (
                f"ALPR-{vehicle_item.get('model_id')}-{plate_item.get('model_id')}-"
                f"{character_item.get('model_id')}"
            )
        package_id = _safe_id(
            request.package_id
            or package_seed,
            fallback="ALPR-package",
        )
        created_at = _utc_now_iso()
        model_refs = {
            role: _package_model_ref_payload(role, item)
            for role, item in models.items()
            if isinstance(item, Mapping)
        }
        manifest = {
            "schema": MOBILE_ALPR_PACKAGE_SCHEMA,
            "package_id": package_id,
            "name": request.name or package_id,
            "version": str(request.version or "1"),
            "kind": "complete_alpr_pipeline",
            "created_at": created_at,
            "models": models,
            "model_refs": model_refs,
            "pipeline": pipeline,
            "ranking_dataset": dict(request.ranking_dataset or {}),
            "calibration_dataset": dict(request.calibration_dataset or {}),
            "metadata": dict(request.metadata or {}),
            "exporter": {
                "name": "auto_annotation_tool.mobile_alpr_package_exporter",
                "schema": MOBILE_ALPR_PACKAGE_SCHEMA,
                "single_model_schema": MOBILE_MODEL_SCHEMA,
                "pipeline_variant": "MP+MT+MZ" if vehicle_item else "MT+MZ",
            },
            "export_reproducibility": _alpr_package_reproducibility_payload(
                request=request,
                package_id=package_id,
                created_at=created_at,
                models=models,
                pipeline=pipeline,
            ),
        }
        return manifest

    def _validate_manifest_basic(self, manifest: dict[str, Any]) -> None:
        if not isinstance(manifest, dict):
            raise MobileExportError("Manifest pakietu ALPR nie jest obiektem JSON.")
        if manifest.get("schema") != MOBILE_ALPR_PACKAGE_SCHEMA:
            raise MobileExportError(f"Nieobslugiwany schemat pakietu ALPR: {manifest.get('schema')}")
        package_id = str(manifest.get("package_id") or "").strip()
        if not SAFE_MODEL_ID_RE.match(package_id):
            raise MobileExportError(f"Nieprawidlowy package_id: {package_id}")
        models = dict(manifest.get("models") or {})
        if not models.get("plate") or not models.get("character"):
            raise MobileExportError(COMPLETE_ALPR_MODELS_REQUIRED)
        roles = ["plate", "character"]
        has_vehicle = bool(isinstance(models.get("vehicle"), dict) and models.get("vehicle"))
        if has_vehicle:
            roles.insert(0, "vehicle")
        for role in roles:
            item = dict(models.get(role) or {})
            if not item:
                raise MobileExportError(f"Manifest pakietu ALPR nie zawiera modelu {role}.")
            if str(item.get("role") or "").strip().lower() != role:
                raise MobileExportError(f"Model {role} ma niespojna role w manifiescie.")
            task = str(item.get("task") or "").strip().lower()
            expected_task = "pose" if role == "plate" else "detect"
            if task != expected_task:
                raise MobileExportError(f"Model {role} ma task {task}, oczekiwano {expected_task}.")
            if str(item.get("schema") or "") != MOBILE_MODEL_SCHEMA:
                raise MobileExportError(f"Model {role} nie jest pakietem {MOBILE_MODEL_SCHEMA}.")
            if not str(item.get("package_file") or "").strip():
                raise MobileExportError(f"Model {role} nie wskazuje pliku .alprmodel.")
            if not isinstance(item.get("sha256"), dict) or not item.get("sha256"):
                raise MobileExportError(f"Model {role} nie ma sum SHA-256.")
        pipeline = list(manifest.get("pipeline") or [])
        expected_stages = (
            ["vehicle_detection", "plate_detection", "plate_rectification", "character_detection", "sequence_assembly"]
            if has_vehicle
            else ["plate_detection", "plate_rectification", "character_detection", "sequence_assembly"]
        )
        actual_stages = [str(item.get("stage") or "") for item in pipeline if isinstance(item, dict)]
        if actual_stages != expected_stages:
            raise MobileExportError(
                "Manifest pakietu ALPR ma niespojny pipeline: "
                f"{actual_stages}, oczekiwano {expected_stages}."
            )

        try:
            import jsonschema  # type: ignore

            jsonschema.validate(manifest, self.schema())
        except ImportError:
            pass
        except Exception as exc:
            raise MobileExportError(f"Manifest pakietu ALPR nie przechodzi JSON Schema: {exc}") from exc

    def _write_zip(self, package_root: Path, destination: Path) -> None:
        written: set[str] = set()
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package_root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(package_root).as_posix()
                posix = PurePosixPath(relative)
                if posix.is_absolute() or ".." in posix.parts:
                    raise MobileExportError(f"Niedozwolona sciezka pakietu ALPR: {relative}")
                if relative in written:
                    raise MobileExportError(f"Powtorzony wpis pakietu ALPR: {relative}")
                written.add(relative)
                archive.write(path, relative)

    @staticmethod
    def schema() -> dict:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": True,
            "required": ["schema", "package_id", "kind", "models", "pipeline"],
            "properties": {
                "schema": {"const": MOBILE_ALPR_PACKAGE_SCHEMA},
                "package_id": {"type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$"},
                "kind": {"const": "complete_alpr_pipeline"},
                "models": {
                    "type": "object",
                    "required": ["plate", "character"],
                    "properties": {
                        "vehicle": {"type": "object"},
                        "plate": {"type": "object"},
                        "character": {"type": "object"},
                    },
                },
                "pipeline": {"type": "array", "minItems": 4, "maxItems": 5},
            },
        }
