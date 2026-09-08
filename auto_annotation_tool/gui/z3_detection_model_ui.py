from __future__ import annotations

import csv
import json
import re
import tkinter as tk
from pathlib import Path
from textwrap import shorten
from tkinter import filedialog, messagebox, ttk

from ..campaign_ingest_planner import CHAR_ALPHABET
from ..campaign_manager import CAMPAIGN
from ..config import CONFIG, logger
from ..validators import format_yolo_model_identity, validate_model_file
from .web_slim_scrollbar import blend_hex_colors


def get_detection_active_model_status(host) -> tuple[str, str]:
    method_key = host._get_detection_method_key()
    uses_yolo = method_key in ("YOLO", "BOTH", "YOLO_OCR", "YOLO_BOX", "YOLO_SYMBOL")
    model_path = str(host._get_effective_yolo_model_path() or "").strip()
    path_locked = bool(getattr(host, "_step3_linear_mode", False))

    if model_path and Path(model_path).exists():
        model_name = Path(model_path).name
        version, size = host._infer_yolo_arch_from_model_path(model_path)
        if version and size:
            model_name = f"{model_name} (YOLOv{version}{size})"

        source_suffix = " z projektu" if path_locked else ""
        if uses_yolo:
            return (
                f"Model detekcji znaków w PZ2: {model_name}{source_suffix}. "
                "Służy tylko do inferencji: wykrywa/proponuje ramki i klasy znaków. "
                "Nie jest wyborem modelu do treningu.",
                "neutral",
            )
        return (
            f"Model detekcji znaków gotowy: {model_name}{source_suffix}. "
            "Aktualny pipeline OCR go nie używa; wybór nie zmienia modelu treningowego.",
            "muted",
        )

    if uses_yolo:
        return (
            "Brak modelu detekcji znaków dla PZ2. Wybierz wytrenowany .pt tylko wtedy, "
            "gdy pipeline ma korzystać z YOLO; trening wybierasz w karcie treningu.",
            "warning",
        )
    if path_locked:
        return "Model detekcji znaków jest sterowany przez projekt i nieużywany w trybie OCR.", "muted"
    return "Model detekcji znaków nie jest używany w trybie OCR.", "muted"


def has_configured_yolo_detection_model(host) -> bool:
    try:
        host._sync_yolo_model_binding()
    except Exception:
        pass

    model_path = str(host._get_effective_yolo_model_path() or "").strip()
    if not model_path:
        return False
    try:
        return Path(model_path).exists()
    except Exception:
        return False


def get_campaign_char_model_path(host) -> str:
    try:
        before_iteration = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        before_iteration = None
    try:
        model_info = dict(
            CAMPAIGN.get_effective_project_model(
                "char",
                before_iteration=before_iteration,
            )
            or {}
        )
        model_path = str(model_info.get("path") or "").strip()
        if model_path and Path(model_path).exists():
            return str(Path(model_path))
    except Exception:
        pass
    return ""


def get_campaign_detection_yolo_model_path(host) -> str:
    try:
        model_path = str(CAMPAIGN.get_step3_detection_yolo_model() or "").strip()
    except Exception:
        model_path = ""
    if model_path:
        try:
            path = Path(model_path)
            if path.exists() and path.is_file():
                return str(path)
        except Exception:
            pass
    return ""


def get_effective_yolo_model_path(host) -> str:
    if getattr(host, "_step3_linear_mode", False):
        configured_model = host._get_campaign_detection_yolo_model_path()
        if configured_model:
            return configured_model
        return host._get_campaign_char_model_path()

    raw = (host.yolo_model_path_var.get() or "").strip()
    if raw and raw != "Brak modelu znaków w projekcie" and Path(raw).exists():
        return raw
    return ""


def sync_yolo_model_binding(host) -> None:
    if getattr(host, "_step3_linear_mode", False):
        project_model = host._get_campaign_detection_yolo_model_path() or host._get_campaign_char_model_path()
        if project_model:
            host.yolo_model_path_var.set(project_model)
        else:
            host.yolo_model_path_var.set("Brak modelu znaków w projekcie")
    else:
        if (host.yolo_model_path_var.get() or "").strip() == "Brak modelu znaków w projekcie":
            host.yolo_model_path_var.set("")


def infer_yolo_arch_from_model_path(model_path: str):
    raw = (model_path or "").strip().lower()
    if not raw:
        return None, None

    name = Path(raw).name.lower()
    match = re.search(r"yolo(8|11|26)([nsmlx])", name)
    if not match:
        return None, None

    version = match.group(1)
    size = match.group(2)
    return version, size


def auto_device_label() -> str:
    return "Auto"


def get_available_devices(host):
    # Opening PZ2 must not import torch or initialize CUDA on the Tk thread.
    # Hardware discovery belongs to the application's shared configuration.
    devices = [host._auto_device_label(), "CPU"]
    try:
        getter = getattr(host.app, "get_available_yolo_devices", None)
        if callable(getter):
            for option in getter(allow_probe=False):
                if str(option).lower().startswith("cuda:") and option not in devices:
                    devices.append(option)
    except Exception:
        pass
    return devices


def normalize_selected_device(host, raw_value: str | None = None, devices=None) -> str:
    available = list(host._get_available_devices() if devices is None else devices)
    current = str(raw_value if raw_value is not None else host.yolo_device_var.get() or "").strip()
    current_lower = current.lower()

    if not current or current_lower.startswith("auto"):
        return available[0] if available else host._auto_device_label()
    if current_lower.startswith("cpu"):
        return "CPU"
    if current_lower.startswith("cuda:"):
        prefix = current.split()[0]
        for option in available:
            if str(option or "").lower().split()[0] == prefix.lower():
                return option
        return host._auto_device_label()

    return current if (not available or current in available) else (available[0] if available else host._auto_device_label())


def get_effective_detection_device_choice(host) -> str:
    raw_choice = None
    try:
        app_device_getter = getattr(host.app, "get_global_yolo_device_choice", None)
        if callable(app_device_getter):
            raw_choice = app_device_getter()
    except Exception:
        raw_choice = None

    normalized = host._normalize_selected_device(raw_value=raw_choice or host.yolo_device_var.get())
    try:
        if normalized and host.yolo_device_var.get() != normalized:
            host.yolo_device_var.set(normalized)
    except Exception:
        pass
    return normalized


def refresh_device_options(host) -> None:
    devices = host._get_available_devices()
    if hasattr(host, "det_device_combo"):
        try:
            host.det_device_combo.configure(values=devices)
        except Exception:
            pass

    normalized = host._normalize_selected_device(devices=devices)
    if normalized:
        host.yolo_device_var.set(normalized)

    host._update_device_hint()


def device_to_ultralytics(host, s: str):
    raw = str(s or "").strip().lower()
    if not raw or raw.startswith("auto"):
        try:
            import torch

            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
        return "cpu"
    if raw.startswith("cpu"):
        return "cpu"
    if raw.startswith("cuda:"):
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
            return int(str(s).split(":")[1].split()[0])
        except Exception:
            return "cpu"
    return "cpu"


def device_to_ocr(host, s: str) -> str:
    return "cuda" if host._device_to_ultralytics(s) != "cpu" else "cpu"


def update_device_hint(host, event=None) -> None:
    label = getattr(host, "det_device_hint_lbl", None)
    if label is None:
        return

    devices = host._get_available_devices()
    normalized = host._normalize_selected_device(devices=devices)
    current = str(host.yolo_device_var.get() or "").strip()
    if normalized != current:
        host.yolo_device_var.set(normalized)
        current = normalized

    gpu_devices = [item for item in devices if item.startswith("cuda:")]
    current_lower = current.lower()

    if current_lower.startswith("auto"):
        if gpu_devices:
            text = f"Auto najpierw spróbuje akceleracji na {gpu_devices[0]}. Gdy GPU/CUDA nie będzie dostępne, system spadnie do CPU."
            tone = "info"
        elif bool(getattr(host.app, "_global_yolo_devices_cache_ready", False)):
            text = "Auto nie wykryło karty CUDA, więc zostanie użyty CPU."
            tone = "warning"
        else:
            text = "Auto dobierze urządzenie przy detekcji. Dostępność GPU sprawdzisz w Konfiguracji."
            tone = "info"
    elif current_lower.startswith("cpu"):
        text = "CPU wymusza pracę bez akceleracji GPU. To wolniejsze, ale przewidywalne."
        tone = "muted"
    else:
        text = f"Wybrana karta: {current}. YOLO i OCR spróbują użyć tej akceleracji."
        tone = "success"

    host._set_themed_label_state(label, text=text, tone=tone)


def apply_global_yolo_device_choice(host, value: str) -> None:
    normalized = host._normalize_selected_device(raw_value=value)
    try:
        host.yolo_device_var.set(normalized)
    except Exception:
        pass
    try:
        host._update_device_hint()
    except Exception:
        pass


def set_button_emphasis(host, frame_attr: str, enabled: bool, color: str = "#f39c12") -> None:
    btn = host._resolve_guidance_button(frame_attr)
    if btn is None:
        return

    try:
        host.app.set_button_emphasis(btn, enabled)
    except Exception as exc:
        logger.debug(f"Nie udało się ustawić podświetlenia przycisku dla {frame_attr}: {exc}")


def ensure_yolo_model_checkpoint(host) -> str:
    """
    Zwraca ścieżkę do wytrenowanego checkpointu YOLO znaków.
    Z3/PZ2 nie korzysta z niewytrenowanych wariantów architektury.
    """
    effective_model = host._get_effective_yolo_model_path()
    if effective_model and Path(effective_model).exists():
        model_path = Path(effective_model)
        if model_path.suffix.lower() != ".pt":
            raise RuntimeError("Model YOLO znaków musi mieć rozszerzenie .pt.")
        host._log(host.test_log_text, f"[INFO] Używam lokalnego modelu: {model_path}", "INFO")
        return str(model_path)

    if getattr(host, "_step3_linear_mode", False):
        raise RuntimeError(
            "Brak wytrenowanego modelu znaków przypiętego do projektu. "
            "Najpierw przygotuj i wytrenuj model w Z4."
        )

    raise RuntimeError(
        "Wskaż wytrenowany model YOLO znaków (.pt). "
        "W Z3/PZ2 nie korzystamy z niewytrenowanych wariantów architektury."
    )


def _detection_model_sidecar_candidates(model_path: Path) -> list[Path]:
    candidates: list[Path] = []

    def _add(candidate: Path | None) -> None:
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
        _add(model_path.with_suffix(f"{suffix}.metadata.json"))
    _add(model_path.with_suffix(".metadata.json"))
    _add(model_path.with_suffix(".json"))
    _add(model_path.with_name(f"{model_path.stem}_metadata.json"))
    _add(model_path.with_name("model_metadata.json"))
    _add(model_path.with_name("metadata.json"))
    return candidates


def _read_detection_model_extra_metadata(model_path: Path) -> dict:
    for metadata_path in _detection_model_sidecar_candidates(Path(model_path)):
        if not metadata_path.exists() or not metadata_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        if not metrics and isinstance(extra.get("metrics"), dict):
            metrics = extra.get("metrics") or {}
        training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
        if not training and isinstance(extra.get("training"), dict):
            training = extra.get("training") or {}
        model_payload = payload.get("model") if isinstance(payload.get("model"), dict) else {}
        return {
            "metadata_path": str(metadata_path),
            "created_at": str(payload.get("created_at") or "").strip(),
            "metrics": dict(metrics or {}),
            "training": dict(training or {}),
            "model": dict(model_payload or {}),
            "schema": str(payload.get("schema") or "").strip(),
        }
    return {}


def _detection_metric_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _detection_metric_percent(value) -> str:
    numeric = _detection_metric_float(value)
    if numeric is None:
        return "-"
    if abs(numeric) <= 1.000001:
        numeric *= 100.0
    return f"{numeric:.1f}%"


def _detection_metric_value(metrics: dict, keys: tuple[str, ...]):
    if not isinstance(metrics, dict):
        return None
    for key in keys:
        if key in metrics and metrics.get(key) not in (None, ""):
            return metrics.get(key)
    latest = metrics.get("latest") if isinstance(metrics.get("latest"), dict) else {}
    best_row = metrics.get("best_row") if isinstance(metrics.get("best_row"), dict) else {}
    for source in (best_row, latest):
        for key in keys:
            if key in source and source.get(key) not in (None, ""):
                return source.get(key)
    return None


def _read_detection_results_csv_metrics(model_path: Path) -> dict:
    candidates: list[Path] = []
    try:
        candidates.append(model_path.parent.parent / "results.csv")
        candidates.append(model_path.parent / "results.csv")
    except Exception:
        pass
    for csv_path in candidates:
        if not csv_path.exists() or not csv_path.is_file():
            continue
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            continue
        if not rows:
            continue

        best_row = None
        best_score = None
        for row in rows:
            score = _detection_metric_float(
                row.get("metrics/mAP50-95(B)")
                or row.get("metrics/mAP50-95(P)")
                or row.get("metrics/mAP50-95")
            )
            if score is None:
                score = _detection_metric_float(
                    row.get("metrics/mAP50(B)")
                    or row.get("metrics/mAP50(P)")
                    or row.get("metrics/mAP50")
                )
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_row = row
        best_row = best_row or rows[-1]
        return {
            "map50": _detection_metric_float(
                best_row.get("metrics/mAP50(B)")
                or best_row.get("metrics/mAP50(P)")
                or best_row.get("metrics/mAP50")
            ),
            "map50_95": _detection_metric_float(
                best_row.get("metrics/mAP50-95(B)")
                or best_row.get("metrics/mAP50-95(P)")
                or best_row.get("metrics/mAP50-95")
            ),
            "precision": _detection_metric_float(
                best_row.get("metrics/precision(B)")
                or best_row.get("metrics/precision(P)")
                or best_row.get("metrics/precision")
            ),
            "recall": _detection_metric_float(
                best_row.get("metrics/recall(B)")
                or best_row.get("metrics/recall(P)")
                or best_row.get("metrics/recall")
            ),
            "epoch": _detection_metric_float(best_row.get("epoch")),
            "metrics_source": str(csv_path),
        }
    return {}


def get_detection_model_quick_stats(model_path: str | Path) -> dict:
    safe_path = Path(model_path)
    if not str(safe_path).strip() or not safe_path.exists():
        return {"identity": "", "map50_95_text": "-"}

    extra = _read_detection_model_extra_metadata(safe_path)
    metrics = dict(extra.get("metrics") or {})
    model_payload = extra.get("model") if isinstance(extra.get("model"), dict) else {}
    model_info = model_payload.get("info") if isinstance(model_payload.get("info"), dict) else {}
    info = dict(model_info or {})

    csv_metrics = _read_detection_results_csv_metrics(safe_path)
    for key, value in csv_metrics.items():
        if value not in (None, "") and metrics.get(key) in (None, ""):
            metrics[key] = value
        if value not in (None, "") and info.get(key) in (None, ""):
            info[key] = value

    identity = str(format_yolo_model_identity(info) or "").strip()
    if not identity:
        identity = str(info.get("architecture_label") or info.get("source_architecture_label") or "").strip()

    map5095 = _detection_metric_value(
        metrics,
        (
            "box_map50_95",
            "map50_95",
            "best_map50_95",
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95(P)",
            "metrics/mAP50-95",
        ),
    )
    return {
        "identity": identity,
        "map50_95_text": _detection_metric_percent(map5095),
    }


def _detection_model_class_names(info: dict) -> list[str]:
    raw = (
        info.get("classes")
        or info.get("names")
        or info.get("class_names")
        or info.get("labels")
        or []
    )
    if isinstance(raw, dict):
        try:
            ordered = [raw[key] for key in sorted(raw, key=lambda item: int(item))]
        except Exception:
            ordered = list(raw.values())
        return [str(name).strip() for name in ordered if str(name).strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(name).strip() for name in raw if str(name).strip()]
    return []


def _detection_looks_like_character_model_classes(class_names: list[str]) -> bool:
    normalized = [str(name).strip().upper() for name in class_names if str(name).strip()]
    if len(normalized) < 8:
        return False
    allowed = set(CHAR_ALPHABET)
    return all(len(token) == 1 and token in allowed for token in normalized)


def _detection_is_pose_model_info(info: dict) -> bool:
    task = str(info.get("task") or "").strip().lower()
    inferred_type = str(info.get("type") or "").strip().lower()
    kpt_shape = info.get("kpt_shape")
    has_keypoints = bool(info.get("keypoints")) or bool(kpt_shape)
    if has_keypoints or task == "pose" or inferred_type == "pose":
        return True
    if task in {"detect", "detection"} or inferred_type in {"detect", "detection"}:
        return False
    architecture_text = " ".join(
        str(info.get(key) or "").strip().lower()
        for key in ("architecture_label", "source_architecture_label")
    )
    return "pose" in architecture_text


def _format_detection_model_source(host, path_value) -> str:
    raw_text = str(path_value or "").strip()
    if not raw_text:
        return "Nie wskazano"
    try:
        safe_path = Path(raw_text).resolve()
    except Exception:
        safe_path = Path(raw_text)

    formatter_names = (
        "_format_project_relative_path",
        "_format_workspace_relative_path",
    )
    for formatter_name in formatter_names:
        formatter = getattr(host, formatter_name, None)
        if not callable(formatter):
            continue
        try:
            formatted = str(formatter(str(safe_path)) or "").strip()
            if formatted:
                return formatted
        except Exception:
            pass

    try:
        project_root = CAMPAIGN.get_active_project_root_dir()
        if project_root is not None:
            project_root = Path(project_root).resolve()
            if str(safe_path).lower().startswith(str(project_root).lower()):
                return str(safe_path.relative_to(project_root))
    except Exception:
        pass

    try:
        workspace_root = Path(CONFIG.WORKSPACE_DIR).resolve()
        if str(safe_path).lower().startswith(str(workspace_root).lower()):
            return str(safe_path.relative_to(workspace_root))
    except Exception:
        pass
    return str(safe_path)


def _detection_model_candidate_roots(host) -> list[Path]:
    roots: list[Path] = []

    def _add_root(path_value) -> None:
        if not path_value:
            return
        try:
            path = Path(path_value)
        except Exception:
            return
        if not path.exists() or not path.is_dir():
            return
        try:
            key = str(path.resolve()).lower()
        except Exception:
            key = str(path).lower()
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    seen: set[str] = set()
    try:
        active_model = str(host._get_effective_yolo_model_path() or "").strip()
        if active_model:
            active_path = Path(active_model)
            if active_path.exists():
                _add_root(active_path.parent)
    except Exception:
        pass
    try:
        project_model = str(CAMPAIGN.get_global_model("char") or "").strip()
        if project_model:
            project_path = Path(project_model)
            if project_path.exists():
                _add_root(project_path.parent)
    except Exception:
        pass
    try:
        project_root = CAMPAIGN.get_active_project_root_dir()
        if project_root is not None:
            project_root = Path(project_root)
            _add_root(project_root / "5_training_runs")
            _add_root(project_root / "6_models")
    except Exception:
        pass
    try:
        for candidate in CONFIG.get_model_search_dirs("char"):
            _add_root(candidate)
    except Exception:
        _add_root(CONFIG.get_trained_models_dir("char"))
        _add_root(CONFIG.DIR_6_MODELS)
    return roots


def _summarize_detection_model_candidate(host, model_path: Path, root: Path | None = None) -> dict:
    safe_path = Path(model_path)
    try:
        stat = safe_path.stat()
        mtime = float(getattr(stat, "st_mtime", 0.0) or 0.0)
        size_mb = float(getattr(stat, "st_size", 0) or 0) / (1024 * 1024)
    except Exception:
        mtime = 0.0
        size_mb = 0.0

    try:
        ok_light, message, info = validate_model_file(
            safe_path,
            allow_heavy_load=False,
            write_sidecar=False,
        )
    except Exception as exc:
        ok_light, message, info = False, str(exc), {}
    info = dict(info or {})

    extra = _read_detection_model_extra_metadata(safe_path)
    metrics = dict(extra.get("metrics") or {})
    training = dict(extra.get("training") or {})
    model_payload = extra.get("model") if isinstance(extra.get("model"), dict) else {}
    model_info = model_payload.get("info") if isinstance(model_payload.get("info"), dict) else {}
    if model_info:
        for key, value in model_info.items():
            if key not in info or info.get(key) in (None, "", [], {}):
                info[key] = value

    csv_metrics = _read_detection_results_csv_metrics(safe_path)
    for key, value in csv_metrics.items():
        if value not in (None, "") and metrics.get(key) in (None, ""):
            metrics[key] = value
        if value not in (None, "") and info.get(key) in (None, ""):
            info[key] = value

    identity = str(format_yolo_model_identity(info) or "").strip()
    if not identity:
        identity = str(info.get("architecture_label") or info.get("source_architecture_label") or "").strip()
    if not identity:
        identity = "YOLO Pose" if "pose" in safe_path.name.lower() else "YOLO Detect" if safe_path.suffix.lower() == ".pt" else "Nieustalone"

    class_names = _detection_model_class_names(info)
    looks_like_char_model = _detection_looks_like_character_model_classes(class_names)
    is_pose = _detection_is_pose_model_info(info)
    task = str(info.get("task") or info.get("type") or "").strip().lower()
    path_hint = str(safe_path).lower()
    char_path_hint = any(token in path_hint for token in ("char", "chars", "znak", "characters_ocr"))
    target_match = (not is_pose) and (
        looks_like_char_model
        or (task in {"detect", "detection"} and not class_names and char_path_hint)
        or (not class_names and char_path_hint and safe_path.suffix.lower() == ".pt")
    )

    map50 = _detection_metric_value(
        metrics,
        ("box_map50", "map50", "best_map50", "metrics/mAP50(B)", "metrics/mAP50"),
    )
    map5095 = _detection_metric_value(
        metrics,
        (
            "box_map50_95",
            "map50_95",
            "best_map50_95",
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95",
        ),
    )
    precision = _detection_metric_value(
        metrics,
        ("box_precision", "precision", "metrics/precision(B)", "metrics/precision"),
    )
    recall = _detection_metric_value(
        metrics,
        ("box_recall", "recall", "metrics/recall(B)", "metrics/recall"),
    )

    has_light_info = bool(ok_light or extra or csv_metrics)
    if target_match and has_light_info:
        tone = "success"
        status = "Pasuje"
    elif target_match:
        tone = "warning"
        status = "Sprawdź"
    elif is_pose:
        tone = "error"
        status = "Model tablic"
    else:
        tone = "error"
        status = "Inny typ"

    return {
        "path": str(safe_path),
        "name": safe_path.name,
        "root": _format_detection_model_source(host, root) if root is not None else "",
        "relative_path": _format_detection_model_source(host, safe_path),
        "identity": identity,
        "status": status,
        "tone": tone,
        "message": str(message or "").strip(),
        "target_match": bool(target_match),
        "map50": map50,
        "map50_95": map5095,
        "precision": precision,
        "recall": recall,
        "map50_text": _detection_metric_percent(map50),
        "map50_95_text": _detection_metric_percent(map5095),
        "precision_text": _detection_metric_percent(precision),
        "recall_text": _detection_metric_percent(recall),
        "size_mb": float(size_mb),
        "size_text": f"{size_mb:.1f} MB" if size_mb > 0 else "-",
        "mtime": mtime,
        "created_at": str(extra.get("created_at") or training.get("finished_at") or training.get("created_at") or "").strip(),
        "metadata_path": str(extra.get("metadata_path") or "").strip(),
    }


def _find_detection_model_candidates(host) -> tuple[list[dict], list[Path]]:
    roots = _detection_model_candidate_roots(host)
    candidates: list[dict] = []
    seen: set[str] = set()
    for root in roots:
        try:
            model_paths = Path(root).rglob("*.pt")
        except Exception:
            model_paths = ()
        for model_path in model_paths:
            try:
                key = str(Path(model_path).resolve()).lower()
            except Exception:
                key = str(model_path).lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(_summarize_detection_model_candidate(host, Path(model_path), root))
            if len(candidates) >= 260:
                break
        if len(candidates) >= 260:
            break

    def _score(item: dict) -> tuple[int, float, float, float]:
        map5095 = _detection_metric_float(item.get("map50_95")) or -1.0
        map50 = _detection_metric_float(item.get("map50")) or -1.0
        if abs(map5095) <= 1.000001:
            map5095 *= 100.0
        if abs(map50) <= 1.000001:
            map50 *= 100.0
        return (
            1 if bool(item.get("target_match")) else 0,
            float(map5095),
            float(map50),
            float(item.get("mtime", 0.0) or 0.0),
        )

    candidates.sort(key=_score, reverse=True)
    return candidates, roots


def _apply_detection_yolo_model_selection(host, chosen: Path) -> str:
    if chosen.suffix.lower() != ".pt" or not chosen.exists():
        messagebox.showwarning(
            "Błędny model YOLO",
            "Wskaż poprawny, wytrenowany model YOLO znaków z rozszerzeniem .pt.",
        )
        return ""

    if getattr(host, "_step3_linear_mode", False):
        try:
            CAMPAIGN.set_step3_detection_yolo_model(str(chosen))
        except Exception as exc:
            logger.debug(f"Nie udalo sie zapisac modelu detekcji PZ2 w projekcie: {exc}")

    host.yolo_model_path_var.set(str(chosen))
    try:
        host._force_save_all()
    except Exception:
        pass
    host._refresh_detect_mode_cards()
    host._update_yolo_visibility()
    try:
        host._refresh_detection_pipeline_builder()
    except Exception:
        pass
    return str(chosen)


def _pick_yolo_model_file_dialog(host, initial_dir: Path | None = None) -> str:
    safe_initial_dir = initial_dir or CONFIG.get_trained_models_dir("char")
    if not safe_initial_dir.exists():
        safe_initial_dir = CONFIG.DIR_6_MODELS
    p = filedialog.askopenfilename(
        initialdir=str(Path(safe_initial_dir).absolute()),
        title="Wybierz model detekcji YOLO znaków (.pt)",
        filetypes=[("PyTorch", "*.pt")]
    )
    if not p:
        return ""

    try:
        chosen = Path(p)
    except Exception:
        return ""
    return _apply_detection_yolo_model_selection(host, chosen)


def _open_detection_yolo_model_candidate_browser(host) -> str:
    candidates, roots = _find_detection_model_candidates(host)
    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    field_bg = palette.get("field", "#1a1a1a")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    success = palette.get("success", "#2ecc71")
    warning = palette.get("warning", "#f39c12")
    error = palette.get("error", "#e74c3c")
    accent = palette.get("accent", success)
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    selected_path: dict[str, str] = {"value": ""}

    parent = getattr(host, "_detection_pipeline_modal", None) or getattr(host, "frame", None)
    browser = tk.Toplevel(parent or host.frame)
    try:
        host.app.style_dialog_window(
            browser,
            title="Wybierz model detekcji znaków",
            geometry="1180x680",
            parent=parent or host.frame,
        )
    except Exception:
        browser.title("Wybierz model detekcji znaków")

    build_surface = getattr(host.app, "_build_themed_dialog_surface", None)
    if callable(build_surface):
        body = build_surface(browser, tone="info")
    else:
        body = tk.Frame(browser, bg=panel_bg)
        body.pack(fill=tk.BOTH, expand=True)
    body_bg = str(body.cget("bg") or panel_bg)

    tk.Label(
        body,
        text="Model detekcji znaków dla PZ2",
        fg=fg,
        bg=body_bg,
        font=("Segoe UI", 13, "bold"),
        anchor="w",
    ).pack(fill=tk.X, padx=16, pady=(14, 4))

    roots_text = ", ".join(str(root.name or root) for root in roots[:4]) or "brak katalogów"
    if len(roots) > 4:
        roots_text += f" +{len(roots) - 4}"
    tk.Label(
        body,
        text=(
            "Wybierz wytrenowany model YOLO Detect używany wyłącznie do inferencji w PZ2: "
            "wykrywania i proponowania ramek znaków. Ten wybór nie zmienia modelu startowego treningu. "
            f"Skanowane katalogi: {roots_text}."
        ),
        fg=muted,
        bg=body_bg,
        justify=tk.LEFT,
        anchor="w",
        wraplength=1110,
    ).pack(fill=tk.X, padx=16, pady=(0, 12))

    table_shell = tk.Frame(
        body,
        bg=body_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
    )
    table_shell.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 10))

    header = (
        "Model",
        "Typ",
        "Status",
        "mAP50-95",
        "mAP50",
        "Precision",
        "Recall",
        "Rozmiar",
        "Ścieżka",
        "Wskaż",
    )
    tree_columns = tuple(f"c{i}" for i in range(len(header)))
    tree = ttk.Treeview(table_shell, columns=tree_columns, show="headings", selectmode="browse", height=16)
    tree_scroll = ttk.Scrollbar(table_shell, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=tree_scroll.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)

    try:
        style = ttk.Style(tree)
        style.configure(
            "DetectionModelCandidates.Treeview",
            background=field_bg,
            fieldbackground=field_bg,
            foreground=fg,
            rowheight=30,
            borderwidth=0,
        )
        style.map(
            "DetectionModelCandidates.Treeview",
            background=[("selected", blend_hex_colors(field_bg, accent, 0.18))],
            foreground=[("selected", fg)],
        )
        style.configure("DetectionModelCandidates.Treeview.Heading", font=("Segoe UI", 8, "bold"))
        tree.configure(style="DetectionModelCandidates.Treeview")
    except Exception:
        pass

    column_widths = (250, 132, 92, 78, 66, 76, 64, 70, 230, 72)
    candidate_by_iid: dict[str, dict] = {}
    sort_state = {"column": None, "reverse": True}
    hover_state: dict[str, str] = {"row_id": "", "column_id": ""}
    hover_label = tk.Label(
        tree,
        text="",
        fg=fg,
        bg=blend_hex_colors(field_bg, success, 0.20),
        bd=0,
        highlightthickness=0,
        font=("Segoe UI", 8, "bold"),
        anchor="center",
        cursor="hand2",
    )

    def _sort_metric(value) -> float:
        numeric = _detection_metric_float(value)
        if numeric is None:
            return -1.0
        if abs(numeric) <= 1.000001:
            numeric *= 100.0
        return float(numeric)

    def _sort_value(candidate: dict, col: int):
        if col == 0:
            return str(candidate.get("name") or "").strip().lower()
        if col == 1:
            return str(candidate.get("identity") or "").strip().lower()
        if col == 2:
            tone_rank = {"success": 3, "warning": 2, "error": 1}
            return (
                int(tone_rank.get(str(candidate.get("tone") or "").strip(), 0)),
                str(candidate.get("status") or "").strip().lower(),
            )
        if col == 3:
            return _sort_metric(candidate.get("map50_95"))
        if col == 4:
            return _sort_metric(candidate.get("map50"))
        if col == 5:
            return _sort_metric(candidate.get("precision"))
        if col == 6:
            return _sort_metric(candidate.get("recall"))
        if col == 7:
            return float(candidate.get("size_mb", 0.0) or 0.0)
        if col == 8:
            return str(candidate.get("relative_path") or "").strip().lower()
        return ""

    def _sorted_candidates() -> list[dict]:
        col = sort_state.get("column")
        if col is None:
            return list(candidates)
        return sorted(
            list(candidates),
            key=lambda item: (
                _sort_value(item, int(col)),
                float(item.get("mtime", 0.0) or 0.0),
                str(item.get("name") or "").strip().lower(),
            ),
            reverse=bool(sort_state.get("reverse")),
        )

    def _set_sort(col: int) -> None:
        if col >= len(header) - 1:
            return
        if sort_state.get("column") == col:
            sort_state["reverse"] = not bool(sort_state.get("reverse"))
        else:
            sort_state["column"] = col
            sort_state["reverse"] = col in {2, 3, 4, 5, 6, 7}
        _refresh_table()
        try:
            tree.yview_moveto(0)
        except Exception:
            pass

    def _select_candidate(candidate: dict) -> None:
        model_path = Path(str(candidate.get("path") or ""))
        if model_path.suffix.lower() != ".pt" or not model_path.exists():
            messagebox.showerror(
                "Nieprawidłowy model",
                "Wybrany plik nie istnieje albo nie ma rozszerzenia .pt.",
                parent=browser,
            )
            return

        if not bool(candidate.get("target_match")):
            proceed = messagebox.askyesno(
                "Model nie wygląda jak detektor znaków",
                (
                    "Wybrany plik nie wygląda jak model YOLO Detect znaków. "
                    "Użycie modelu tablic/pose w PZ2 może dać błędne ramki. "
                    "Czy mimo to użyć tego pliku?"
                ),
                parent=browser,
            )
            if not proceed:
                return

        selected = _apply_detection_yolo_model_selection(host, model_path)
        if not selected:
            return
        selected_path["value"] = selected
        try:
            host.app.update_status(f"Wybrano model detekcji znaków: {model_path.name}", "info")
        except Exception:
            pass
        _close_browser()

    def _refresh_table() -> None:
        _hide_hover()
        for item in tree.get_children(""):
            tree.delete(item)
        candidate_by_iid.clear()
        active_col = sort_state.get("column")
        reverse = bool(sort_state.get("reverse"))
        for col, col_id in enumerate(tree_columns):
            suffix = " ↓" if active_col == col and reverse else " ↑" if active_col == col else ""
            tree.heading(col_id, text=f"{header[col]}{suffix}", anchor="w")

        if not candidates:
            iid = "empty"
            tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    "Nie znaleziono kandydatów .pt.",
                    "Wskaż ręcznie",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ),
            )
            return

        for row_idx, candidate in enumerate(_sorted_candidates(), start=1):
            iid = f"model-{row_idx}"
            candidate_by_iid[iid] = candidate
            values = (
                str(candidate.get("name") or "-"),
                str(candidate.get("identity") or "-"),
                str(candidate.get("status") or "-"),
                str(candidate.get("map50_95_text") or "-"),
                str(candidate.get("map50_text") or "-"),
                str(candidate.get("precision_text") or "-"),
                str(candidate.get("recall_text") or "-"),
                str(candidate.get("size_text") or "-"),
                shorten(str(candidate.get("relative_path") or "-"), width=42, placeholder="..."),
                "[Wskaż]",
            )
            tree.insert("", "end", iid=iid, values=values)

    def _choose_selection() -> None:
        try:
            selection = tree.selection()
            if not selection:
                return
            candidate = candidate_by_iid.get(str(selection[0]))
            if candidate:
                _select_candidate(candidate)
        except Exception:
            pass

    def _hide_hover() -> None:
        try:
            hover_label.place_forget()
        except Exception:
            pass
        hover_state["row_id"] = ""
        hover_state["column_id"] = ""

    def _hover_target(event=None) -> tuple[str, str] | None:
        try:
            row_id = str(tree.identify_row(event.y) or "")
            column_id = str(tree.identify_column(event.x) or "")
            if row_id and row_id in candidate_by_iid and column_id == f"#{len(header)}":
                return row_id, column_id
        except Exception:
            return None
        return None

    def _show_hover(row_id: str, column_id: str) -> None:
        try:
            if (
                row_id == str(hover_state.get("row_id") or "")
                and column_id == str(hover_state.get("column_id") or "")
            ):
                return
            bbox = tree.bbox(row_id, column_id)
            if not bbox:
                _hide_hover()
                return
            x, y, width, height = bbox
            text = str(tree.set(row_id, tree_columns[-1]) or "")
            hover_label.config(text=text)
            hover_label.place(
                x=int(x) + 1,
                y=int(y) + 1,
                width=max(1, int(width) - 2),
                height=max(1, int(height) - 2),
            )
            hover_state["row_id"] = row_id
            hover_state["column_id"] = column_id
        except Exception:
            _hide_hover()

    def _update_hover(event=None) -> None:
        target = _hover_target(event)
        try:
            if target:
                tree.configure(cursor="hand2")
                _show_hover(target[0], target[1])
            else:
                tree.configure(cursor="")
                _hide_hover()
        except Exception:
            pass

    def _activate_hover(_event=None) -> str:
        try:
            row_id = str(hover_state.get("row_id") or "")
            if row_id and row_id in candidate_by_iid:
                tree.selection_set(row_id)
                tree.focus(row_id)
                _select_candidate(candidate_by_iid[row_id])
        except Exception:
            pass
        return "break"

    def _manual_choose() -> None:
        initial = CONFIG.get_trained_models_dir("char")
        try:
            if roots:
                initial = roots[0]
        except Exception:
            pass
        _close_browser()
        selected = _pick_yolo_model_file_dialog(host, initial)
        if selected:
            selected_path["value"] = selected

    def _close_browser() -> None:
        try:
            browser.grab_release()
        except Exception:
            pass
        try:
            browser.destroy()
        except Exception:
            pass

    for col, col_id in enumerate(tree_columns):
        tree.heading(col_id, command=lambda column=col: _set_sort(column))
        anchor = "center" if col in {3, 4, 5, 6, 7, 9} else "w"
        stretch = col in {0, 1, 8}
        tree.column(col_id, width=column_widths[col], minwidth=46, anchor=anchor, stretch=stretch)
    tree.bind("<Double-1>", lambda _event: _choose_selection(), add="+")
    tree.bind("<Return>", lambda _event: _choose_selection(), add="+")
    tree.bind("<ButtonRelease-1>", lambda event: _select_candidate(candidate_by_iid[str(tree.identify_row(event.y))]) if str(tree.identify_row(event.y) or "") in candidate_by_iid and str(tree.identify_column(event.x) or "") == f"#{len(header)}" else None, add="+")
    tree.bind("<Motion>", _update_hover, add="+")
    tree.bind("<Leave>", lambda _event: (_hide_hover(), tree.configure(cursor="")), add="+")
    hover_label.bind("<ButtonRelease-1>", _activate_hover, add="+")
    hover_label.bind("<Leave>", lambda _event: _hide_hover(), add="+")
    _refresh_table()

    footer = tk.Frame(body, bg=body_bg)
    footer.pack(fill=tk.X, padx=16, pady=(0, 14))
    tk.Label(
        footer,
        text="mAP50-95 jest głównym skrótem jakości; Precision i Recall pomagają ocenić ostrożność oraz czułość modelu.",
        fg=muted,
        bg=body_bg,
        anchor="w",
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    tk.Button(
        footer,
        text="Wskaż plik ręcznie",
        command=_manual_choose,
        cursor="hand2",
        bg=panel_alt,
        fg=fg,
        relief=tk.FLAT,
        padx=12,
        pady=7,
    ).pack(side=tk.RIGHT, padx=(8, 0))
    tk.Button(
        footer,
        text="Zamknij",
        command=_close_browser,
        cursor="hand2",
        bg=panel_alt,
        fg=fg,
        relief=tk.FLAT,
        padx=12,
        pady=7,
    ).pack(side=tk.RIGHT)

    try:
        browser.protocol("WM_DELETE_WINDOW", _close_browser)
        browser.transient(parent or host.frame)
        browser.grab_set()
    except Exception:
        pass
    try:
        browser.wait_window()
    except Exception:
        pass
    return selected_path["value"]


def pick_yolo_model(host) -> str:
    try:
        return str(_open_detection_yolo_model_candidate_browser(host) or "")
    except Exception as exc:
        logger.debug(f"Nie udało się otworzyć przeglądarki modeli detekcji PZ2: {exc}")
        return _pick_yolo_model_file_dialog(host)


def update_yolo_visibility(host):
    self = host
    if not hasattr(self, "yolo_panel"):
        return

    try:
        self._sync_yolo_model_binding()
    except Exception:
        pass

    method = self._get_detection_method_key()
    has_yolo_model = self._has_configured_yolo_detection_model()
    if method in ("YOLO", "BOTH", "YOLO_OCR", "YOLO_BOX", "YOLO_SYMBOL") and not has_yolo_model:
        method = "OCR"
        try:
            self.detection_method_var.set(method)
        except Exception:
            pass

    is_ocr = method == "OCR"
    is_yolo = method == "YOLO"
    is_yolo_box = method == "YOLO_BOX"
    is_yolo_symbol = method == "YOLO_SYMBOL"
    is_hybrid = method == "BOTH"
    is_yolo_ocr = method == "YOLO_OCR"
    uses_yolo = is_yolo or is_yolo_box or is_yolo_symbol or is_hybrid or is_yolo_ocr

    if hasattr(self, "hybrid_rescue_frame"):
        if is_hybrid or is_yolo:
            self.hybrid_rescue_frame.pack(fill=tk.X, pady=(0, 8))
        else:
            self.hybrid_rescue_frame.pack_forget()

    hybrid_backend_widgets = (
        ("hybrid_box_backend_stage_lbl", {"fill": tk.X, "pady": (0, 4)}),
        ("hybrid_box_backend_stage_divider", {"fill": tk.X, "pady": (0, 8)}),
        ("hybrid_yolo_box_backend_row", {"fill": tk.X, "pady": (0, 4)}),
        ("hybrid_box_backend_info_lbl", {"anchor": tk.W, "fill": tk.X, "pady": (0, 6)}),
    )
    for attr_name, pack_options in hybrid_backend_widgets:
        widget = getattr(self, attr_name, None)
        if widget is None:
            continue
        try:
            if is_hybrid:
                widget.pack(**pack_options)
            else:
                widget.pack_forget()
        except Exception:
            pass

    if not str(self.yolo_panel.winfo_manager()):
        self.yolo_panel.pack(fill=tk.X, pady=(5, 0))

    for attr_name in (
        "yolo_conf_spin",
        "yolo_iou_spin",
        "yolo_overlap_spin",
        "yolo_seq_center_y_scale",
        "yolo_seq_min_h_scale",
        "yolo_seq_max_h_scale",
        "yolo_seq_max_w_scale",
        "yolo_seq_soft_overlap_scale",
        "yolo_seq_hard_overlap_scale",
    ):
        widget = getattr(self, attr_name, None)
        if widget is not None:
            self._set_widget_state(widget, "normal")

    if hasattr(self, "hybrid_rescue_row_info"):
        self.hybrid_rescue_row_info["enabled"] = bool(is_hybrid or is_yolo)
        self._refresh_selection_row(self.hybrid_rescue_row_info)

    if hasattr(self, "hybrid_yolo_box_backend_row_info"):
        self.hybrid_yolo_box_backend_row_info["enabled"] = bool(is_hybrid)
        self._refresh_selection_row(self.hybrid_yolo_box_backend_row_info)

    if hasattr(self, "yolo_agnostic_row_info"):
        self.yolo_agnostic_row_info["enabled"] = True
        self._refresh_selection_row(self.yolo_agnostic_row_info)

    if hasattr(self, "det_yolo_model_entry"):
        self._set_widget_state(self.det_yolo_model_entry, "readonly")

    if hasattr(self, "det_yolo_model_browse_btn"):
        self._set_widget_state(self.det_yolo_model_browse_btn, "normal")

    try:
        self._refresh_yolo_model_picker_state()
    except Exception:
        pass

    if hasattr(self, "btn_ocr_lab"):
        self._set_widget_state(self.btn_ocr_lab, "disabled" if uses_yolo and not is_hybrid and not is_yolo_ocr else "normal")

    if hasattr(self, "btn_rank_presets"):
        self._set_widget_state(self.btn_rank_presets, "normal" if is_ocr else "disabled")

    if hasattr(self, "winner_name_lbl") and hasattr(self, "winner_acc_lbl"):
        if is_ocr:
            self._update_winner_label()
        elif is_yolo or is_yolo_box or is_yolo_symbol:
            self._set_winner_name("Brak rankingu OCR", "neutral")
            self._set_winner_acc("Tryb YOLO nie bierze udziału w turnieju OCR", "muted")
        elif is_yolo_ocr:
            self._set_winner_name("Brak rankingu OCR", "neutral")
            self._set_winner_acc("Tryb YOLO boxy + OCR nie ustala zwycięzcy turnieju OCR", "muted")
        else:
            self._set_winner_name("Brak rankingu OCR", "neutral")
            self._set_winner_acc("Tryb hybrydowy nie ustala zwycięzcy turnieju OCR", "muted")

    if hasattr(self, "test_status_lbl"):
        if is_yolo:
            self._set_test_status(
                self._compose_detection_method_status(
                    "wskaż wytrenowany model znaków .pt i uruchom detekcję"
                ),
                "muted",
            )
        elif is_yolo_box:
            self._set_test_status(
                self._compose_detection_method_status("YB: gotowe do wykrywania samych ramek"),
                "muted",
            )
        elif is_yolo_symbol:
            self._set_test_status(
                self._compose_detection_method_status("YS: gotowe do wpisywania znakow w istniejace ramki"),
                "muted",
            )
        elif is_yolo_ocr:
            self._set_test_status(
                self._compose_detection_method_status(self._get_yolo_box_ocr_status_text()),
                "info",
            )
        elif is_hybrid:
            self._set_test_status(
                self._compose_detection_method_status(self._get_hybrid_detection_status_text()),
                "info",
            )
        else:
            self._set_test_status(
                self._compose_detection_method_status("gotowa do uruchomienia"),
                "success",
            )

    self._refresh_detect_mode_cards()
    self._refresh_detection_workflow_info_label()
    self._refresh_detection_advanced_sections()


def refresh_yolo_model_picker_state(host):
    self = host
    row = getattr(self, "yolo_model_row", None)
    status_lbl = getattr(self, "yolo_model_status_lbl", None)
    browse_btn = getattr(self, "det_yolo_model_browse_btn", None)
    if row is None:
        return

    try:
        self._sync_yolo_model_binding()
    except Exception:
        pass

    project_mode = bool(getattr(self, "_step3_linear_mode", False))
    yolo_model_path = str(self._get_effective_yolo_model_path() or "").strip()
    yolo_ready = bool(yolo_model_path and Path(yolo_model_path).exists())
    model_name = Path(yolo_model_path).name if yolo_ready else ""
    if yolo_ready:
        version, size = self._infer_yolo_arch_from_model_path(yolo_model_path)
        if version and size:
            model_name = f"{model_name} (YOLOv{version}{size})"

    if status_lbl is not None:
        if yolo_ready and project_mode:
            text = (
                f"Model detekcji znaków z projektu: {model_name}. "
                "Używany tylko do wykrywania/proponowania ramek w PZ2, nie do treningu."
            )
            tone = "neutral"
        elif yolo_ready:
            text = (
                f"Model detekcji znaków: {model_name}. "
                "Używany tylko w pipeline OCR/YOLO PZ2; model treningowy wybierasz osobno."
            )
            tone = "neutral"
        elif project_mode:
            text = (
                "Projekt nie ma przypiętego modelu detekcji znaków dla PZ2. "
                "Wskaż wytrenowany .pt, jeśli chcesz użyć pipeline z YOLO."
            )
            tone = "warning"
        else:
            text = (
                "Model detekcji YOLO nie jest jeszcze wybrany. "
                "To model do analizy znaków w PZ2, nie model startowy treningu."
            )
            tone = "muted"
        self._set_themed_label_state(status_lbl, text=text, tone=tone)
    try:
        self._refresh_detection_active_model_label()
    except Exception:
        pass

    if browse_btn is not None:
        try:
            browse_btn.configure(text=("Zmień model detekcji" if yolo_ready else "Wybierz model detekcji"))
        except Exception:
            pass
        try:
            browse_btn.grid_remove()
        except Exception:
            pass
        self._set_widget_state(browse_btn, "disabled")
