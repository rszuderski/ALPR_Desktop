#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wewnętrzna przeglądarka istniejących runów PZ1 dla swobodnego Z3."""

from __future__ import annotations

import datetime as _datetime
import json
import re
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

from PIL import Image, ImageOps, ImageTk


_RUN_TIMESTAMP_RE = re.compile(r"(?P<date>20\d{6})[_-]?(?P<time>\d{6})(?!\d)")
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_free_preview_run_candidates(host) -> list[dict]:
    if bool(getattr(host, "_step3_linear_mode", False)):
        return []

    try:
        root = Path(host._get_step3_chars_root_dir(ensure_exists=False))
    except Exception:
        return []
    if not root.exists() or not root.is_dir():
        return []

    try:
        metadata_paths = list(root.rglob("metadata.json"))
    except Exception:
        metadata_paths = []

    candidates = []
    for metadata_path in metadata_paths:
        run_dir = metadata_path.parent
        images_dir = run_dir / "images"
        try:
            if not images_dir.exists() or not images_dir.is_dir():
                continue
            if not host._is_usable_step3_preview_dir(
                run_dir,
                require_plates=True,
                check_campaign_inflated=False,
            ):
                continue
        except Exception:
            continue

        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(payload, dict) or not payload:
            continue

        plate_count = 0
        with_chars = 0
        gold_count = 0
        excluded_count = 0
        crop_registry_count = 0
        for plate_id, row in payload.items():
            if not str(plate_id or "").strip() or not isinstance(row, dict):
                continue
            plate_count += 1
            chars = row.get("characters")
            if isinstance(chars, list) and chars:
                with_chars += 1
            gold = row.get("gold_state")
            if isinstance(gold, dict):
                excluded = bool(gold.get("excluded", False))
                approved = bool(gold.get("approved", False))
                if excluded:
                    excluded_count += 1
                if approved and not excluded:
                    gold_count += 1
            if (
                str(row.get("crop_id") or "").strip()
                and str(row.get("crop_identity_sha256") or "").strip()
            ):
                crop_registry_count += 1

        if plate_count <= 0:
            continue

        created_at, created_sort = _resolve_run_created_at(run_dir, metadata_path)
        candidates.append({
            "run_dir": run_dir,
            "metadata_path": metadata_path,
            "images_dir": images_dir,
            "metadata": payload,
            "plate_count": plate_count,
            "with_chars": with_chars,
            "gold_count": gold_count,
            "excluded_count": excluded_count,
            "crop_registry_count": crop_registry_count,
            "created_at": created_at,
            "created_sort": created_sort,
        })

    candidates.sort(
        key=lambda item: (
            float(item.get("created_sort", 0.0) or 0.0),
            str(item.get("run_dir") or "").lower(),
        ),
        reverse=True,
    )
    return candidates


def format_preview_run_label(candidate: dict) -> str:
    run_dir = Path(candidate.get("run_dir") or "")
    date_text = str(candidate.get("created_at") or "").strip()
    plates = int(candidate.get("plate_count", 0) or 0)
    chars = int(candidate.get("with_chars", 0) or 0)
    suffix = f"tablice: {plates} | znaki: {chars}/{plates}"
    return f"{run_dir.name} | {date_text} | {suffix}" if date_text else f"{run_dir.name} | {suffix}"


def refresh_preview_run_browser(host, *, force: bool = False) -> list[dict]:
    if bool(getattr(host, "_step3_linear_mode", False)):
        return []
    if not force and bool(getattr(host, "_extract_preview_run_browser_loaded", False)):
        return list(getattr(host, "_extract_preview_run_candidates", []) or [])

    candidates = collect_free_preview_run_candidates(host)
    values = []
    label_map = {}
    for candidate in candidates:
        label = format_preview_run_label(candidate)
        if label in label_map:
            base = label
            ordinal = 2
            while f"{base} [{ordinal}]" in label_map:
                ordinal += 1
            label = f"{base} [{ordinal}]"
        values.append(label)
        label_map[label] = candidate

    host._extract_preview_run_candidates = candidates
    host._extract_preview_run_label_map = label_map
    host._extract_preview_run_browser_loaded = True

    combo = getattr(host, "extract_preview_run_combo", None)
    var = getattr(host, "extract_preview_run_var", None)
    if combo is not None:
        try:
            combo.configure(values=values)
        except Exception:
            pass

    current = str(var.get() or "").strip() if var is not None else ""
    if current not in label_map:
        current = _preferred_label_for_current_preview(host, label_map) or (values[0] if values else "")
        if var is not None:
            try:
                var.set(current)
            except Exception:
                pass

    _render_selected_candidate(host)
    _refresh_action_states(host)
    return candidates


def on_preview_run_selection_changed(host, _event=None) -> None:
    _render_selected_candidate(host)
    _refresh_action_states(host)


def use_selected_preview_run(host) -> bool:
    candidate = _get_selected_candidate(host)
    if not candidate:
        return False
    return use_preview_run_path(host, candidate["run_dir"])


def pick_external_preview_run(host) -> bool:
    try:
        initial_dir = str(host._get_step3_chars_root_dir(ensure_exists=False))
    except Exception:
        initial_dir = ""

    selected = filedialog.askdirectory(
        initialdir=initial_dir or None,
        title="Wskaż run wyodrębnionych tablic (metadata.json + images)",
        parent=getattr(host, "frame", None),
    )
    if not selected:
        return False
    return use_preview_run_path(host, selected)


def use_preview_run_path(host, run_dir) -> bool:
    try:
        candidate = Path(run_dir)
    except Exception:
        candidate = None

    valid = False
    if candidate is not None:
        try:
            valid = bool(host._is_usable_step3_preview_dir(
                candidate,
                require_plates=True,
                check_campaign_inflated=False,
            ))
        except Exception:
            valid = False

    if not valid:
        messagebox.showwarning(
            "Nieprawidłowy run tablic",
            "Wybrany katalog nie jest gotowym runem PZ1.\n\nWymagane są metadata.json, katalog images/ oraz co najmniej jedna tablica.",
            parent=getattr(host, "frame", None),
        )
        return False

    host._source_binding_sync_in_progress = True
    try:
        for var_name in ("annotation_run_dir_var", "xml_path_var", "images_dir_var"):
            var = getattr(host, var_name, None)
            if var is not None:
                var.set("")
        host.preview_dir_var.set(str(candidate))
    finally:
        host._source_binding_sync_in_progress = False

    host._extract_last_source_binding_result = {"ok": False}
    try:
        host._reset_preview_cache()
    except Exception:
        pass
    try:
        host._save_local_setting("char_preview_dir", str(candidate))
    except Exception:
        pass
    try:
        host._force_save_all()
    except Exception:
        pass
    try:
        host._persist_step3_extract_state()
    except Exception:
        pass

    opener = getattr(host, "_open_detection_subtab_with_preview", None)
    if callable(opener):
        try:
            if opener(candidate, force_reload=True):
                return True
        except Exception:
            pass

    messagebox.showerror(
        "Nie otwarto PZ2",
        "Run został rozpoznany, ale nie udało się otworzyć go w PZ2.",
        parent=getattr(host, "frame", None),
    )
    return False


def _preferred_label_for_current_preview(host, label_map: dict[str, dict]) -> str:
    try:
        raw = str(host.preview_dir_var.get() or "").strip()
    except Exception:
        return ""
    if not raw:
        return ""
    current = Path(raw)
    for label, candidate in label_map.items():
        try:
            if Path(candidate["run_dir"]).resolve() == current.resolve():
                return label
        except Exception:
            if str(candidate["run_dir"]) == str(current):
                return label
    return ""


def _get_selected_candidate(host) -> dict | None:
    var = getattr(host, "extract_preview_run_var", None)
    label = str(var.get() or "").strip() if var is not None else ""
    mapping = getattr(host, "_extract_preview_run_label_map", {}) or {}
    candidate = mapping.get(label)
    return candidate if isinstance(candidate, dict) else None


def _refresh_action_states(host) -> None:
    btn = getattr(host, "extract_preview_run_use_btn", None)
    if btn is not None:
        try:
            btn.configure(state=("normal" if _get_selected_candidate(host) else "disabled"))
        except Exception:
            pass


def _render_selected_candidate(host) -> None:
    candidate = _get_selected_candidate(host)
    grid = getattr(host, "extract_preview_run_grid", None)
    summary_var = getattr(host, "extract_preview_run_summary_var", None)
    if grid is None:
        return

    try:
        for child in list(grid.winfo_children()):
            child.destroy()
    except Exception:
        pass
    host._extract_preview_run_photo_refs = []

    if not candidate:
        if summary_var is not None:
            try:
                summary_var.set("Brak lokalnych runów PZ1. Możesz użyć źródła spoza listy albo XML + obrazy.")
            except Exception:
                pass
        return

    plates = int(candidate.get("plate_count", 0) or 0)
    with_chars = int(candidate.get("with_chars", 0) or 0)
    gold = int(candidate.get("gold_count", 0) or 0)
    excluded = int(candidate.get("excluded_count", 0) or 0)
    registered = int(candidate.get("crop_registry_count", 0) or 0)
    if summary_var is not None:
        try:
            summary_var.set(
                f"Tablice: {plates}  •  z anotacjami znaków: {with_chars}  •  GOLD: {gold}  •  N: {excluded}  •  ID cropu: {registered}/{plates}"
            )
        except Exception:
            pass

    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    plate_ids = [str(pid) for pid, row in metadata.items() if str(pid or "").strip() and isinstance(row, dict)]
    sample_ids = _select_evenly_spaced(plate_ids, limit=8)

    palette = getattr(getattr(host, "app", None), "palette", {}) or {}
    field = palette.get("field", "#1a1a1a")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    refs = []
    columns = 2
    for col in range(columns):
        try:
            grid.grid_columnconfigure(col, weight=1, uniform="pz1run")
        except Exception:
            pass

    for index, pid in enumerate(sample_ids):
        row_data = metadata.get(pid) if isinstance(metadata.get(pid), dict) else {}
        card = tk.Frame(grid, bg=field, bd=0, highlightthickness=1, highlightbackground=border, highlightcolor=border)
        card.grid(row=index // columns, column=index % columns, sticky="nsew", padx=4, pady=4)

        image_path = _find_plate_image(candidate["images_dir"], pid)
        photo = None
        if image_path is not None:
            try:
                with Image.open(image_path) as source:
                    image = _prepare_preview_image(
                        source,
                        max_size=(520, 180),
                    )
                photo = ImageTk.PhotoImage(image)
                refs.append(photo)
            except Exception:
                photo = None

        if photo is not None:
            tk.Label(card, image=photo, bg=field, bd=0).pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 2))
        else:
            tk.Label(card, text="Brak obrazu", bg=field, fg=muted).pack(fill=tk.BOTH, expand=True, padx=4, pady=(8, 2))

        chars = row_data.get("characters")
        char_count = len(chars) if isinstance(chars, list) else 0
        status = str(row_data.get("status") or "").strip().lower() or "—"
        tk.Label(
            card,
            text=f"{pid}  •  znaki: {char_count}  •  status: {status}",
            bg=field,
            fg=fg,
            anchor="w",
            justify=tk.LEFT,
            wraplength=500,
            font=("Segoe UI", 8),
        ).pack(fill=tk.X, padx=8, pady=(3, 7))

    host._extract_preview_run_photo_refs = refs



def _prepare_preview_image(source, *, max_size=(520, 180)):
    """Zachowaj cały crop tablicy i dodaj tylko wizualny margines."""
    image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        return image

    pad_x = max(10, int(round(width * 0.045)))
    pad_y = max(8, int(round(height * 0.12)))
    try:
        fill = image.getpixel((0, 0))
    except Exception:
        fill = (32, 32, 32)

    image = ImageOps.expand(
        image,
        border=(pad_x, pad_y, pad_x, pad_y),
        fill=fill,
    )
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def _find_plate_image(images_dir, plate_id: str) -> Path | None:
    root = Path(images_dir)
    direct = root / plate_id
    if direct.exists() and direct.is_file():
        return direct
    for suffix in _IMAGE_SUFFIXES:
        candidate = root / f"{plate_id}{suffix}"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _select_evenly_spaced(items, limit=8):
    values = list(items or [])
    limit = max(1, int(limit or 8))
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    indices = []
    for slot in range(limit):
        idx = int(round(slot * last / (limit - 1)))
        if idx not in indices:
            indices.append(idx)
    return [values[idx] for idx in indices]


def _resolve_run_created_at(run_dir: Path, metadata_path: Path) -> tuple[str, float]:
    match = _RUN_TIMESTAMP_RE.search(run_dir.name)
    if match:
        try:
            dt = _datetime.datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S")
            return dt.strftime("%d.%m.%Y %H:%M"), dt.timestamp()
        except Exception:
            pass

    manifest_path = run_dir / "extract_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except Exception:
            manifest = {}
        if isinstance(manifest, dict):
            for key in ("created_at", "generated_at", "completed_at"):
                raw = str(manifest.get(key) or "").strip()
                if not raw:
                    continue
                try:
                    dt = _datetime.datetime.fromisoformat(raw)
                    return dt.strftime("%d.%m.%Y %H:%M"), dt.timestamp()
                except Exception:
                    continue

    try:
        stamp = float(metadata_path.stat().st_mtime)
        dt = _datetime.datetime.fromtimestamp(stamp)
        return f"{dt.strftime('%d.%m.%Y %H:%M')}*", stamp
    except Exception:
        return "", 0.0
