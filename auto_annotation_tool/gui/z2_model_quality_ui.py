#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 model-quality formatting and table rendering helpers."""

import math
import tkinter as tk
from pathlib import Path

from ..validators import format_yolo_model_identity, validate_model_file
from .z3_model_metadata_dialog import show_yolo_model_metadata_dialog
from .web_slim_scrollbar import blend_hex_colors


def get_model_identity_caption(model_path: Path | None) -> str:
    if model_path is None or not model_path.exists():
        return ""
    try:
        _ok, _message, info = validate_model_file(model_path, allow_heavy_load=False)
    except Exception:
        info = {}
    return format_yolo_model_identity(info)


def model_quality_float(value) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        numeric = float(str(value).strip().replace(",", "."))
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def model_quality_value(mapping: dict | None, keys: list[str] | tuple[str, ...]) -> float | None:
    if not isinstance(mapping, dict):
        return None

    for key in keys:
        if key in mapping:
            value = model_quality_float(mapping.get(key))
            if value is not None:
                return value

    lower_map = {str(k or "").strip().lower(): v for k, v in mapping.items()}
    for key in keys:
        value = model_quality_float(lower_map.get(str(key or "").strip().lower()))
        if value is not None:
            return value
    return None


def format_model_quality_number(value) -> str:
    numeric = model_quality_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.3f}"


def format_model_quality_epoch(value) -> str:
    numeric = model_quality_float(value)
    if numeric is None:
        return "-"
    return str(max(0, int(round(numeric))))


def describe_auto_model_quality(value) -> str:
    numeric = model_quality_float(value)
    if numeric is None:
        return ""
    if numeric >= 0.80:
        return "bardzo dobry poziom"
    if numeric >= 0.60:
        return "dobry poziom"
    if numeric >= 0.40:
        return "poziom używalny, ale do kontroli"
    return "niski poziom - wybierz ostrożnie"


def auto_model_metric_tone(value, profile: str = "strict") -> str:
    numeric = model_quality_float(value)
    if numeric is None:
        return "info"
    normalized = str(profile or "").strip().lower()
    if normalized == "loose":
        return "success" if numeric >= 0.70 else "warning"
    if normalized == "standard":
        return "success" if numeric >= 0.50 else "warning"
    return "success" if numeric >= 0.60 else "warning"


def shorten_model_quality_text(value, limit: int = 72) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= int(limit):
        return text
    return text[: max(1, int(limit) - 3)] + "..."


def _is_metrics_source_row(label_text: str) -> bool:
    normalized = str(label_text or "").strip().casefold()
    if not normalized:
        return False
    asciiish = (
        normalized.replace("ź", "z")
        .replace("ż", "z")
        .replace("ó", "o")
        .replace("ł", "l")
        .replace("ą", "a")
        .replace("ę", "e")
        .replace("ś", "s")
        .replace("ć", "c")
        .replace("ń", "n")
    )
    return (
        "metryk" in normalized
        and (
            "źród" in normalized
            or "zrod" in asciiish
            or "source" in normalized
            or "�" in normalized
        )
    )


def render_auto_annotation_model_quality_table(
    owner,
    host,
    rows: list[tuple[str, str, str]],
    tone: str,
    *,
    bg: str | None = None,
    fg: str | None = None,
    muted: str | None = None,
    border: str | None = None,
    success: str | None = None,
    warning: str | None = None,
    wraplength: int = 390,
    label_width: int = 15,
    model_path: Path | str | None = None,
) -> None:
    if host is None:
        return
    for child in list(host.winfo_children()):
        try:
            child.destroy()
        except Exception:
            pass

    safe_rows = [
        (str(label or "").strip(), str(value or "").strip(), str(row_tone or "").strip().lower())
        for label, value, row_tone in list(rows or [])
        if str(label or "").strip() and str(value or "").strip()
    ]
    if not safe_rows:
        return

    palette = getattr(getattr(owner, "app", None), "palette", {}) or {}
    table_bg = str(bg or palette.get("field", palette.get("panel_alt", "#2d2d30")))
    table_fg = str(fg or palette.get("fg", "#f3f3f3"))
    table_muted = str(muted or palette.get("muted", "#c7c7c7"))
    table_border = str(border or palette.get("panel_border", palette.get("border", "#3c3c3c")))
    success_color = str(success or palette.get("success", "#2ecc71"))
    warning_color = str(warning or palette.get("warning", "#f39c12"))
    table_tone = str(tone or "").strip().lower()
    accent = success_color if table_tone == "success" else warning_color
    row_alt = blend_hex_colors(table_border, table_bg, 0.18)

    shell = tk.Frame(
        host,
        bg=table_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=accent,
        highlightcolor=accent,
    )
    shell.pack(fill=tk.X)

    details_model_path = None
    try:
        details_model_path = Path(str(model_path).strip()) if str(model_path or "").strip() else None
    except Exception:
        details_model_path = None

    for idx, (label_text, value_text, row_tone) in enumerate(safe_rows):
        row_bg = table_bg if idx % 2 == 0 else row_alt
        row = tk.Frame(shell, bg=row_bg, bd=0, highlightthickness=0)
        row.pack(fill=tk.X, padx=1, pady=(1 if idx == 0 else 0, 1))

        tk.Label(
            row,
            text=label_text,
            bg=row_bg,
            fg=table_muted,
            font=("Segoe UI", 8),
            anchor="w",
            justify=tk.LEFT,
            width=max(8, int(label_width)),
            padx=7,
            pady=3,
        ).pack(side=tk.LEFT, fill=tk.Y)

        value_fg = (
            success_color
            if row_tone == "success"
            else warning_color
            if row_tone == "warning"
            else table_fg
        )
        value_host = tk.Frame(row, bg=row_bg, bd=0, highlightthickness=0)
        value_host.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            value_host,
            text=value_text,
            bg=row_bg,
            fg=value_fg,
            font=("Segoe UI", 8, "bold" if row_tone in {"success", "warning"} else "normal"),
            anchor="w",
            justify=tk.LEFT,
            wraplength=max(180, int(wraplength) - 95 if _is_metrics_source_row(label_text) else int(wraplength)),
            padx=7,
            pady=3,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        if _is_metrics_source_row(label_text):
            tk.Button(
                value_host,
                text="Metadane",
                bg=row_bg,
                fg=success_color,
                activebackground=row_bg,
                activeforeground=success_color,
                relief=tk.FLAT,
                bd=0,
                highlightthickness=0,
                cursor="hand2",
                font=("Segoe UI", 8, "bold"),
                padx=6,
                pady=1,
                command=lambda path=details_model_path: show_yolo_model_metadata_dialog(
                    owner,
                    str(path or ""),
                    title="Metaparametry modelu",
                ),
            ).pack(side=tk.RIGHT, padx=(4, 7), pady=1)
