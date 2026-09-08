#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Desktop browser for Android mobile ALPR benchmark reports."""

from __future__ import annotations

import json
import hashlib
import statistics
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import ttk, messagebox

from ..config import CONFIG, logger
from ..ranking import (
    MobileBenchmarkReport,
    MobilePackageExperimentStore,
    MobileReportBundle,
    read_mobile_report_bundles,
    score_mobile_report,
)
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors

MOBILE_REPORT_RAW_PREVIEW_LIST_LIMIT = 24
MOBILE_REPORT_RAW_STORAGE_LIST_LIMIT = 8
MOBILE_REPORT_RAW_PREVIEW_TEXT_LIMIT = 3600
MOBILE_REPORT_RAW_STORAGE_TEXT_LIMIT = 1200
MOBILE_REPORT_RAW_TEXT_LIMIT = 160_000

_HEAVY_MOBILE_REPORT_RAW_KEYS = {
    "application_log",
    "annotations",
    "crops",
    "event_stream",
    "events",
    "frame_flow",
    "images",
    "log",
    "records",
    "samples",
    "thermal",
    "thermal_samples",
    "thermal_trace",
    "trace",
    "traces",
}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    parsed = _safe_float(value, None)
    if parsed is None:
        return default
    try:
        return int(parsed)
    except Exception:
        return default


def _nested_value(data: dict[str, Any] | None, *paths: str) -> Any:
    if not isinstance(data, dict):
        return None
    for path in paths:
        current: Any = data
        ok = True
        for part in str(path or "").split("."):
            if isinstance(current, dict) and part in current:
                current = current.get(part)
            else:
                ok = False
                break
        if ok and current not in (None, ""):
            return current
    return None


_MODEL_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "mp": ("mp", "MP", "vehicle", "vehicles", "Vehicle", "VEHICLE"),
    "mt": ("mt", "MT", "plate", "plates", "Plate", "PLATE"),
    "mz": ("mz", "MZ", "character", "characters", "char", "chars", "Character", "CHARACTER"),
}


def _merge_model_fingerprint_aliases(target: dict[str, Any], value: Any) -> None:
    if not isinstance(value, dict):
        return
    for nested_key in ("model_fingerprints", "models"):
        nested = value.get(nested_key)
        if isinstance(nested, dict):
            _merge_model_fingerprint_aliases(target, nested)
    for canonical, aliases in _MODEL_ROLE_ALIASES.items():
        for alias in aliases:
            if alias not in value:
                continue
            role_value = value.get(alias)
            if role_value not in (None, "", {}, []) and canonical not in target:
                target[canonical] = role_value
                break


def _format_datetime(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    text = text.replace("T", " ")
    text = text.replace("+00:00", " UTC").replace("Z", " UTC")
    if "." in text:
        text = text.split(".", 1)[0]
    return text


def _format_number(value: Any, *, digits: int = 2) -> str:
    parsed = _safe_float(value, None)
    if parsed is None:
        return "-" if value in (None, "") else str(value)
    if abs(parsed - int(parsed)) < 0.000001:
        return str(int(parsed))
    return f"{parsed:.{digits}f}".rstrip("0").rstrip(".")


def _format_percent(value: Any) -> str:
    parsed = _safe_float(value, None)
    if parsed is None:
        return "-"
    if parsed <= 1.0:
        parsed *= 100.0
    return f"{parsed:.1f}%"


def _format_ms(value: Any) -> str:
    parsed = _safe_float(value, None)
    if parsed is None:
        return "-"
    return f"{parsed:.1f} ms"


def _format_mb(value: Any) -> str:
    parsed = _safe_float(value, None)
    if parsed is None:
        return "-"
    return f"{parsed:.1f} MB"


def _short_text(value: Any, limit: int = 120) -> str:
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
    else:
        text = str(value or "")
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text or "-"


def _compact_mobile_report_payload(
    value: Any,
    *,
    list_limit: int = MOBILE_REPORT_RAW_PREVIEW_LIST_LIMIT,
    text_limit: int = MOBILE_REPORT_RAW_PREVIEW_TEXT_LIMIT,
    depth: int = 0,
) -> Any:
    """Keep report previews useful without feeding huge payloads into Tk widgets."""
    if depth > 6:
        return _short_text(value, min(480, max(80, int(text_limit))))
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key or "")
            lower_key = key_text.lower()
            if lower_key in _HEAVY_MOBILE_REPORT_RAW_KEYS:
                result[key_text] = _summarize_mobile_report_heavy_value(
                    child,
                    list_limit=list_limit,
                    text_limit=text_limit,
                    depth=depth + 1,
                )
            else:
                result[key_text] = _compact_mobile_report_payload(
                    child,
                    list_limit=list_limit,
                    text_limit=text_limit,
                    depth=depth + 1,
                )
        return result
    if isinstance(value, (list, tuple)):
        values = list(value)
        preview = [
            _compact_mobile_report_payload(
                item,
                list_limit=list_limit,
                text_limit=text_limit,
                depth=depth + 1,
            )
            for item in values[:list_limit]
        ]
        if len(values) <= list_limit:
            return preview
        return {
            "_preview_items": preview,
            "_total_items": len(values),
            "_omitted_items": len(values) - len(preview),
        }
    if isinstance(value, str) and len(value) > text_limit:
        return {
            "_preview_text": value[:text_limit],
            "_total_chars": len(value),
            "_omitted_chars": len(value) - text_limit,
        }
    return value


def _summarize_mobile_report_heavy_value(
    value: Any,
    *,
    list_limit: int,
    text_limit: int,
    depth: int,
) -> dict[str, Any]:
    if isinstance(value, dict):
        records = None
        for key in ("records", "rows", "samples", "events", "traces", "data"):
            if isinstance(value.get(key), list):
                records = value.get(key)
                break
        if records is not None:
            return {
                "_kind": "records",
                "_total_items": len(records),
                "_preview_items": [
                    _compact_mobile_report_payload(
                        item,
                        list_limit=list_limit,
                        text_limit=text_limit,
                        depth=depth + 1,
                    )
                    for item in records[:list_limit]
                ],
            }
        return {
            "_kind": "object",
            "_preview": _compact_mobile_report_payload(
                value,
                list_limit=list_limit,
                text_limit=text_limit,
                depth=depth + 1,
            ),
        }
    if isinstance(value, (list, tuple)):
        values = list(value)
        return {
            "_kind": "list",
            "_total_items": len(values),
            "_preview_items": [
                _compact_mobile_report_payload(
                    item,
                    list_limit=list_limit,
                    text_limit=text_limit,
                    depth=depth + 1,
                )
                for item in values[:list_limit]
            ],
            "_omitted_items": max(0, len(values) - min(len(values), list_limit)),
        }
    if isinstance(value, str):
        return {
            "_kind": "text",
            "_preview_text": value[:text_limit],
            "_total_chars": len(value),
            "_omitted_chars": max(0, len(value) - text_limit),
        }
    return {"_kind": type(value).__name__, "_value": _compact_mobile_report_payload(value, list_limit=list_limit, text_limit=text_limit, depth=depth + 1)}


def _compact_mobile_report_for_store(report: MobileBenchmarkReport) -> MobileBenchmarkReport:
    try:
        compact_raw = _compact_mobile_report_payload(
            report.raw,
            list_limit=MOBILE_REPORT_RAW_STORAGE_LIST_LIMIT,
            text_limit=MOBILE_REPORT_RAW_STORAGE_TEXT_LIMIT,
        )
        return replace(report, raw=compact_raw)
    except Exception:
        return report


def _mobile_report_preview_dict(report: MobileBenchmarkReport) -> dict[str, Any]:
    data = report.to_dict()
    data["raw"] = _compact_mobile_report_payload(
        getattr(report, "raw", {}) or {},
        list_limit=MOBILE_REPORT_RAW_PREVIEW_LIST_LIMIT,
        text_limit=MOBILE_REPORT_RAW_PREVIEW_TEXT_LIMIT,
    )
    return data


def _mobile_report_bundle_preview_dict(bundle: MobileReportBundle | None) -> dict[str, Any] | None:
    if bundle is None:
        return None

    def _rows_preview(rows: tuple[dict[str, str], ...], total: int, columns: tuple[str, ...]) -> dict[str, Any]:
        row_list = list(rows or ())
        return {
            "total": int(total or len(row_list)),
            "preview_rows": row_list[:MOBILE_REPORT_RAW_PREVIEW_LIST_LIMIT],
            "preview_count": min(len(row_list), MOBILE_REPORT_RAW_PREVIEW_LIST_LIMIT),
            "columns": list(columns or ()),
        }

    try:
        validation = bundle.validation.to_dict()
    except Exception:
        validation = {}
    entries = list(bundle.entries or ())
    return {
        "path": bundle.path,
        "source_archive_sha256": bundle.source_archive_sha256,
        "bundle_kind": bundle.bundle_kind,
        "bundle_schema": bundle.bundle_schema,
        "validation": validation,
        "experiment_session": _compact_mobile_report_payload(bundle.experiment_session or {}),
        "manifest": _compact_mobile_report_payload(bundle.manifest or {}),
        "metadata": _compact_mobile_report_payload(bundle.metadata or {}),
        "pipeline_manifests": _compact_mobile_report_payload(getattr(bundle, "pipeline_manifests", {}) or {}),
        "model_refs": _compact_mobile_report_payload(getattr(bundle, "model_refs", {}) or {}),
        "sample_schema": dict(bundle.sample_schema),
        "collection_session": _compact_mobile_report_payload(bundle.collection_session),
        "report_payload_preview": _compact_mobile_report_payload(bundle.report_payload or {}),
        "artifacts": {
            "traces": _rows_preview(bundle.trace_rows, bundle.trace_total, bundle.trace_columns),
            "thermal": _rows_preview(bundle.thermal_rows, bundle.thermal_total, bundle.thermal_columns),
            "frame_flow": _rows_preview(bundle.frame_flow_rows, bundle.frame_flow_total, bundle.frame_flow_columns),
            "events": _rows_preview(bundle.event_rows, bundle.event_total, bundle.event_columns),
            "samples": {
                "total": int(bundle.sample_total or len(bundle.sample_rows or ())),
                "preview_rows": list(bundle.sample_rows or ())[:MOBILE_REPORT_RAW_PREVIEW_LIST_LIMIT],
                "preview_count": min(len(bundle.sample_rows or ()), MOBILE_REPORT_RAW_PREVIEW_LIST_LIMIT),
            },
            "crops": int(bundle.crop_count or 0),
            "annotations": int(bundle.annotation_count or 0),
            "attempts": _rows_preview(bundle.attempt_rows, bundle.attempt_total, tuple(bundle.attempt_rows[0]) if bundle.attempt_rows else ()),
            "log_preview_chars": len(bundle.log_preview or ""),
        },
        "archive_entries": {
            "total": len(entries),
            "preview": [entry.to_dict() for entry in entries[:80]],
            "omitted": max(0, len(entries) - 80),
        },
    }


def _flatten_rows(data: Any, *, prefix: str = "", limit: int = 90) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    def walk(value: Any, key_prefix: str) -> None:
        if len(rows) >= limit:
            return
        if isinstance(value, dict):
            if not value:
                rows.append((key_prefix or "-", "-", ""))
                return
            for key, child in value.items():
                name = f"{key_prefix}.{key}" if key_prefix else str(key)
                walk(child, name)
        elif isinstance(value, list):
            rows.append((key_prefix or "-", f"lista: {len(value)}", _short_text(value[:3], 180)))
        else:
            rows.append((key_prefix or "-", _short_text(value, 240), ""))

    walk(data, prefix)
    if len(rows) >= limit:
        rows.append(("...", "ucięto podgląd", f"Pokazano pierwsze {limit} pól. Pełny JSON jest w zakładce Surowe dane."))
    return rows


def _device_label(report: MobileBenchmarkReport) -> str:
    device = dict(report.device or {})
    pieces = [
        str(device.get("manufacturer") or "").strip(),
        str(device.get("model") or device.get("name") or device.get("device_name") or "").strip(),
    ]
    label = " ".join(part for part in pieces if part).strip()
    return label or str(device.get("name") or "-")


def _pipeline_p95(report: MobileBenchmarkReport) -> Any:
    return _nested_value(
        report.latency,
        "pipeline.p95_ms",
        "pipeline.latency_pipeline_ms_p95",
        "pipeline_ms_p95",
        "latency_pipeline_ms_p95",
        "p95_ms",
    )


def _quality_value(report: MobileBenchmarkReport) -> str:
    quality = dict(report.quality or {})
    if quality.get("available") is False:
        return "brak GT"
    exact = _nested_value(quality, "exact_match_rate", "plate_exact_match", "plate_accuracy", "accuracy_plate")
    if exact is not None:
        return _format_percent(exact)
    cer = _nested_value(quality, "cer", "character_error_rate")
    if cer is not None:
        return f"CER {_format_percent(cer)}"
    return "-"


def _report_list_values(report: MobileBenchmarkReport) -> tuple[str, str, str, str, str, str, str]:
    try:
        score = score_mobile_report(report)
        score_text = "-" if score.rejected else f"{score.total:.3f}"
    except Exception:
        score_text = "-"
    return (
        _format_datetime(report.measured_at),
        str(report.package_id or "-"),
        str(report.variant_id or "-"),
        _device_label(report),
        _format_ms(_pipeline_p95(report)),
        _quality_value(report),
        score_text,
    )


class _TreeTable:
    def __init__(
        self,
        parent,
        columns: tuple[str, ...],
        headings: tuple[str, ...],
        widths: tuple[int, ...],
        *,
        palette: dict[str, str],
        height: int = 10,
        stretch_last: bool = True,
    ):
        self.shell = tk.Frame(parent, bg=palette.get("panel", "#252526"))
        self.shell.grid_columnconfigure(0, weight=1)
        self.shell.grid_rowconfigure(0, weight=1)
        style_name = f"MobileReport.{id(self)}.Treeview"
        try:
            style = ttk.Style(self.shell)
            style.configure(
                style_name,
                background=palette.get("panel", "#252526"),
                fieldbackground=palette.get("panel", "#252526"),
                foreground=palette.get("fg", "#f3f3f3"),
                rowheight=25,
                borderwidth=0,
            )
            style.map(
                style_name,
                background=[("selected", blend_hex_colors(palette.get("accent", "#4f8de3"), palette.get("panel", "#252526"), 0.25))],
                foreground=[("selected", "#ffffff")],
            )
        except Exception:
            pass
        self.tree = ttk.Treeview(
            self.shell,
            columns=columns,
            show="headings",
            height=height,
            selectmode="browse",
            style=style_name,
        )
        for index, column in enumerate(columns):
            self.tree.heading(column, text=headings[index] if index < len(headings) else column)
            self.tree.column(
                column,
                width=widths[index] if index < len(widths) else 120,
                minwidth=44,
                stretch=bool(stretch_last and index == len(columns) - 1),
                anchor=tk.W,
            )
        scroll = WebSlimScrollbar(
            self.shell,
            orient=tk.VERTICAL,
            command=self.tree.yview,
            track_color=palette.get("panel", "#252526"),
            thumb_color=blend_hex_colors(palette.get("accent", "#4f8de3"), palette.get("fg", "#f3f3f3"), 0.25),
        )
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        for tag, color in (
            ("success", palette.get("success", "#2ecc71")),
            ("warning", palette.get("warning", "#f1c40f")),
            ("error", palette.get("error", "#e74c3c")),
            ("muted", palette.get("muted", "#c7c7c7")),
        ):
            try:
                self.tree.tag_configure(tag, foreground=color)
            except Exception:
                pass

    def set_rows(self, rows: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for index, row in enumerate(rows or []):
            tag = ""
            values = row
            if row and isinstance(row[-1], dict):
                options = row[-1]
                values = row[:-1]
                tag = str(options.get("tag") or "")
            self.tree.insert("", tk.END, iid=f"row_{index}", values=tuple(str(value) for value in values), tags=(tag,) if tag else ())


class _WrappedLabelTable:
    def __init__(
        self,
        parent,
        columns: tuple[str, ...],
        headings: tuple[str, ...],
        widths: tuple[int, ...],
        *,
        palette: dict[str, str],
        height: int = 10,
    ):
        self._columns = tuple(columns)
        self._widths = tuple(max(1, int(width)) for width in widths)
        self._total_width = max(1, sum(self._widths))
        self._palette = dict(palette or {})
        self._panel = self._palette.get("panel", "#252526")
        self._bg = self._palette.get("bg", self._panel)
        self._fg = self._palette.get("fg", "#f3f3f3")
        self._muted = self._palette.get("muted", "#c7c7c7")
        self._border = self._palette.get("border", "#4a4a4a")
        self._accent = self._palette.get("accent", "#4f8de3")
        self._row_labels: list[tk.Label] = []

        self.shell = tk.Frame(parent, bg=self._panel)
        self.shell.grid_columnconfigure(0, weight=1)
        self.shell.grid_rowconfigure(1, weight=1)

        header_bg = blend_hex_colors(self._panel, self._accent, 0.08)
        self._header = tk.Frame(self.shell, bg=header_bg)
        self._header.grid(row=0, column=0, sticky="ew", pady=(0, 3))
        self._header_labels: list[tk.Label] = []

        for index, column in enumerate(self._columns):
            weight = self._widths[index] if index < len(self._widths) else 1
            self._header.grid_columnconfigure(index, weight=weight, uniform="mobile_report_wrapped")
            label = tk.Label(
                self._header,
                text=headings[index] if index < len(headings) else column,
                bg=header_bg,
                fg=self._fg,
                font=("Segoe UI", 8, "bold"),
                padx=8,
                pady=6,
                anchor=tk.W,
                justify=tk.LEFT,
            )
            label.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 1, 0))
            self._header_labels.append(label)

        self._canvas = tk.Canvas(
            self.shell,
            bg=self._panel,
            highlightthickness=1,
            highlightbackground=blend_hex_colors(self._border, self._panel, 0.28),
            bd=0,
            height=max(120, int(height) * 30),
        )
        self._scroll = WebSlimScrollbar(
            self.shell,
            orient=tk.VERTICAL,
            command=self._canvas.yview,
            track_color=self._palette.get("scrollbar_track", self._panel),
            thumb_color=self._palette.get("scrollbar_thumb", self._accent),
            thumb_hover_color=self._palette.get("scrollbar_thumb_hover", self._palette.get("accent_hover", self._accent)),
        )
        self._canvas.configure(yscrollcommand=self._scroll.set)
        self._canvas.grid(row=1, column=0, sticky="nsew")
        self._scroll.grid(row=1, column=1, sticky="ns")

        self._content = tk.Frame(self._canvas, bg=self._panel)
        self._window_id = self._canvas.create_window((0, 0), window=self._content, anchor=tk.NW)
        for index, width in enumerate(self._widths):
            self._content.grid_columnconfigure(index, weight=width, uniform="mobile_report_wrapped")

        self._canvas.bind("<Configure>", self._sync_width, add="+")
        self._content.bind("<Configure>", self._sync_scrollregion, add="+")
        self._bind_mousewheel(self._canvas)
        self._bind_mousewheel(self._content)

    def _tag_color(self, tag: str) -> str:
        return {
            "success": self._palette.get("success", "#2ecc71"),
            "warning": self._palette.get("warning", "#f1c40f"),
            "error": self._palette.get("error", "#e74c3c"),
            "muted": self._palette.get("muted", "#c7c7c7"),
        }.get(str(tag or ""), self._fg)

    def _bind_mousewheel(self, widget) -> None:
        try:
            widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_mousewheel, add="+")
            widget.bind("<Button-5>", self._on_mousewheel, add="+")
        except Exception:
            pass

    def _on_mousewheel(self, event):
        try:
            if getattr(event, "num", None) == 4:
                units = -3
            elif getattr(event, "num", None) == 5:
                units = 3
            else:
                units = -max(-6, min(6, int(getattr(event, "delta", 0) / 120))) or 0
            if units:
                self._canvas.yview_scroll(units, "units")
        except Exception:
            pass
        return "break"

    def _sync_scrollregion(self, _event=None) -> None:
        try:
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        except Exception:
            pass

    def _column_wraplengths(self, width: int) -> list[int]:
        content_width = max(120, int(width or self._canvas.winfo_width() or 1))
        return [
            max(42, int((content_width * (column_width / self._total_width)) - 18))
            for column_width in self._widths
        ]

    def _sync_width(self, event=None) -> None:
        try:
            width = max(1, int(getattr(event, "width", 0) or self._canvas.winfo_width() or 1))
            self._canvas.itemconfigure(self._window_id, width=width)
            wraplengths = self._column_wraplengths(width)
            for index, label in enumerate(self._header_labels):
                label.configure(wraplength=wraplengths[index] if index < len(wraplengths) else 90)
            for index, label in enumerate(self._row_labels):
                column_index = int(getattr(label, "_mobile_report_column_index", 0) or 0)
                label.configure(wraplength=wraplengths[column_index] if column_index < len(wraplengths) else 90)
            self._sync_scrollregion()
            self._canvas.after_idle(self._sync_scrollregion)
        except Exception:
            pass

    def set_rows(self, rows: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...]) -> None:
        for child in self._content.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass
        self._row_labels = []

        for row_index, row in enumerate(rows or []):
            tag = ""
            values = row
            if row and isinstance(row[-1], dict):
                options = row[-1]
                values = row[:-1]
                tag = str(options.get("tag") or "")
            row_bg = self._panel if row_index % 2 == 0 else blend_hex_colors(self._panel, self._bg, 0.28)
            fg = self._tag_color(tag)
            for column_index, _column in enumerate(self._columns):
                value = values[column_index] if column_index < len(values) else ""
                label = tk.Label(
                    self._content,
                    text=str(value if value not in (None, "") else "-"),
                    bg=row_bg,
                    fg=fg,
                    font=("Segoe UI", 8),
                    padx=8,
                    pady=7,
                    anchor=tk.NW,
                    justify=tk.LEFT,
                    relief=tk.FLAT,
                )
                setattr(label, "_mobile_report_column_index", column_index)
                label.grid(row=row_index, column=column_index, sticky="nsew", padx=(0 if column_index == 0 else 1, 0), pady=(0, 1))
                self._bind_mousewheel(label)
                self._row_labels.append(label)
        self._sync_width()
        self._canvas.after_idle(self._sync_scrollregion)


class MobileReportBrowser:
    def __init__(self, owner, parent=None):
        self.owner = owner
        self.parent = parent or getattr(getattr(owner, "app", None), "root", None) or getattr(owner, "frame", None)
        self.palette = dict(getattr(getattr(owner, "app", None), "palette", {}) or {})
        self.bg = self.palette.get("bg", self.palette.get("panel", "#252526"))
        self.panel = self.palette.get("panel", "#252526")
        self.fg = self.palette.get("fg", "#f3f3f3")
        self.muted = self.palette.get("muted", "#c7c7c7")
        self.accent = self.palette.get("accent", "#4f8de3")
        self.success = self.palette.get("success", "#2ecc71")
        self.warning = self.palette.get("warning", "#f1c40f")
        self.error = self.palette.get("error", "#e74c3c")
        self.card_bg = blend_hex_colors(self.panel, self.accent, 0.045)
        self.border = blend_hex_colors(self.accent, self.panel, 0.38)
        self.store = MobilePackageExperimentStore()
        self.report_by_iid: dict[str, MobileBenchmarkReport] = {}
        self.bundle_by_report_id: dict[str, MobileReportBundle] = {}
        self.current_bundle: MobileReportBundle | None = None
        self._closing = False
        self._import_in_progress = False
        self._import_cancel_requested = False
        self._suppress_report_select = False
        self.window = tk.Toplevel(self.parent)
        try:
            setattr(self.window, "_aat_skip_window_recovery", True)
        except Exception:
            pass
        try:
            setattr(owner, "_mobile_report_browser_dialog", self.window)
        except Exception:
            pass
        self.window.withdraw()
        self.window.configure(bg=self.bg)
        self.window.title("Raporty z telefonu")
        self._configure_window()
        self._build()
        self._show_empty()
        try:
            self.window.deiconify()
            self.window.lift()
            self.window.focus_force()
        except Exception:
            pass
        try:
            self.window.after(80, lambda: self._refresh_report_list(auto_select=False))
        except Exception:
            self._refresh_report_list(auto_select=False)

    def _configure_window(self) -> None:
        try:
            screen_w = int(self.window.winfo_screenwidth() or 1360)
            screen_h = int(self.window.winfo_screenheight() or 840)
        except Exception:
            screen_w, screen_h = 1360, 840
        width = min(1240, max(1040, screen_w - 130))
        height = min(760, max(620, screen_h - 130))
        x = max(24, int((screen_w - width) / 2))
        y = max(24, int((screen_h - height) / 2))
        try:
            self.window.geometry(f"{width}x{height}+{x}+{y}")
            self.window.minsize(980, 600)
            self.window.resizable(True, True)
        except Exception:
            pass
        try:
            self.window.grab_release()
        except Exception:
            pass
        try:
            self.window.wm_transient("")
        except Exception:
            pass
        try:
            self.window.bind("<Destroy>", self._on_destroy, add="+")
        except Exception:
            pass
        try:
            self.window.protocol("WM_DELETE_WINDOW", self._request_close)
        except Exception:
            pass

    def _on_destroy(self, event=None) -> None:
        if getattr(event, "widget", None) is not self.window:
            return
        self._closing = True
        self._import_cancel_requested = True
        self._import_in_progress = False
        try:
            if getattr(self.owner, "_mobile_report_browser_dialog", None) is self.window:
                setattr(self.owner, "_mobile_report_browser_dialog", None)
        except Exception:
            pass
        try:
            app_obj = getattr(self.owner, "app", None)
            if app_obj is not None and getattr(app_obj, "_mobile_report_browser_dialog", None) is self.window:
                setattr(app_obj, "_mobile_report_browser_dialog", None)
        except Exception:
            pass

    def _window_alive(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except Exception:
            return False

    def _run_on_ui(self, callback) -> bool:
        if self._closing or not self._window_alive():
            return False

        def guarded_callback() -> None:
            if self._closing or not self._window_alive():
                return
            callback()

        try:
            self.window.after(0, guarded_callback)
            return True
        except Exception:
            return False

    def _request_close(self) -> None:
        reviews = [panel for panel in getattr(self, "_sample_reviews", {}).values() if not panel._closed]
        if reviews:
            for panel in reviews:
                panel.close()
            if any(not panel._closed for panel in reviews):
                def finish_close():
                    if any(panel._saving for panel in reviews):
                        self.window.after(80, finish_close)
                    elif any(not panel._closed for panel in reviews):
                        self._set_status("Nie udało się zapisać weryfikacji. Sprawdź komunikat w jej oknie.", "warning")
                    else:
                        self._request_close()
                self.window.after(80, finish_close)
                return
        self._closing = True
        self._import_cancel_requested = True
        try:
            self.window.destroy()
        except Exception:
            pass

    def _set_import_controls(self, running: bool) -> None:
        self._import_in_progress = bool(running)
        for attr, state in (
            ("btn_import_reports", tk.DISABLED if running else tk.NORMAL),
            ("btn_refresh_reports", tk.DISABLED if running else tk.NORMAL),
        ):
            button = getattr(self, attr, None)
            if button is None:
                continue
            try:
                button.configure(state=state)
            except Exception:
                pass

    @staticmethod
    def _is_mobile_report_file(path: Path) -> bool:
        if Path(path).name.lower() == MobilePackageExperimentStore.FILE_NAME.lower():
            return False
        return str(path.suffix or "").lower() in {".alprsession", ".zip", ".json"}

    @staticmethod
    def _format_file_size(path: Path) -> str:
        try:
            size = int(path.stat().st_size)
        except Exception:
            return "-"
        if size < 1024:
            return f"{size} B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.1f} MB"

    @staticmethod
    def _format_file_mtime(path: Path) -> str:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "-"

    def _ask_mobile_report_paths(self, initial_dir: Path) -> tuple[Path, ...]:
        """Stable in-app multi-file picker for mobile reports.

        Native Windows file dialogs fight with our modeless Tk windows on some
        machines, so report import uses a small Tk picker instead.
        """

        start_dir = Path(initial_dir or Path.home())
        if not start_dir.exists() or not start_dir.is_dir():
            start_dir = Path.home()

        selected_paths: list[Path] = []
        state: dict[str, Any] = {"dir": start_dir}
        entries: list[dict[str, Any]] = []

        dialog = tk.Toplevel(self.window)
        dialog.withdraw()
        dialog.title("Wybierz raporty z telefonu")
        dialog.configure(bg=self.bg)
        try:
            dialog.geometry("920x620")
            dialog.minsize(760, 500)
            dialog.resizable(True, True)
        except Exception:
            pass

        shell = tk.Frame(dialog, bg=self.bg, padx=14, pady=12)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_rowconfigure(3, weight=1)

        title = tk.Label(
            shell,
            text="Wybierz jeden lub kilka raportów z aplikacji mobilnej",
            bg=self.bg,
            fg=self.fg,
            font=("Segoe UI", 13, "bold"),
            anchor=tk.W,
        )
        title.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        subtitle = tk.Label(
            shell,
            text="Obsługiwane formaty: .alprsession, .zip, .json. Zaznacz wiele pozycji klawiszem Ctrl albo Shift.",
            bg=self.bg,
            fg=self.muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=820,
        )
        subtitle.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        path_row = tk.Frame(shell, bg=self.bg)
        path_row.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        path_row.grid_columnconfigure(1, weight=1)
        tk.Label(
            path_row,
            text="Folder:",
            bg=self.bg,
            fg=self.fg,
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        path_var = tk.StringVar(value=str(start_dir))
        path_entry = tk.Entry(
            path_row,
            textvariable=path_var,
            bg=self.panel,
            fg=self.fg,
            insertbackground=self.fg,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.border,
            highlightcolor=self.accent,
            font=("Segoe UI", 9),
        )
        path_entry.grid(row=0, column=1, sticky="ew", ipady=5)

        list_shell = tk.Frame(
            shell,
            bg=self.panel,
            highlightthickness=1,
            highlightbackground=self.border,
            highlightcolor=self.border,
        )
        list_shell.grid(row=3, column=0, sticky="nsew")
        list_shell.grid_columnconfigure(0, weight=1)
        list_shell.grid_rowconfigure(1, weight=1)

        header = tk.Frame(list_shell, bg=blend_hex_colors(self.panel, self.accent, 0.1))
        header.grid(row=0, column=0, sticky="ew")
        for column, (text, width) in enumerate((("Nazwa", 52), ("Typ", 12), ("Rozmiar", 12), ("Data", 18))):
            header.grid_columnconfigure(column, weight=width)
            tk.Label(
                header,
                text=text,
                bg=header.cget("bg"),
                fg=self.fg,
                font=("Segoe UI", 8, "bold"),
                anchor=tk.W,
                padx=8,
                pady=6,
            ).grid(row=0, column=column, sticky="ew")

        body = tk.Frame(list_shell, bg=self.panel)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        listbox = tk.Listbox(
            body,
            selectmode=tk.EXTENDED,
            exportselection=False,
            activestyle="dotbox",
            bg=self.panel,
            fg=self.fg,
            selectbackground=blend_hex_colors(self.accent, self.panel, 0.28),
            selectforeground=self.fg,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            font=("Consolas", 10),
        )
        scrollbar = WebSlimScrollbar(
            body,
            orient=tk.VERTICAL,
            command=listbox.yview,
            track_color=self.palette.get("scrollbar_track", self.panel),
            thumb_color=self.palette.get("scrollbar_thumb", self.accent),
            thumb_hover_color=self.palette.get("scrollbar_thumb_hover", self.palette.get("accent_hover", self.accent)),
        )
        listbox.configure(yscrollcommand=scrollbar.set)
        listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        status_var = tk.StringVar(value="")
        status = tk.Label(
            shell,
            textvariable=status_var,
            bg=self.bg,
            fg=self.muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
        )
        status.grid(row=4, column=0, sticky="ew", pady=(8, 8))

        actions = tk.Frame(shell, bg=self.bg)
        actions.grid(row=5, column=0, sticky="ew")
        actions.grid_columnconfigure(6, weight=1)

        def _display_row(name: str, kind: str, size: str, mtime: str) -> str:
            return f"{name[:60]:<62} {kind[:10]:<10} {size[:10]:>10}  {mtime[:16]:<16}"

        def _set_status(text: str, tone: str = "info") -> None:
            status_var.set(text)
            color = {
                "error": self.error,
                "warning": self.warning,
                "success": self.success,
            }.get(tone, self.muted)
            try:
                status.configure(fg=color)
            except Exception:
                pass

        def _load_dir(path: Path | str) -> None:
            try:
                target = Path(str(path or "")).expanduser()
                if not target.exists() or not target.is_dir():
                    _set_status("Ten folder nie istnieje albo nie jest dostępny.", "error")
                    return
                target = target.resolve()
            except Exception as exc:
                _set_status(f"Nie udało się otworzyć folderu: {exc}", "error")
                return

            entries.clear()
            listbox.delete(0, tk.END)
            state["dir"] = target
            path_var.set(str(target))

            try:
                children = list(target.iterdir())
            except Exception as exc:
                _set_status(f"Nie udało się odczytać folderu: {exc}", "error")
                return

            if target.parent != target:
                entries.append({"path": target.parent, "dir": True, "report": False})
                listbox.insert(tk.END, _display_row("..", "folder", "", ""))

            dirs = sorted((child for child in children if child.is_dir()), key=lambda p: p.name.lower())
            files = sorted(
                (child for child in children if child.is_file() and self._is_mobile_report_file(child)),
                key=lambda p: p.name.lower(),
            )
            for child in dirs:
                entries.append({"path": child, "dir": True, "report": False})
                listbox.insert(tk.END, _display_row(child.name, "folder", "", self._format_file_mtime(child)))
            for child in files:
                entries.append({"path": child, "dir": False, "report": True})
                listbox.insert(
                    tk.END,
                    _display_row(child.name, child.suffix.lower().lstrip(".") or "plik", self._format_file_size(child), self._format_file_mtime(child)),
                )

            _set_status(
                f"Folder: {target} | Raporty: {len(files)} | Podfoldery: {len(dirs)}",
                "info",
            )

        def _selected_report_paths() -> list[Path]:
            result: list[Path] = []
            for raw_index in listbox.curselection():
                try:
                    entry = entries[int(raw_index)]
                except Exception:
                    continue
                path = Path(entry.get("path"))
                if entry.get("report") and path.exists() and path.is_file():
                    result.append(path)
            return result

        def _open_or_accept(_event=None):
            selection = list(listbox.curselection())
            if len(selection) == 1:
                try:
                    entry = entries[int(selection[0])]
                except Exception:
                    entry = {}
                if entry.get("dir"):
                    _load_dir(Path(entry.get("path")))
                    return "break"
            _accept()
            return "break"

        def _accept() -> None:
            reports = _selected_report_paths()
            if not reports:
                _set_status("Zaznacz przynajmniej jeden raport. Folder otworzysz podwójnym kliknięciem.", "warning")
                return
            selected_paths[:] = reports
            _close()

        def _close() -> None:
            try:
                dialog.grab_release()
            except Exception:
                pass
            try:
                dialog.destroy()
            except Exception:
                pass

        def _go_up() -> None:
            current = Path(state.get("dir") or start_dir)
            if current.parent != current:
                _load_dir(current.parent)

        quick_dirs = [
            ("Raporty", start_dir),
            ("Pobrane", Path.home() / "Downloads"),
            ("Pulpit", Path.home() / "Desktop"),
        ]
        for index, (label, folder) in enumerate(quick_dirs):
            ttk.Button(actions, text=label, command=lambda p=folder: _load_dir(p)).grid(
                row=0,
                column=index,
                sticky="w",
                padx=(0, 6),
                ipadx=6,
                ipady=2,
            )
        ttk.Button(actions, text="Folder wyżej", command=_go_up).grid(row=0, column=3, sticky="w", padx=(8, 6), ipadx=6, ipady=2)
        ttk.Button(actions, text="Odśwież", command=lambda: _load_dir(Path(state.get("dir") or start_dir))).grid(
            row=0,
            column=4,
            sticky="w",
            padx=(0, 6),
            ipadx=6,
            ipady=2,
        )
        ttk.Button(actions, text="Anuluj", command=_close).grid(row=0, column=7, sticky="e", padx=(8, 6), ipadx=8, ipady=2)
        ttk.Button(actions, text="Importuj zaznaczone", command=_accept).grid(row=0, column=8, sticky="e", ipadx=10, ipady=2)

        path_entry.bind("<Return>", lambda _event: (_load_dir(path_var.get()), "break")[-1], add="+")
        listbox.bind("<Double-Button-1>", _open_or_accept, add="+")
        listbox.bind("<Return>", _open_or_accept, add="+")
        dialog.bind("<Escape>", lambda _event: (_close(), "break")[-1], add="+")
        dialog.protocol("WM_DELETE_WINDOW", _close)

        _load_dir(start_dir)
        try:
            dialog.update_idletasks()
            parent_x = int(self.window.winfo_rootx())
            parent_y = int(self.window.winfo_rooty())
            parent_w = int(self.window.winfo_width() or 920)
            parent_h = int(self.window.winfo_height() or 620)
            width = min(940, max(760, int(parent_w * 0.86)))
            height = min(660, max(500, int(parent_h * 0.82)))
            x = parent_x + max(18, int((parent_w - width) / 2))
            y = parent_y + max(18, int((parent_h - height) / 2))
            dialog.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            pass
        try:
            dialog.deiconify()
            dialog.lift()
            dialog.focus_force()
            dialog.grab_set()
            listbox.focus_set()
        except Exception:
            pass
        try:
            dialog.wait_window()
        except Exception:
            pass
        return tuple(selected_paths)

    def _build(self) -> None:
        root = tk.Frame(self.window, bg=self.bg, padx=12, pady=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(2, weight=1)

        header = tk.Frame(root, bg=self.bg)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="Przeglądarka raportów mobilnego ALPR",
            bg=self.bg,
            fg=self.fg,
            font=("Segoe UI", 13, "bold"),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            header,
            text=(
                "Importuj jeden lub wiele plików `.alprsession` albo ZIP z Androida. Metryki zbiorcze pochodzą z report.json, "
                "a wykresy klatkowe z traces.csv."
            ),
            bg=self.bg,
            fg=self.muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=980,
        ).grid(row=1, column=0, sticky="ew", pady=(3, 0))

        actions = tk.Frame(root, bg=self.bg)
        actions.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        actions.grid_columnconfigure(3, weight=1)
        self.btn_import_reports = ttk.Button(actions, text="Importuj raporty z Androida", command=self.import_report)
        self.btn_import_reports.grid(
            row=0, column=0, sticky="w", padx=(0, 8), ipadx=10, ipady=2
        )
        self.btn_refresh_reports = ttk.Button(actions, text="Odśwież zapisane", command=self._refresh_report_list)
        self.btn_refresh_reports.grid(
            row=0, column=1, sticky="w", padx=(0, 8), ipadx=8, ipady=2
        )
        self.btn_review_samples = ttk.Button(actions, text="Weryfikacja próbek", command=self.open_sample_review, state="disabled")
        self.btn_review_samples.grid(row=0, column=2, sticky="w", padx=(0, 8), ipadx=8, ipady=2)
        ttk.Button(actions, text="Zamknij", command=self._request_close).grid(row=0, column=4, sticky="e", ipadx=8, ipady=2)

        self.status_var = tk.StringVar(value="Gotowe. Wskaż raport lub kilka raportów z telefonu albo wybierz zapisany raport z listy.")
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=150)
        self.status_label = tk.Label(
            actions,
            textvariable=self.status_var,
            bg=self.bg,
            fg=self.muted,
            font=("Segoe UI", 9),
            anchor=tk.W,
        )
        self.status_label.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(8, 0))

        workspace = tk.Frame(root, bg=self.bg)
        workspace.grid(row=2, column=0, sticky="nsew")
        workspace.grid_columnconfigure(0, weight=1, minsize=360)
        workspace.grid_columnconfigure(1, weight=2, minsize=560)
        workspace.grid_rowconfigure(0, weight=1)

        left = tk.Frame(
            workspace,
            bg=self.card_bg,
            padx=9,
            pady=8,
            highlightthickness=1,
            highlightbackground=self.border,
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)
        tk.Label(
            left,
            text="Raporty pomiarów",
            bg=self.card_bg,
            fg=self.fg,
            font=("Segoe UI", 10, "bold"),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            left,
            text="Lista pokazuje zapisane raporty. Import nie uruchamia modeli i nie rozpakowuje paczki do projektu.",
            bg=self.card_bg,
            fg=self.muted,
            font=("Segoe UI", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=330,
        ).grid(row=1, column=0, sticky="ew", pady=(2, 7))

        self.report_table = _TreeTable(
            left,
            ("Data", "Pakiet", "Wariant", "Telefon", "p95", "Jakość", "Score"),
            ("Data", "Pakiet", "Wariant", "Telefon", "p95", "Jakość", "Score"),
            (92, 120, 90, 100, 70, 70, 58),
            palette={**self.palette, "panel": self.card_bg},
            height=18,
        )
        self.report_table.shell.grid(row=2, column=0, sticky="nsew")
        self.report_table.tree.bind("<<TreeviewSelect>>", self._on_report_selected, add="+")

        right = tk.Frame(workspace, bg=self.bg)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        self.card_shell = tk.Frame(right, bg=self.bg)
        self.card_shell.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for column in range(4):
            self.card_shell.grid_columnconfigure(column, weight=1, uniform="report_cards")
        self.card_vars = {
            "integrity": tk.StringVar(value="Integralność: -"),
            "frames": tk.StringVar(value="Klatki: -"),
            "quality": tk.StringVar(value="Jakość: -"),
            "latency": tk.StringVar(value="p95: -"),
        }
        self.card_labels: dict[str, tk.Label] = {}
        for index, (key, title, color) in enumerate(
            (
                ("integrity", "Integralność", self.accent),
                ("frames", "Przebieg", self.warning),
                ("quality", "Jakość", self.success),
                ("latency", "Opóźnienia", self.accent),
            )
        ):
            card = tk.Frame(
                self.card_shell,
                bg=blend_hex_colors(self.card_bg, color, 0.055),
                padx=9,
                pady=7,
                highlightthickness=1,
                highlightbackground=blend_hex_colors(color, self.bg, 0.45),
            )
            card.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 6, 0))
            tk.Label(card, text=title, bg=card["bg"], fg=self.muted, font=("Segoe UI", 8, "bold"), anchor=tk.W).pack(
                fill=tk.X
            )
            label = tk.Label(
                card,
                textvariable=self.card_vars[key],
                bg=card["bg"],
                fg=color,
                font=("Segoe UI", 10, "bold"),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=180,
            )
            label.pack(fill=tk.X, pady=(2, 0))
            self.card_labels[key] = label

        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self._build_tabs()

    def _build_tabs(self) -> None:
        def tab_frame() -> tk.Frame:
            frame = tk.Frame(self.notebook, bg=self.panel, padx=8, pady=8)
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(0, weight=1)
            return frame

        summary = tab_frame()
        self.summary_table = _WrappedLabelTable(
            summary,
            ("Pole", "Wartość", "Opis"),
            ("Pole", "Wartość", "Opis"),
            (190, 250, 330),
            palette={**self.palette, "panel": self.panel},
            height=17,
        )
        self.summary_table.shell.grid(row=0, column=0, sticky="nsew")
        self.notebook.add(summary, text="Podsumowanie")

        comparison = tab_frame()
        self.comparison_table = _WrappedLabelTable(
            comparison,
            ("Kryterium", "Wybrany raport", "Seria", "Status", "Znaczenie"),
            ("Kryterium", "Wybrany raport", "Seria", "Status", "Znaczenie"),
            (170, 220, 160, 120, 330),
            palette={**self.palette, "panel": self.panel},
            height=17,
        )
        self.comparison_table.shell.grid(row=0, column=0, sticky="nsew")
        self.notebook.add(comparison, text="Porównywalność")

        config = tab_frame()
        self.config_table = _WrappedLabelTable(
            config,
            ("Sekcja", "Wartość", "Opis"),
            ("Sekcja", "Wartość", "Opis"),
            (260, 360, 240),
            palette={**self.palette, "panel": self.panel},
            height=17,
        )
        self.config_table.shell.grid(row=0, column=0, sticky="nsew")
        self.notebook.add(config, text="Konfiguracja")

        latency = tk.Frame(self.notebook, bg=self.panel, padx=8, pady=8)
        latency.grid_columnconfigure(0, weight=1)
        latency.grid_rowconfigure(1, weight=1)
        self.latency_canvas = tk.Canvas(latency, bg=self.panel, highlightthickness=1, highlightbackground=self.border, height=180)
        self.latency_canvas.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.latency_canvas.bind("<Configure>", lambda _event: self._draw_latency_chart(), add="+")
        self.latency_table = _TreeTable(
            latency,
            ("Etap", "count", "mean", "p50", "p90", "p95", "p99", "min", "max", "std"),
            ("Etap", "n", "mean", "p50", "p90", "p95", "p99", "min", "max", "std"),
            (160, 48, 72, 72, 72, 72, 72, 72, 72, 72),
            palette={**self.palette, "panel": self.panel},
            height=11,
        )
        self.latency_table.shell.grid(row=1, column=0, sticky="nsew")
        self.notebook.add(latency, text="Opóźnienia")

        artifacts = tab_frame()
        self.artifacts_table = _WrappedLabelTable(
            artifacts,
            ("Artefakt", "Źródło", "Preview", "Status", "Opis"),
            ("Artefakt", "Źródło", "Preview", "Status", "Opis"),
            (170, 90, 90, 110, 430),
            palette={**self.palette, "panel": self.panel},
            height=17,
        )
        self.artifacts_table.shell.grid(row=0, column=0, sticky="nsew")
        self.notebook.add(artifacts, text="Artefakty")

        quality = tab_frame()
        self.quality_table = _WrappedLabelTable(
            quality,
            ("Metryka", "Wartość", "Komentarz"),
            ("Metryka", "Wartość", "Komentarz"),
            (230, 160, 420),
            palette={**self.palette, "panel": self.panel},
            height=17,
        )
        self.quality_table.shell.grid(row=0, column=0, sticky="nsew")
        self.notebook.add(quality, text="Jakość")

        diagnostics = tk.Frame(self.notebook, bg=self.panel, padx=8, pady=8)
        diagnostics.grid_columnconfigure(0, weight=1)
        diagnostics.grid_rowconfigure(0, weight=1)
        diagnostics.grid_rowconfigure(1, weight=1)
        self.diagnostics_table = _WrappedLabelTable(
            diagnostics,
            ("Obszar", "Wartość", "Opis"),
            ("Obszar", "Wartość", "Opis"),
            (260, 240, 360),
            palette={**self.palette, "panel": self.panel},
            height=8,
        )
        self.diagnostics_table.shell.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.log_text = tk.Text(
            diagnostics,
            bg=blend_hex_colors(self.panel, self.bg, 0.35),
            fg=self.fg,
            insertbackground=self.fg,
            height=8,
            wrap=tk.WORD,
            font=("Consolas", 9),
            relief=tk.FLAT,
        )
        log_scroll = WebSlimScrollbar(diagnostics, orient=tk.VERTICAL, command=self.log_text.yview, track_color=self.panel)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=1, column=0, sticky="nsew")
        log_scroll.grid(row=1, column=1, sticky="ns")
        self.notebook.add(diagnostics, text="Diagnostyka")

        crops = tab_frame()
        self.crops_table = _TreeTable(
            crops,
            ("ID", "Track", "Status", "Predykcja", "GT", "Czas", "Crop"),
            ("ID", "Track", "Status", "Predykcja", "GT", "Czas", "Crop"),
            (120, 70, 100, 120, 120, 100, 240),
            palette={**self.palette, "panel": self.panel},
            height=17,
        )
        self.crops_table.shell.grid(row=0, column=0, sticky="nsew")
        self.notebook.add(crops, text="Cropy")

        raw = tk.Frame(self.notebook, bg=self.panel, padx=8, pady=8)
        raw.grid_columnconfigure(0, weight=1)
        raw.grid_rowconfigure(0, weight=1)
        self.raw_text = tk.Text(
            raw,
            bg=blend_hex_colors(self.panel, self.bg, 0.35),
            fg=self.fg,
            insertbackground=self.fg,
            wrap=tk.NONE,
            font=("Consolas", 9),
            relief=tk.FLAT,
        )
        raw_scroll_y = WebSlimScrollbar(raw, orient=tk.VERTICAL, command=self.raw_text.yview, track_color=self.panel)
        raw_scroll_x = WebSlimScrollbar(raw, orient=tk.HORIZONTAL, command=self.raw_text.xview, track_color=self.panel)
        self.raw_text.configure(yscrollcommand=raw_scroll_y.set, xscrollcommand=raw_scroll_x.set)
        self.raw_text.grid(row=0, column=0, sticky="nsew")
        raw_scroll_y.grid(row=0, column=1, sticky="ns")
        raw_scroll_x.grid(row=1, column=0, sticky="ew")
        self.notebook.add(raw, text="Surowe dane")

    def _set_status(self, text: str, tone: str = "info") -> None:
        colors = {"info": self.muted, "success": self.success, "warning": self.warning, "error": self.error}
        try:
            self.status_var.set(text)
            self.status_label.configure(fg=colors.get(tone, self.muted))
        except Exception:
            pass

    def _set_loading(self, loading: bool, *, determinate: bool = False, maximum: int = 100, value: float = 0.0) -> None:
        try:
            if loading:
                self.progress.stop()
                self.progress.configure(
                    mode="determinate" if determinate else "indeterminate",
                    maximum=max(1, int(maximum or 1)),
                    value=max(0.0, float(value or 0.0)),
                )
                self.progress.grid(row=0, column=2, sticky="w", padx=(0, 10))
                if not determinate:
                    self.progress.start(14)
            else:
                self.progress.stop()
                self.progress.grid_remove()
        except Exception:
            pass

    def _set_loading_progress(self, value: float, *, maximum: int | None = None) -> None:
        try:
            if maximum is not None:
                self.progress.configure(maximum=max(1, int(maximum or 1)))
            self.progress.configure(value=max(0.0, float(value or 0.0)))
        except Exception:
            pass

    def _refresh_report_list(self, select_report_id: str = "", *, auto_select: bool = True) -> None:
        try:
            self.store._load()
        except Exception:
            pass
        try:
            self.store.reports = [_compact_mobile_report_for_store(report) for report in self.store.reports]
        except Exception:
            pass
        self.report_by_iid = {}
        rows: list[tuple[Any, ...]] = []
        selected_iid = ""
        for index, report in enumerate(sorted(self.store.reports, key=lambda item: str(item.measured_at or ""), reverse=True)):
            iid = f"report_{index}_{report.report_id}"
            self.report_by_iid[iid] = report
            rows.append(_report_list_values(report))
            if select_report_id and report.report_id == select_report_id:
                selected_iid = iid
        self.report_table.set_rows(rows)
        # TreeTable generates row_ indexes, so map them back to report objects.
        ordered_reports = sorted(self.store.reports, key=lambda item: str(item.measured_at or ""), reverse=True)
        self.report_by_iid = {f"row_{index}": report for index, report in enumerate(ordered_reports)}
        if ordered_reports and (auto_select or select_report_id):
            if select_report_id:
                selected_iid = next(
                    (iid for iid, report in self.report_by_iid.items() if report.report_id == select_report_id),
                    "row_0",
                )
            else:
                selected_iid = "row_0"
            self._suppress_report_select = True
            try:
                self.report_table.tree.selection_set(selected_iid)
                self.report_table.tree.focus(selected_iid)
                self.report_table.tree.see(selected_iid)
            except Exception:
                pass
            finally:
                self._suppress_report_select = False
            selected_report = self.report_by_iid.get(selected_iid)
            selected_bundle = self.bundle_by_report_id.get(selected_report.report_id) if selected_report else None
            self._show_report(selected_report, selected_bundle)
        elif ordered_reports:
            self._show_report_hint(len(ordered_reports))
        else:
            self._show_empty()
        self._set_status(f"Zapisane raporty: {len(ordered_reports)}.", "info")

    def _on_report_selected(self, _event=None) -> None:
        if self._suppress_report_select:
            return
        try:
            selected = self.report_table.tree.selection()
        except Exception:
            selected = ()
        if not selected:
            return
        report = self.report_by_iid.get(str(selected[0]))
        if report:
            self._show_report(report, self.bundle_by_report_id.get(report.report_id))

    def import_report(self) -> None:
        if self._import_in_progress:
            self._set_status("Import raportów już trwa. Poczekaj na zakończenie albo zamknij okno, aby przerwać odświeżanie UI.", "warning")
            return
        self._import_cancel_requested = False
        initial_dir = Path(getattr(CONFIG, "DIR_7_RANKINGS_MOBILE_PACKAGES", CONFIG.DIR_7_RANKINGS / "mobile_packages"))
        try:
            initial_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._set_import_controls(True)
        self._set_status("Otwieram wybór raportów z telefonu...", "info")
        path_values = self._ask_mobile_report_paths(initial_dir)
        if not path_values:
            self._set_import_controls(False)
            self._set_status("Import anulowany. Nie wybrano plików raportów.", "warning")
            return
        report_paths = [Path(value) for value in path_values if str(value or "").strip()]
        if not report_paths:
            self._set_import_controls(False)
            self._set_status("Import anulowany. Nie wybrano poprawnych plików.", "warning")
            return
        total_files = len(report_paths)
        if total_files == 1:
            self._set_status(f"Czytam raport: {report_paths[0].name}...", "info")
            self._set_loading(True)
        else:
            self._set_status(f"Czytam pliki raportów: 0/{total_files}.", "info")
            self._set_loading(True, determinate=True, maximum=total_files, value=0)

        def worker() -> None:
            bundles: list[tuple[Path, MobileReportBundle]] = []
            invalid: list[tuple[Path, MobileReportBundle]] = []
            failures: list[tuple[Path, str]] = []
            saved = 0
            last_saved_report_id = ""
            save_error = ""
            try:
                for index, report_path in enumerate(report_paths, start=1):
                    if self._import_cancel_requested:
                        break
                    self._run_on_ui(
                        lambda i=index, p=report_path: (
                            self._set_status(f"Czytam plik {i}/{total_files}: {p.name}...", "info"),
                            self._set_loading_progress(i - 1, maximum=total_files) if total_files > 1 else None,
                        )
                    )
                    try:
                        for bundle in read_mobile_report_bundles(report_path):
                            if self._import_cancel_requested:
                                break
                            bundles.append((report_path, bundle))
                            if bundle.validation.ok:
                                try:
                                    self.store.reports = [
                                        _compact_mobile_report_for_store(report)
                                        for report in self.store.reports
                                    ]
                                    self.store.add_report(_compact_mobile_report_for_store(bundle.report), save=False)
                                except Exception as exc:
                                    failures.append((report_path, f"Raport odczytany, ale nie udało się go przygotować do zapisu: {exc}"))
                                    continue
                                saved += 1
                                last_saved_report_id = bundle.report.report_id
                            else:
                                invalid.append((report_path, bundle))
                    except Exception as exc:
                        failures.append((report_path, str(exc)))
                        logger.exception(f"Nie udało się odczytać raportu mobilnego: {report_path}")
                    if total_files > 1:
                        self._run_on_ui(lambda i=index: self._set_loading_progress(i, maximum=total_files))

                if saved and not self._import_cancel_requested:
                    try:
                        self._run_on_ui(lambda: self._set_status("Zapisuję magazyn raportów...", "info"))
                        self.store.reports = [
                            _compact_mobile_report_for_store(report)
                            for report in self.store.reports
                        ]
                        self.store.save()
                    except Exception as exc:
                        save_error = str(exc)
                        logger.exception("Nie udało się zapisać raportu mobilnego")

                def done() -> None:
                    self._set_import_controls(False)
                    self._set_loading(False)
                    if self._import_cancel_requested:
                        self._set_status("Import przerwany. Okno można bezpiecznie zamknąć.", "warning")
                        return
                    for path, bundle in bundles:
                        self.current_bundle = bundle
                        self.bundle_by_report_id[bundle.report.report_id] = bundle
                    if save_error:
                        self._set_status(f"Raporty odczytane, ale nie udało się zapisać magazynu: {save_error}", "warning")
                        if bundles:
                            self._show_report(bundles[-1][1].report, bundles[-1][1])
                        return

                    if saved:
                        self._refresh_report_list(last_saved_report_id)
                    elif bundles:
                        self._show_report(bundles[-1][1].report, bundles[-1][1])
                    else:
                        self._show_empty()

                    failed_count = len(failures)
                    invalid_count = len(invalid)
                    if saved and not failed_count and not invalid_count:
                        if total_files == 1 and saved == 1 and bundles:
                            report = bundles[0][1].report
                            self._set_status(f"Raport zaimportowany: {report.package_id} / {report.variant_id}.", "success")
                        else:
                            self._set_status(f"Zaimportowano {saved} raportów z {total_files} plików.", "success")
                        return

                    if saved:
                        self._set_status(
                            f"Zaimportowano {saved} raportów. Do kontroli: {invalid_count}, błędy odczytu: {failed_count}.",
                            "warning",
                        )
                        return

                    self._set_status(
                        f"Nie zapisano raportów. Do kontroli: {invalid_count}, błędy odczytu: {failed_count}.",
                        "error",
                    )
                    details = []
                    for path, bundle in invalid[:4]:
                        reason = "; ".join(bundle.validation.errors[:2]) or "walidacja zgłosiła błędy"
                        details.append(f"- {path.name}: {reason}")
                    for path, reason in failures[:4]:
                        details.append(f"- {path.name}: {reason}")
                    if details:
                        messagebox.showerror("Import raportów", "\n".join(details), parent=self.window)

                self._run_on_ui(done)
            except Exception as exc:
                error_text = str(exc)
                logger.exception("Nie udało się zaimportować raportów mobilnych")

                def failed() -> None:
                    self._set_import_controls(False)
                    self._set_loading(False)
                    self._set_status(f"Import nieudany: {error_text}", "error")
                    messagebox.showerror("Błąd importu raportu", error_text, parent=self.window)

                self._run_on_ui(failed)

        threading.Thread(target=worker, name="mobile-report-import", daemon=True).start()

    def _show_empty(self) -> None:
        self.current_bundle = None
        self.current_report = None
        if hasattr(self, "btn_review_samples"):
            self.btn_review_samples.configure(state="disabled")
        for var in self.card_vars.values():
            var.set("-")
        self.summary_table.set_rows([("Brak raportów", "Importuj raport z Androida", "Obsługiwane są .alprsession, ZIP benchmarku i JSON.")])
        self.comparison_table.set_rows([])
        self.config_table.set_rows([])
        self.latency_table.set_rows([])
        self.artifacts_table.set_rows([])
        self.quality_table.set_rows([])
        self.diagnostics_table.set_rows([])
        self.crops_table.set_rows([])
        self._replace_text(self.log_text, "")
        self._replace_text(self.raw_text, "")
        self._draw_latency_chart()

    def _show_report_hint(self, report_count: int) -> None:
        self.current_bundle = None
        self.current_report = None
        if hasattr(self, "btn_review_samples"):
            self.btn_review_samples.configure(state="disabled")
        for var in self.card_vars.values():
            var.set("-")
        self.summary_table.set_rows(
            [
                (
                    "Raporty gotowe",
                    str(max(0, int(report_count or 0))),
                    "Wybierz raport z listy po lewej, aby załadować szczegóły, wykresy i surowy podgląd.",
                )
            ]
        )
        self.comparison_table.set_rows([])
        self.config_table.set_rows([])
        self.latency_table.set_rows([])
        self.artifacts_table.set_rows([])
        self.quality_table.set_rows([])
        self.diagnostics_table.set_rows([])
        self.crops_table.set_rows([])
        self._replace_text(self.log_text, "")
        self._replace_text(self.raw_text, "")
        self._draw_latency_chart()

    def _show_report(self, report: MobileBenchmarkReport | None, bundle: MobileReportBundle | None = None) -> None:
        if report is None:
            self._show_empty()
            return
        self.current_bundle = bundle
        self.current_report = report
        if hasattr(self, "btn_review_samples"):
            self.btn_review_samples.configure(state="normal" if (bundle or report.source_path) else "disabled")
        self._update_cards(report, bundle)
        self._populate_summary(report, bundle)
        self._populate_comparison_guard(report, bundle)
        self._populate_config(report)
        self._populate_latency(report, bundle)
        self._populate_artifacts(report, bundle)
        self._populate_quality(report)
        self._populate_diagnostics(report, bundle)
        self._populate_crops(report, bundle)
        self._populate_raw(report, bundle)

    def open_sample_review(self) -> None:
        report = getattr(self, "current_report", None)
        if report is None:
            self._set_status("Wybierz sesję z listy raportów.", "warning")
            return
        from .z4_mobile_sample_review import open_mobile_sample_review
        open_mobile_sample_review(self, report, self.current_bundle)

    def _update_cards(self, report: MobileBenchmarkReport, bundle: MobileReportBundle | None) -> None:
        if bundle:
            if bundle.validation.ok:
                integrity = f"OK | SHA: {bundle.validation.checked_hashes}"
                tone = self.success
            else:
                integrity = f"Błędy: {len(bundle.validation.errors)}"
                tone = self.error
            if bundle.validation.warnings and bundle.validation.ok:
                integrity += f" | ostrz. {len(bundle.validation.warnings)}"
        else:
            source_hash = str(getattr(report, "source_archive_sha256", "") or "").strip()
            integrity = f"Zapisany | SHA {source_hash[:8]}" if source_hash else "Zapisany JSON"
            tone = self.accent
        trace_total = bundle.trace_total if bundle else _safe_int(_nested_value(report.raw, "summary.processed_frames"), 0)
        if not trace_total:
            trace_total = _safe_int(_nested_value(report.raw, "summary.processed_frames"), 0)
        self.card_vars["integrity"].set(integrity)
        self.card_vars["frames"].set(f"{trace_total or '-'} klatek")
        self.card_vars["quality"].set(_quality_value(report))
        self.card_vars["latency"].set(_format_ms(_pipeline_p95(report)))
        try:
            self.card_labels["integrity"].configure(fg=tone)
        except Exception:
            pass

    def _populate_summary(self, report: MobileBenchmarkReport, bundle: MobileReportBundle | None) -> None:
        validation_rows: list[tuple[Any, ...]] = []
        if bundle:
            validation_rows.append(
                (
                    "Walidacja paczki",
                    "OK" if bundle.validation.ok else "Błąd",
                    f"SHA sprawdzone: {bundle.validation.checked_hashes}, ostrzeżenia: {len(bundle.validation.warnings)}",
                    {"tag": "success" if bundle.validation.ok else "error"},
                )
            )
            for message in bundle.validation.errors[:6]:
                validation_rows.append(("Błąd walidacji", message, "Raport pokazany tylko diagnostycznie.", {"tag": "error"}))
            for message in bundle.validation.warnings[:6]:
                validation_rows.append(("Ostrzeżenie", message, "Dane niekrytyczne albo zgodność wsteczna.", {"tag": "warning"}))
        rows: list[tuple[Any, ...]] = [
            ("Typ paczki", bundle.bundle_kind if bundle else "zapisany raport", bundle.bundle_schema if bundle else ""),
            ("Report ID", report.report_id, "Identyfikator pomiaru z Androida."),
            (
                "Hash źródła",
                str(bundle.source_archive_sha256 if bundle else getattr(report, "source_archive_sha256", "") or "")[:16] or "-",
                "Deduplikacja identycznego archiwum/pliku i ślad odtwarzalności.",
            ),
            ("Pakiet", report.package_id, "Model/pakiet badany w telefonie."),
            ("Wariant", report.variant_id, "Runtime, precyzja lub wariant wykonawczy."),
            ("Pomiar", _format_datetime(report.measured_at), "Data i czas zapisane przez klienta mobilnego."),
            ("Urządzenie", _device_label(report), _short_text(report.device, 180)),
            (
                "Klatki",
                str(bundle.trace_total if bundle else _nested_value(report.raw, "summary.processed_frames") or "-"),
                "Ślad per klatka pochodzi z traces.csv albo traces[].",
            ),
            (
                "Cropy",
                str((bundle.crop_count if bundle else 0) or _safe_int(_nested_value(report.raw, "crop_session.collected_count"), 0) or "-"),
                "Obrazy nie są ładowane automatycznie do pamięci.",
            ),
        ]
        index = dict((bundle.experiment_session if bundle else getattr(report, "experiment_index", {})) or {})
        if index:
            rows.extend(
                [
                    ("Seria", str(index.get("series_id") or "-"), "Identyfikator kampanii porównawczej."),
                    ("Scenariusz", str(index.get("scenario_id") or "-"), "Materiał albo warunki sceny eksperymentalnej."),
                    ("Zmienna/wariant", str(index.get("experiment_variant") or "-"), "Badany wariant w ramach serii."),
                    ("Replika", str(index.get("replicate_index") or "-"), "Numer powtórzenia tego samego wariantu."),
                ]
            )
        self.summary_table.set_rows(validation_rows + rows)

    def _report_experiment_index(
        self,
        report: MobileBenchmarkReport,
        bundle: MobileReportBundle | None = None,
    ) -> dict[str, Any]:
        if bundle and isinstance(bundle.experiment_session, dict):
            return dict(bundle.experiment_session or {})
        raw = dict(report.raw or {})
        return dict(getattr(report, "experiment_index", {}) or raw.get("desktop_experiment_index") or {})

    def _guard_profile_hash(self, value: Any) -> str:
        if value in (None, "", {}, []):
            return "-"
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            text = str(value)
        digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:8]
        return f"profil {digest}"

    def _guard_resolution_value(self, report: MobileBenchmarkReport, index: dict[str, Any]) -> str:
        resolution = index.get("resolution") if isinstance(index.get("resolution"), dict) else {}
        raw = dict(report.raw or {})
        width = _nested_value(resolution, "width_px", "width") or _nested_value(raw, "capture.width_px", "capture.width", "camera.width_px", "input.width_px")
        height = _nested_value(resolution, "height_px", "height") or _nested_value(raw, "capture.height_px", "capture.height", "camera.height_px", "input.height_px")
        fps = _nested_value(resolution, "fps") or _nested_value(raw, "capture.fps", "camera.fps", "input.fps")
        if width and height and fps:
            return f"{width}x{height} @ {fps} fps"
        if width and height:
            return f"{width}x{height}"
        return "-"

    def _guard_android_version_value(self, report: MobileBenchmarkReport, index: dict[str, Any]) -> str:
        device = dict(getattr(report, "device", {}) or {})
        raw = dict(getattr(report, "raw", {}) or {})
        return str(
            device.get("android_version")
            or _nested_value(index, "device.android_version", "android_version")
            or _nested_value(raw, "device.android_version", "android_version", "system.android_version")
            or "-"
        )

    def _guard_app_git_sha_value(self, report: MobileBenchmarkReport, index: dict[str, Any]) -> str:
        raw = dict(getattr(report, "raw", {}) or {})
        return str(
            index.get("app_git_sha")
            or _nested_value(raw, "app_build.git_commit", "app_build.git_sha", "build.git_commit", "app_git_sha")
            or "-"
        )

    def _guard_app_version_value(self, report: MobileBenchmarkReport, index: dict[str, Any]) -> str:
        raw = dict(getattr(report, "raw", {}) or {})
        return str(
            index.get("app_version")
            or _nested_value(raw, "app_build.version", "app_version", "application.version")
            or "-"
        )

    def _guard_model_fingerprint_value(self, report: MobileBenchmarkReport, index: dict[str, Any], role: str) -> str:
        role_key = str(role or "").strip().lower()
        raw = dict(getattr(report, "raw", {}) or {})
        fingerprints: dict[str, Any] = {}
        for candidate in (
            index.get("model_fingerprints") if isinstance(index.get("model_fingerprints"), dict) else {},
            _nested_value(raw, "model_fingerprints"),
            _nested_value(raw, "execution"),
            _nested_value(raw, "execution.model_fingerprints"),
            _nested_value(raw, "execution.models"),
            _nested_value(raw, "runtime_composition"),
            _nested_value(raw, "runtime_composition.models"),
            _nested_value(raw, "models"),
        ):
            if isinstance(candidate, dict):
                _merge_model_fingerprint_aliases(fingerprints, candidate)
        value = fingerprints.get(role_key)
        if value in (None, "", {}, []):
            return "-"
        if isinstance(value, dict):
            for key in (
                "sha256",
                "file_sha256",
                "model_sha256",
                "checkpoint_sha256",
                "fingerprint",
                "hash",
                "id",
                "model_id",
            ):
                text = str(value.get(key) or "").strip()
                if text:
                    return text[:16] if len(text) > 20 else text
            return self._guard_profile_hash(value).replace("profil ", "fp ")
        text = str(value or "").strip()
        return text[:16] if len(text) > 20 else (text or "-")

    def _guard_artifact_count(self, index: dict[str, Any], key: str) -> int:
        counts = index.get("artifact_counts") if isinstance(index.get("artifact_counts"), dict) else {}
        if key == "traces":
            return _safe_int(index.get("trace_total_source") or counts.get("traces"))
        return _safe_int(counts.get(key))

    def _populate_comparison_guard(self, report: MobileBenchmarkReport, bundle: MobileReportBundle | None) -> None:
        selected_index = self._report_experiment_index(report, bundle)
        series_id = str(selected_index.get("series_id") or "").strip()
        scenario_id = str(selected_index.get("scenario_id") or "").strip()

        def matches_group(candidate: MobileBenchmarkReport) -> bool:
            index = self._report_experiment_index(candidate)
            candidate_series = str(index.get("series_id") or "").strip()
            candidate_scenario = str(index.get("scenario_id") or "").strip()
            if series_id and scenario_id:
                return candidate_series == series_id and candidate_scenario == scenario_id
            if series_id:
                return candidate_series == series_id
            return False

        cohort = [candidate for candidate in self.store.reports if matches_group(candidate)]
        if not any(candidate.identity == report.identity for candidate in cohort):
            cohort.append(report)

        rows: list[tuple[Any, ...]] = []
        if not series_id:
            rows.append(
                (
                    "Indeks eksperymentu",
                    "-",
                    "-",
                    "Brak serii",
                    "Raport da się obejrzeć, ale bez series_id nie powinien automatycznie trafiać do porównań badawczych.",
                    {"tag": "warning"},
                )
            )
        else:
            scope = f"{series_id} / {scenario_id or 'brak scenariusza'}"
            rows.append(
                (
                    "Zakres porównania",
                    scope,
                    f"{len(cohort)} raportów",
                    "Gotowe" if scenario_id else "Niepełne",
                    "Porównujemy tylko raporty z tej samej serii i tego samego scenariusza.",
                    {"tag": "success" if scenario_id else "warning"},
                )
            )

        def index_for(candidate: MobileBenchmarkReport) -> dict[str, Any]:
            return self._report_experiment_index(candidate)

        field_defs = (
            (
                "Urządzenie",
                lambda candidate, index: _device_label(candidate),
                "Ten sam telefon ogranicza ryzyko, że porównujemy sprzęt zamiast modeli.",
                True,
            ),
            (
                "Android version",
                lambda candidate, index: self._guard_android_version_value(candidate, index),
                "Wersja Androida jest częścią środowiska pomiarowego i musi być jawnie kontrolowana.",
                True,
            ),
            (
                "Runtime",
                lambda candidate, index: str(candidate.runtime or "-"),
                "Runtime powinien być stały, jeśli badamy wpływ modelu albo pakietu.",
                True,
            ),
            (
                "Delegate",
                lambda candidate, index: str(candidate.delegate or "-"),
                "CPU/GPU/NNAPI wpływa na opóźnienia i zużycie energii.",
                True,
            ),
            (
                "Rozdzielczość wejścia",
                lambda candidate, index: self._guard_resolution_value(candidate, index),
                "Zmiana rozdzielczości zmienia koszt inferencji i jakość detekcji.",
                True,
            ),
            (
                "Build aplikacji",
                lambda candidate, index: self._guard_app_git_sha_value(candidate, index),
                "Commit/build klienta musi być jawny; brak app_git_sha traktujemy jako brak danych, nie jako wersję aplikacji.",
                True,
            ),
            (
                "Wersja aplikacji",
                lambda candidate, index: self._guard_app_version_value(candidate, index),
                "Wersja wydania jest pomocnicza i nie zastępuje identyfikatora buildu.",
                True,
            ),
            (
                "Profil rozpoznawania",
                lambda candidate, index: self._guard_profile_hash(index.get("recognition_profile") or _nested_value(candidate.raw, "recognition_profile", "profile")),
                "Profil progu, autozoomu i postprocessingu powinien być świadomą zmienną albo stałą.",
                True,
            ),
            (
                "MP fingerprint",
                lambda candidate, index: self._guard_model_fingerprint_value(candidate, index, "mp"),
                "Model pojazdów w pakiecie musi być ten sam, jeśli nie jest badaną zmienną.",
                True,
            ),
            (
                "MT fingerprint",
                lambda candidate, index: self._guard_model_fingerprint_value(candidate, index, "mt"),
                "Model tablic w pakiecie musi być ten sam, jeśli nie jest badaną zmienną.",
                True,
            ),
            (
                "MZ fingerprint",
                lambda candidate, index: self._guard_model_fingerprint_value(candidate, index, "mz"),
                "Model znaków w pakiecie musi być ten sam, jeśli nie jest badaną zmienną.",
                True,
            ),
            (
                "Trace klatek",
                lambda candidate, index: str(self._guard_artifact_count(index, "traces") or "-"),
                "Pełny ślad klatek jest podstawą wykresów opóźnień i stabilności.",
                False,
            ),
            (
                "Próbki/GT",
                lambda candidate, index: str(self._guard_artifact_count(index, "samples") or _nested_value(candidate.quality, "ground_truth_samples") or "-"),
                "Bez próbek z GT nie wolno mylić confidence z accuracy.",
                False,
            ),
        )

        for label, getter, description, require_same in field_defs:
            selected_value = str(getter(report, selected_index) or "-")
            values = [str(getter(candidate, index_for(candidate)) or "-") for candidate in cohort]
            non_empty_values = [value for value in values if value and value != "-"]
            distinct = sorted(set(non_empty_values))
            if selected_value in ("", "-"):
                status = "Brak danych"
                tag = "warning"
                series_value = "-"
            elif require_same and len(distinct) > 1:
                status = "Różne"
                tag = "error"
                series_value = f"{len(distinct)} wartości"
            elif require_same:
                status = "Spójne"
                tag = "success"
                series_value = selected_value
            else:
                status = "Jest" if selected_value not in ("", "-") else "Brak"
                tag = "success" if selected_value not in ("", "-") else "warning"
                series_value = f"{len(non_empty_values)}/{len(cohort)} raportów"
            rows.append((label, selected_value, series_value, status, description, {"tag": tag}))

        self.comparison_table.set_rows(rows)

    def _populate_config(self, report: MobileBenchmarkReport) -> None:
        raw = dict(report.raw or {})
        rows: list[tuple[str, str, str]] = []
        index = dict(getattr(report, "experiment_index", {}) or raw.get("desktop_experiment_index") or {})
        if index:
            for field, field_value, extra in _flatten_rows(index, prefix="Indeks eksperymentu", limit=28):
                rows.append((field, field_value, extra))
        sections = (
            ("capture", "Akwizycja obrazu"),
            ("recognition_profile", "Profil rozpoznawania"),
            ("normal_configuration", "Konfiguracja normalna"),
            ("experiment", "Eksperyment"),
            ("execution", "Modele MP/MT/MZ i runtime"),
            ("runtime_composition", "Złożenie runtime"),
            ("autotune_profiles", "Autotuning telefonu"),
        )
        for key, title in sections:
            value = raw.get(key)
            if value in (None, "", {}, []):
                continue
            for field, field_value, extra in _flatten_rows(value, prefix=title, limit=16):
                rows.append((field, field_value, extra))
        self.config_table.set_rows(rows or [("Brak konfiguracji", "-", "Raport nie zawiera tej sekcji.")])

    def _latency_rows_from_report(self, report: MobileBenchmarkReport) -> list[tuple[str, str, str, str, str, str, str, str, str, str]]:
        rows: list[tuple[str, str, str, str, str, str, str, str, str, str]] = []
        sources: list[tuple[str, Any]] = []
        latency = dict(report.latency or {})
        for key, label in (("mt", "MT inference"), ("mz", "MZ inference"), ("pipeline", "Pipeline")):
            if isinstance(latency.get(key), dict):
                sources.append((label, latency.get(key)))
        stages = _nested_value(report.raw, "summary.stages")
        if isinstance(stages, dict):
            for key, value in stages.items():
                if isinstance(value, dict):
                    sources.append((str(key), value))
        seen: set[str] = set()
        for label, data in sources:
            if label in seen:
                continue
            seen.add(label)
            rows.append(
                (
                    label,
                    str(_nested_value(data, "count") or "-"),
                    _format_ms(_nested_value(data, "mean_ms", "mean")),
                    _format_ms(_nested_value(data, "p50_ms", "median_ms", "median")),
                    _format_ms(_nested_value(data, "p90_ms", "p90")),
                    _format_ms(_nested_value(data, "p95_ms", "p95")),
                    _format_ms(_nested_value(data, "p99_ms", "p99")),
                    _format_ms(_nested_value(data, "min_ms", "min")),
                    _format_ms(_nested_value(data, "max_ms", "max")),
                    _format_ms(_nested_value(data, "stddev_ms", "standard_deviation", "std")),
                )
            )
        return rows

    def _populate_latency(self, report: MobileBenchmarkReport, bundle: MobileReportBundle | None) -> None:
        rows = self._latency_rows_from_report(report)
        if not rows and bundle and bundle.trace_rows:
            for column in ("total_ms", "plate_inference_ms", "character_inference_ms", "vehicle_inference_ms"):
                values = [_safe_float(row.get(column), None) for row in bundle.trace_rows]
                values = [value for value in values if value is not None]
                if not values:
                    continue
                rows.append(
                    (
                        column,
                        str(len(values)),
                        _format_ms(statistics.mean(values)),
                        _format_ms(statistics.median(values)),
                        "-",
                        _format_ms(sorted(values)[max(0, min(len(values) - 1, int(len(values) * 0.95) - 1))]),
                        "-",
                        _format_ms(min(values)),
                        _format_ms(max(values)),
                        _format_ms(statistics.pstdev(values) if len(values) > 1 else 0),
                    )
                )
        self.latency_table.set_rows(rows or [("Brak danych", "-", "-", "-", "-", "-", "-", "-", "-", "-")])
        self._draw_latency_chart()

    def _populate_artifacts(self, report: MobileBenchmarkReport, bundle: MobileReportBundle | None) -> None:
        index = dict((bundle.experiment_session if bundle else getattr(report, "experiment_index", {})) or {})
        counts = dict(index.get("artifact_counts") or {})
        flags = dict(index.get("artifact_flags") or {})

        def count_value(name: str, fallback: int = 0) -> int:
            return _safe_int(counts.get(name), fallback)

        if bundle:
            values = {
                "traces": (bundle.trace_total, len(bundle.trace_rows)),
                "thermal": (bundle.thermal_total, len(bundle.thermal_rows)),
                "frame_flow": (bundle.frame_flow_total, len(bundle.frame_flow_rows)),
                "events": (bundle.event_total, len(bundle.event_rows)),
                "samples": (bundle.sample_total, len(bundle.sample_rows)),
                "crops": (bundle.crop_count, bundle.crop_count),
                "annotations": (bundle.annotation_count, bundle.annotation_count),
                "log": (1 if bundle.log_preview else 0, 1 if bundle.log_preview else 0),
            }
        else:
            values = {
                "traces": (count_value("traces"), -1),
                "thermal": (count_value("thermal"), -1),
                "frame_flow": (count_value("frame_flow"), -1),
                "events": (count_value("events"), -1),
                "samples": (count_value("samples"), -1),
                "crops": (count_value("crops"), count_value("crops")),
                "annotations": (count_value("annotations"), count_value("annotations")),
                "log": (1 if flags.get("has_log") else 0, 1 if flags.get("has_log") else 0),
            }

        descriptions = {
            "traces": "Czasy i statusy przetwarzania klatek; preview nie może ograniczać przyszłych obliczeń serii.",
            "thermal": "Szereg termiczny telefonu, potrzebny do wykresów throttlingu i stabilności pomiaru.",
            "frame_flow": "Przepływ klatek: otrzymane, przetworzone i świadomie pominięte klatki.",
            "events": "Zdarzenia wysokiego poziomu: consensus, autozoom, lock, zmiany tracków.",
            "samples": "Indeks próbek/cropów używany do jakości, GT i analizy błędów.",
            "crops": "Liczba obrazów cropów w archiwum; same obrazy nie są ładowane automatycznie.",
            "annotations": "Adnotacje próbek, zwykle JSONL, używane do odtworzenia jakości.",
            "log": "Log aplikacji mobilnej jako pomoc diagnostyczna.",
        }
        labels = {
            "traces": "Trace klatek",
            "thermal": "Termika",
            "frame_flow": "Przepływ klatek",
            "events": "Eventy",
            "samples": "Próbki",
            "crops": "Cropy",
            "annotations": "Adnotacje próbek",
            "log": "Log aplikacji",
        }
        rows: list[tuple[Any, ...]] = []
        for key in ("traces", "thermal", "frame_flow", "events", "samples", "crops", "annotations", "log"):
            source_total, preview_total = values.get(key, (0, 0))
            source_total = _safe_int(source_total)
            preview_total = _safe_int(preview_total)
            if source_total:
                if key in {"crops", "annotations", "log"}:
                    preview_text = "-"
                    status = "Jest"
                elif preview_total < 0:
                    preview_text = "zapisany"
                    status = "Jest"
                else:
                    preview_text = str(preview_total)
                    status = "Preview ucięty" if preview_total and source_total > preview_total else "Jest"
            else:
                preview_text = "-"
                status = "Brak"
            rows.append(
                (
                    labels[key],
                    str(source_total or "-"),
                    preview_text,
                    status,
                    descriptions[key],
                    {"tag": "success" if source_total else "muted"},
                )
            )
        self.artifacts_table.set_rows(rows)

    def _populate_quality(self, report: MobileBenchmarkReport) -> None:
        quality = dict(report.quality or {})
        if quality.get("quality_source") == "human_review":
            def ratio(key):
                value = quality.get(key)
                return "—" if value is None else f"{float(value) * 100:.2f}%"
            self.quality_table.set_rows([
                ("Źródło jakości", "Weryfikacja człowieka", "Ukończona weryfikacja; surowe pomiary czasu i pamięci zachowane."),
                ("Tryb weryfikacji", "Zaślepiona GT" if quality.get("review_mode") == "blinded_gt_v1" else "Asystowana", "Tryb zapisany w sidecarze weryfikacji."),
                ("Poprawne odczyty MZ", ratio("exact_read_rate"), "Jednostka: oceniany crop."),
                ("Brak odczytu MZ", ratio("no_read_rate"), "Oceniany crop z GT, bez użytecznego odczytu."),
                ("CER MZ", ratio("cer"), "Suma błędów / suma znaków GT. Wartość może przekraczać 100%."),
                ("Skuteczność ALPR dla tablic", ratio("subject_success_rate"), "Co najmniej jeden poprawny odczyt lub wynik konsensusu."),
                ("Skuteczność lokalizacji MT", ratio("mt_localization_success_rate"), "Ocenione wywołania z jedną widoczną tablicą; nie jest to mAP."),
                ("SHA-256 źródła", quality.get("source_archive_sha256", ""), "Tożsamość archiwum powiązanego z weryfikacją."),
            ])
            return
        if not quality:
            self.quality_table.set_rows([("Brak jakości", "-", "Raport nie zawiera sekcji quality.")])
            return
        available = quality.get("available")
        rows: list[tuple[Any, ...]] = [
            (
                "Ground truth",
                "dostępny" if available else "brak",
                str(quality.get("reason") or "Jakość liczona tylko dla accepted/corrected z ground truth."),
                {"tag": "success" if available else "warning"},
            ),
            ("Jednostka jakości", str(quality.get("unit") or "-"), "Nie liczymy kilku klatek tego samego tracku jako kilku prób."),
            ("Próbki GT", str(quality.get("ground_truth_samples") or 0), "Liczba unikalnych próbek z transkrypcją."),
            ("Exact match", _format_percent(quality.get("exact_match_rate")), "Odsetek pełnych trafień po normalizacji."),
            ("CER", "—" if _safe_float(quality.get("cer")) is None else f"{float(quality['cer']) * 100:.1f}%", "Suma błędów / znaki GT; wartość może przekraczać 100%."),
            (
                "Śr. odległość edycyjna",
                _format_number(quality.get("normalized_edit_distance_mean"), digits=4),
                "Uśredniona odległość Levenshteina względem długości napisu.",
            ),
            ("Zaakceptowane bez korekty", _format_percent(quality.get("accepted_original_rate")), "Udział zaakceptowanych predykcji wśród przejrzanych."),
            ("Przejrzane", str(quality.get("reviewed_samples") or 0), "accepted, corrected i rejected."),
            ("Nieprzejrzane", str(quality.get("not_reviewed_samples") or 0), "Nie używamy ich do exact match/CER."),
        ]
        samples = quality.get("samples")
        if isinstance(samples, list) and samples:
            for sample in samples[:8]:
                if not isinstance(sample, dict):
                    continue
                rows.append(
                    (
                        f"Próbka {sample.get('capture_id') or sample.get('track_id') or '-'}",
                        "OK" if sample.get("exact_match") else "NOK",
                        f"{sample.get('prediction', '')} -> {sample.get('ground_truth', '')}; edit={sample.get('edit_distance', '-')}",
                        {"tag": "success" if sample.get("exact_match") else "error"},
                    )
                )
        self.quality_table.set_rows(rows)

    def _populate_diagnostics(self, report: MobileBenchmarkReport, bundle: MobileReportBundle | None) -> None:
        rows: list[tuple[str, str, str]] = []
        for field, value, extra in _flatten_rows(report.errors, prefix="errors", limit=24):
            rows.append((field, value, extra))
        counters = _nested_value(report.raw, "summary.counters")
        for field, value, extra in _flatten_rows(counters, prefix="counters", limit=28):
            rows.append((field, value, extra))
        statuses = _nested_value(report.raw, "summary.statuses")
        for field, value, extra in _flatten_rows(statuses, prefix="statuses", limit=18):
            rows.append((field, value, extra))
        for field, value, extra in _flatten_rows(report.memory, prefix="memory", limit=12):
            rows.append((field, value, extra))
        self.diagnostics_table.set_rows(rows or [("Brak diagnostyki", "-", "Raport nie zawiera błędów ani liczników.")])
        self._replace_text(
            self.log_text,
            (bundle.log_preview if bundle and bundle.log_preview else "Brak application.log w paczce albo raport został zaimportowany z gołego JSON-a."),
        )

    def _populate_crops(self, report: MobileBenchmarkReport, bundle: MobileReportBundle | None) -> None:
        rows: list[tuple[str, str, str, str, str, str, str]] = []
        if bundle and bundle.sample_rows:
            for row in bundle.sample_rows[:500]:
                rows.append(
                    (
                        row.get("capture_id") or row.get("id") or "-",
                        row.get("track_id") or "-",
                        row.get("verification_status") or row.get("status") or "-",
                        row.get("prediction") or row.get("text") or "-",
                        row.get("ground_truth") or row.get("ground_truth_text") or "-",
                        row.get("captured_at_ms") or row.get("timestamp_ms") or "-",
                        row.get("crop") or row.get("path") or "-",
                    )
                )
        records = _nested_value(report.raw, "crop_session.records")
        if not rows and isinstance(records, list):
            for record in records[:500]:
                if not isinstance(record, dict):
                    continue
                verification = record.get("human_verification") if isinstance(record.get("human_verification"), dict) else {}
                rows.append(
                    (
                        str(record.get("capture_id") or "-"),
                        str(record.get("track_id") or "-"),
                        str(verification.get("status") or record.get("verification_status") or "-"),
                        str(record.get("text") or verification.get("original_prediction") or "-"),
                        str(verification.get("ground_truth_text") or "-"),
                        str(record.get("captured_at_ms") or "-"),
                        str(record.get("image") or "-"),
                    )
                )
        if not rows:
            rows = [("-", "-", "-", "-", "-", "-", "Brak indeksu cropów w raporcie.")]
        self.crops_table.set_rows(rows)

    def _populate_raw(self, report: MobileBenchmarkReport, bundle: MobileReportBundle | None) -> None:
        payload = {
            "note": "Podgląd jest celowo ograniczony, żeby przeglądarka raportów pozostawała responsywna. Pełny artefakt pozostaje w pliku źródłowym raportu.",
            "bundle": _mobile_report_bundle_preview_dict(bundle),
            "report": _mobile_report_preview_dict(report),
        }
        try:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            text = str(payload)
        if len(text) > MOBILE_REPORT_RAW_TEXT_LIMIT:
            text = (
                text[:MOBILE_REPORT_RAW_TEXT_LIMIT].rstrip()
                + f"\n\n... podgląd ucięty po {MOBILE_REPORT_RAW_TEXT_LIMIT} znakach, aby nie blokować interfejsu ..."
            )
        self._replace_text(self.raw_text, text)

    def _replace_text(self, widget: tk.Text, text: str) -> None:
        try:
            widget.configure(state=tk.NORMAL)
            widget.delete("1.0", tk.END)
            widget.insert("1.0", text or "")
            widget.configure(state=tk.DISABLED)
        except Exception:
            pass

    def _draw_latency_chart(self) -> None:
        canvas = getattr(self, "latency_canvas", None)
        if canvas is None:
            return
        try:
            width = max(320, int(canvas.winfo_width() or 760))
            height = max(150, int(canvas.winfo_height() or 180))
            canvas.delete("all")
            canvas.create_rectangle(0, 0, width, height, fill=self.panel, outline=self.border)
            bundle = self.current_bundle
            if not bundle or not bundle.trace_rows:
                canvas.create_text(
                    width / 2,
                    height / 2,
                    text="Brak śladu klatkowego traces.csv do wykresu.",
                    fill=self.muted,
                    font=("Segoe UI", 10, "bold"),
                )
                return
            series_specs = (
                ("total_ms", "pipeline", self.accent),
                ("plate_inference_ms", "MT", self.warning),
                ("character_inference_ms", "MZ", self.success),
                ("vehicle_inference_ms", "MP", "#2fffe6"),
            )
            series: list[tuple[str, list[float], str]] = []
            max_value = 0.0
            for column, label, color in series_specs:
                values = [_safe_float(row.get(column), None) for row in bundle.trace_rows]
                values = [float(value) for value in values if value is not None]
                if not values:
                    continue
                series.append((label, values, color))
                max_value = max(max_value, max(values))
            if not series:
                canvas.create_text(width / 2, height / 2, text="W traces.csv nie znaleziono kolumn *_ms.", fill=self.muted)
                return
            pad_l, pad_r, pad_t, pad_b = 42, 18, 22, 28
            plot_w = max(1, width - pad_l - pad_r)
            plot_h = max(1, height - pad_t - pad_b)
            max_value = max(1.0, max_value)
            for index in range(5):
                y = pad_t + plot_h * index / 4.0
                value = max_value * (1.0 - index / 4.0)
                canvas.create_line(pad_l, y, width - pad_r, y, fill=blend_hex_colors(self.panel, self.fg, 0.12))
                canvas.create_text(8, y, text=f"{value:.0f}", fill=self.muted, font=("Segoe UI", 7), anchor=tk.W)
            canvas.create_text(pad_l, height - 13, text="klatki", fill=self.muted, font=("Segoe UI", 8), anchor=tk.W)
            canvas.create_text(8, 10, text="ms", fill=self.muted, font=("Segoe UI", 8), anchor=tk.W)
            for s_index, (label, values, color) in enumerate(series):
                step = max(1, int(len(values) / max(1, plot_w)))
                sampled = values[::step]
                if len(sampled) < 2:
                    continue
                points: list[float] = []
                for idx, value in enumerate(sampled):
                    x = pad_l + (plot_w * idx / max(1, len(sampled) - 1))
                    y = pad_t + plot_h - (plot_h * max(0.0, min(max_value, value)) / max_value)
                    points.extend([x, y])
                canvas.create_line(*points, fill=color, width=2, smooth=True)
                lx = width - pad_r - 240 + s_index * 58
                canvas.create_rectangle(lx, 7, lx + 10, 17, fill=color, outline="")
                canvas.create_text(lx + 14, 12, text=label, fill=self.fg, font=("Segoe UI", 8), anchor=tk.W)
        except Exception:
            pass


def open_mobile_report_browser(owner, parent=None):
    try:
        existing = getattr(owner, "_mobile_report_browser_dialog", None)
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return existing
    except Exception:
        try:
            setattr(owner, "_mobile_report_browser_dialog", None)
        except Exception:
            pass
    browser = MobileReportBrowser(owner, parent=parent)
    return browser.window


__all__ = ["open_mobile_report_browser", "MobileReportBrowser"]
