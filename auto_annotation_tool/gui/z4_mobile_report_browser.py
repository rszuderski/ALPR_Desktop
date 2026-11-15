#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Desktop browser for Android mobile ALPR benchmark reports."""

from __future__ import annotations

import json
import hashlib
import statistics
import threading
import queue
from dataclasses import replace
from tkinter import filedialog
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont

from ..config import CONFIG, SESSION, logger
from ..ranking import (
    MobileBenchmarkReport,
    MobilePackageExperimentStore,
    MobileReportBundle,
    read_mobile_report_bundles,
    score_mobile_report,
    read_mobile_sample_image,
)
from ..ranking.mobile_package_experiments import (
    plan_mobile_session_import, mobile_import_path_error, validate_known_mobile_experiment, F5A_IMPORT_GROUP,
)
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .mobile_report_plot_axis import axis_ticks, pixel_column_extrema
from .mobile_report_x_zoom import clamp_x_view, zoom_x_view, pan_x_view, visible_xy
from .mobile_report_sparse_axis import (
    display_positions, has_sparse_frame_gap, source_id_ticks, viewport_counts,
)
from .mobile_report_review_overlay import project_desktop_samples
from .mobile_report_chart_styles import LATENCY_SERIES_STYLES, draw_chart_marker
from .mobile_compare_review_gate import imported_verification
from . import mobile_import_preferences
from .z4_mobile_report_browser_ux import (desktop_status, matches_report_filter, report_identity, selected_report_caption)
from .mobile_report_operator_metadata import (
    phone_session_metadata, phone_name_caption, phone_note_caption,
    phone_list_caption, preserve_phone_notes,
)
from .z4_mobile_report_browser_ux import (SORT_COLUMNS, report_measurement_time, report_repeat,
    report_view_key, report_sort_value, sorted_report_view, report_sort_preference, save_report_sort_preference)
from .mobile_report_science_dashboard import ScientificDashboard, build_overview, science_from_verified_bundle
from .mobile_report_scientific_comparability import assess_cohort
# Z4MOBREP019_NO_UNGROUNDED_SCIENTIFIC_COMPARABILITY_20261018
from .mobile_report_measurement_integrity import (
    timeline_stages, continuous_segments, stage_enabled, model_input_sizes,
    local_measurement_time, crash_description,
)
from .mobile_report_session_timeline import (
    session_time_context, session_elapsed_seconds, session_caption,
    time_ticks, point_caption, STAGE_DESCRIPTIONS, STAGE_SHORT,
)
# Z4MOBREP018_EVIDENCE_FAITHFUL_DISPLAY_20261018
from .mobile_report_source_recovery import (restore_source, stored_source, relinked_report, crop_entry_for_row)
from .z4_mobile_report_cards_ux import (HoverCardHelp, cards_for_report, device_details, summary_measure_rows, verified_frame_snapshot)
# Z4MOBREP014_ANDROID_GATE_PERSISTENT_COUNTERS_20261017
# Z4MOBREP013_EXPLAINED_METRIC_CARDS_20261017
# Z4MOBREP012_READABLE_MOBILE_REPORTS_20261017

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
    """Explicit local timestamp, never stripping the UTC offset by truncation."""
    return local_measurement_time(value)


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
        # A measured zero/False must never become a missing-data dash.
        text = "" if value is None else str(value)
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


def _compact_mobile_report_for_store(report: MobileBenchmarkReport, bundle: MobileReportBundle | None = None) -> MobileBenchmarkReport:
    try:
        compact_raw = _compact_mobile_report_payload(
            report.raw,
            list_limit=MOBILE_REPORT_RAW_STORAGE_LIST_LIMIT,
            text_limit=MOBILE_REPORT_RAW_STORAGE_TEXT_LIMIT,
        )
        if isinstance(report.raw, dict) and isinstance(compact_raw, dict):
            preserve_phone_notes(report.raw, compact_raw)
        # Cache tylko z bieżącej, zweryfikowanej paczki, z przypiętym SHA.
        # Nie zapisujemy do oryginalnego archiwum ani do weryfikacji GT.
        if bundle is not None:
            snapshot = verified_frame_snapshot(report, bundle)
            if snapshot and isinstance(compact_raw, dict):
                compact_raw = dict(compact_raw)
                compact_raw['_desktop_frame_flow_totals_v1'] = snapshot
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



def _report_wheel_scroll_units(event) -> int:
    """One vertical wheel direction for Windows, Linux and small trackpad deltas.

    This is a UI input normalization only. It never changes report measurements.
    """
    try:
        if int(getattr(event, "state", 0) or 0) & 0x0004:  # Ctrl+wheel is not page scroll
            return 0
        try:
            button = int(getattr(event, "num", 0) or 0)
        except (TypeError, ValueError):  # Tk uses num="??" on MouseWheel
            button = 0
        if button == 4:
            return -3
        if button == 5:
            return 3
        delta = int(getattr(event, "delta", 0) or 0)
        if not delta:
            return 0
        if abs(delta) < 120:
            return -1 if delta > 0 else 1
        ticks = max(1, min(6, abs(delta) // 120))
        return (-3 if delta > 0 else 3) * ticks
    except (TypeError, ValueError, OverflowError):
        return 0


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
        selectmode: str = "browse",
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
                rowheight=30,
                font=("Segoe UI", 9),
                borderwidth=0,
            )
            style.map(
                style_name,
                background=[("selected", blend_hex_colors(palette.get("accent", "#4f8de3"), palette.get("panel", "#252526"), 0.25))],
                foreground=[("selected", palette.get("fg", "#f3f3f3"))],
            )
        except Exception:
            pass
        self.tree = ttk.Treeview(
            self.shell,
            columns=columns,
            show="headings",
            height=height,
            selectmode=selectmode,
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
                font=("Segoe UI", 9, "bold"),
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
                    font=("Segoe UI", 9),
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


class _MobileFolderImportFlow:
    """Folder phases hosted by the existing picker; workers only publish queue events."""

    def __init__(self, browser, dialog, selection, close):
        self.browser, self.dialog, self.selection, self.close_callback = browser, dialog, selection, close
        self.events = queue.Queue()
        self.cancel_event = threading.Event()
        self.phase, self.plan, self.closed, self.poll_id = "selection", None, False, None
        self.page = tk.Frame(dialog, bg=browser.bg, padx=14, pady=12)
        self.page.columnconfigure(0, weight=1)
        self.page.rowconfigure(2, weight=1)
        self.title = tk.StringVar(value="")
        self.location = tk.StringVar(value="")
        self.note = tk.StringVar(value="")
        tk.Label(self.page, textvariable=self.title, bg=browser.bg, fg=browser.fg,
                 font=("Segoe UI", 14, "bold"), anchor=tk.W).grid(row=0, column=0, sticky="ew")
        self.path_label = tk.Label(self.page, textvariable=self.location, bg=browser.bg, fg=browser.muted,
                                  width=1, height=2, anchor=tk.W, justify=tk.LEFT)
        self.path_label.grid(row=1, column=0, sticky="ew", pady=(3, 6))
        self.work = tk.Frame(self.page, bg=browser.bg)
        self.work.columnconfigure(0, weight=1)
        self.progress_text = tk.StringVar(value="Wyszukiwanie archiwów…")
        self.current_file = tk.StringVar(value="")
        tk.Label(self.work, textvariable=self.progress_text, bg=browser.bg, fg=browser.fg,
                 font=("Segoe UI", 12)).grid(row=0, column=0, sticky="ew", pady=(24, 12))
        self.progress = ttk.Progressbar(self.work, mode="indeterminate")
        self.progress.grid(row=1, column=0, sticky="ew", padx=20, pady=12)
        self.current_label = tk.Label(self.work, textvariable=self.current_file, width=1, height=3,
                                     bg=browser.bg, fg=browser.muted, justify=tk.LEFT, anchor=tk.W)
        self.current_label.grid(row=2, column=0, sticky="ew", padx=20)
        self.result = tk.Frame(self.page, bg=browser.bg)
        self.result.columnconfigure(0, weight=1)
        self.result.rowconfigure(2, weight=1)
        self.result.rowconfigure(3, weight=1)
        self.stats = tk.Frame(self.result, bg=browser.bg)
        self.stats.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.stats.columnconfigure(0, weight=1)
        self.stats.columnconfigure(1, weight=1)
        self.stats_vars = []
        for n in range(8):
            variable = tk.StringVar()
            self.stats_vars.append(variable)
            tk.Label(self.stats, textvariable=variable, width=1, anchor=tk.W, bg=browser.bg,
                     fg=browser.fg, font=("Segoe UI", 10, "bold" if n == 7 else "normal")).grid(
                         row=n // 2, column=n % 2, sticky="ew", padx=(0, 8))
        self.group_box = tk.Frame(self.result, bg=browser.bg)
        self.group_box.columnconfigure(0, weight=1)
        self.group_tree = ttk.Treeview(self.group_box, columns=("new", "existing", "conflicts"), height=3,
                                      selectmode="browse")
        self.group_tree.heading("#0", text="Serie rozpoznane z metadanych")
        self.group_tree.column("#0", width=380, minwidth=120)
        for key, label in (("new", "Nowe"), ("existing", "Istniejące"), ("conflicts", "Niezgodne")):
            self.group_tree.heading(key, text=label)
            self.group_tree.column(key, width=85, minwidth=70, stretch=False, anchor=tk.CENTER)
        self.group_tree.grid(row=0, column=0, sticky="ew")
        self.group_button = ttk.Button(self.group_box, text="Wybierz zaznaczoną serię do importu", command=self.choose_group)
        self.group_button.grid(row=1, column=0, sticky="e", pady=4)
        files_box = tk.Frame(self.result, bg=browser.bg)
        files_box.grid(row=2, column=0, sticky="nsew")
        files_box.rowconfigure(0, weight=1)
        files_box.columnconfigure(0, weight=1)
        self.files = ttk.Treeview(files_box, columns=("identity",), height=7, selectmode="browse")
        self.files.heading("#0", text="Grupy wyników / archiwum")
        self.files.heading("identity", text="Wariant / powtórzenie / blok")
        self.files.column("#0", width=440, minwidth=160)
        self.files.column("identity", width=220, minwidth=120)
        self.files.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(files_box, command=self.files.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.files.configure(yscrollcommand=scroll.set)
        self.files.bind("<<TreeviewSelect>>", self.show_file)
        detail_box = tk.Frame(self.result, bg=browser.bg)
        detail_box.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
        detail_box.columnconfigure(0, weight=1)
        detail_box.rowconfigure(0, weight=1)
        self.details = tk.Text(detail_box, width=1, height=6, wrap=tk.WORD, font=("Segoe UI", 10),
                               bg=browser.panel, fg=browser.fg, relief=tk.FLAT, padx=8, pady=6, state=tk.DISABLED)
        self.details.grid(row=0, column=0, sticky="nsew")
        detail_scroll = ttk.Scrollbar(detail_box, command=self.details.yview)
        detail_scroll.grid(row=0, column=1, sticky="ns")
        self.details.configure(yscrollcommand=detail_scroll.set)
        self.note_label = tk.Label(self.page, textvariable=self.note, width=1, height=2, anchor=tk.W,
                                  justify=tk.LEFT, bg=browser.bg, fg=browser.muted)
        self.note_label.grid(row=3, column=0, sticky="ew", pady=(5, 3))
        self.actions = tk.Frame(self.page, bg=browser.bg)
        self.actions.grid(row=4, column=0, sticky="ew")
        self.back_button = ttk.Button(self.actions, text="Wróć do wyboru folderu", command=self.back)
        self.copy_button = ttk.Button(self.actions, text="Kopiuj podsumowanie", command=self.copy_summary)
        self.cancel_button = ttk.Button(self.actions, text="Anuluj", command=self.cancel_or_close)
        self.import_button = ttk.Button(self.actions, command=self.commit)
        self.page.bind("<Configure>", self.resize, add="+")
        dialog.bind("<Destroy>", self.dispose, add="+")
        dialog._mobile_import_flow = self

    @property
    def busy(self):
        return self.phase in {"scanning", "importing"}

    def resize(self, event):
        width = max(260, event.width - 30)
        for label in (self.path_label, self.current_label, self.note_label):
            label.configure(wraplength=width)

    def dispose(self, event):
        if event.widget is not self.dialog:
            return
        self.closed = True
        self.cancel_event.set()
        if self.poll_id is not None:
            try:
                self.dialog.after_cancel(self.poll_id)
            except tk.TclError:
                pass
        # Release Tcl variables on the UI thread. Otherwise a later worker's
        # cyclic GC can invoke Variable.__del__ against a destroyed interpreter.
        self.stats_vars.clear()
        for name in ("title", "location", "note", "progress_text", "current_file"):
            setattr(self, name, None)
        self.dialog._mobile_import_flow = None
        self.close_callback = None

    def request_close(self):
        if self.phase == "importing":
            self.cancel_event.set()
            self.note.set("Przerywam zapis. Za chwilę pokażę faktyczną liczbę zapisanych sesji.")
            return False
        self.cancel_event.set()
        return True

    def cancel_or_close(self):
        if self.busy:
            self.cancel_event.set()
            self.note.set("Anulowanie po bieżącym archiwum. Zapisane wcześniej sesje pozostają w magazynie.")
            self.cancel_button.configure(state=tk.DISABLED)
        else:
            self.close_callback()

    def _working(self, phase):
        self.phase = phase
        self.cancel_event = threading.Event()
        self.selection.pack_forget()
        self.page.pack(fill=tk.BOTH, expand=True)
        self.result.grid_remove()
        self.work.grid(row=2, column=0, sticky="nsew")
        for button in self.actions.winfo_children():
            button.pack_forget()
        self.cancel_button.configure(text="Anuluj sprawdzanie" if phase == "scanning" else "Przerwij import", state=tk.NORMAL)
        self.cancel_button.pack(side=tk.RIGHT)
        self.title.set("Sprawdzanie archiwów" if phase == "scanning" else "Importowanie sesji")
        self.progress_text.set("Wyszukiwanie archiwów…" if phase == "scanning" else f"Importowanie 0 z {len(self.plan.eligible_items)} sesji")
        self.current_file.set("")
        self.note.set("Sprawdzanie nie zapisuje raportów." if phase == "scanning" else "Sesje są zapisywane kolejno. Przerwanie nie cofa wcześniejszych zapisów.")
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(14)

    def _launch(self, operation):
        # No Tk/after callbacks, widget access or browser UI calls in this worker.
        events, cancelled = self.events, self.cancel_event.is_set
        def worker():
            try:
                result = operation(lambda done, total, path: events.put(("progress", (done, total, str(path)))), cancelled)
                events.put(("result", result))
            except Exception as exc:
                events.put(("error", str(exc)))
        threading.Thread(target=worker, name="mobile-folder-" + self.phase, daemon=True).start()
        self.poll_id = self.dialog.after(40, self.poll)

    def scan(self, root):
        if self.busy:
            return
        self.browser._folder_import_handled = True
        self.plan = None
        root = Path(root)
        self.location.set(str(root))
        existing = tuple(self.browser.store.reports)
        self._working("scanning")
        self._launch(lambda progress, cancelled: plan_mobile_session_import(
            root, existing, progress=progress, cancelled=cancelled, prepare_report=_compact_mobile_report_for_store))

    def commit(self):
        if self.busy or self.plan is None or not self.plan.eligible_items:
            return
        store, plan = self.browser.store, self.plan
        self._working("importing")
        self._launch(lambda progress, cancelled: store.import_mobile_session_plan(plan, progress=progress, cancelled=cancelled))

    def poll(self):
        self.poll_id = None
        if self.closed:
            return
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                done, total, path = value
                if total:
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=total, value=done)
                    self.progress_text.set(f"Sprawdzono {done} z {total} archiwów" if self.phase == "scanning" else f"Importowanie {done} z {total} sesji")
                else:
                    self.progress_text.set("Wyszukiwanie archiwów…")
                self.current_file.set(("Sprawdzanie: " if self.phase == "scanning" else "Zapis: ") + Path(path).name)
            elif kind == "result":
                writing = self.phase == "importing"
                self.plan = value
                if writing:
                    imported = [i for i in value.items if i.status == "IMPORTED"]
                    if imported:
                        mobile_import_preferences.remember_successful_import([i.path for i in imported], folder=value.root)
                        self.browser._folder_import_saved_id = imported[-1].report.report_id
                self.show_result()
            else:
                self.progress.stop()
                self.phase = "error"
                self.title.set("Nie udało się sprawdzić folderu")
                self.progress_text.set(value)
                self.current_file.set("")
                self.cancel_button.configure(text="Anuluj", state=tk.NORMAL)
                self.back_button.pack(side=tk.LEFT)
        if self.busy:
            self.poll_id = self.dialog.after(40, self.poll)

    def show_result(self):
        self.progress.stop()
        self.phase = "finished" if self.plan.finished else "result"
        self.title.set("Wynik importu" if self.plan.finished else "Wynik sprawdzenia")
        self.work.grid_remove()
        self.result.grid(row=2, column=0, sticky="nsew")
        p = self.plan
        other = [i for i in p.items if i.group is not None and i.group != p.selected_group]
        for variable, value in zip(self.stats_vars, p.overview()):
            variable.set(value)
        self.location.set(f"{p.root}\nSeria główna / wybrana: {p.group_caption(p.selected_group)}")
        self.group_tree.delete(*self.group_tree.get_children())
        self.group_keys = {}
        for n, (group, items) in enumerate(p.groups.items()):
            iid = f"g{n}"
            self.group_keys[iid] = group
            self.group_tree.insert("", tk.END, iid=iid, text=p.group_caption(group), values=(
                sum(i.status == "NEW" for i in items), sum(i.status == "EXISTING" for i in items), sum(i.status == "CONFLICT" for i in items)))
            if group == p.selected_group:
                self.group_tree.selection_set(iid)
        if len(p.groups) > 1:
            self.group_box.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        else:
            self.group_box.grid_remove()
        self.files.delete(*self.files.get_children())
        labels = {"good": "Zgodne z protokołem" if p.selected_group == F5A_IMPORT_GROUP else "Zakwalifikowane — kontrola ogólna",
                  "bad": "Niezgodne z protokołem", "other": "Pilotaże / inne serie",
                  "existing": "Już zaimportowane", "invalid": "Uszkodzone lub nieczytelne", "duplicate": "Duplikaty"}
        self.file_items = {}
        for category, label in labels.items():
            self.files.insert("", tk.END, iid=category, text=label, open=category == "bad")
        for n, item in enumerate(p.items):
            # These are overlapping views, not disjoint populations. An existing
            # pilot must remain visible in both its series and existing reports.
            categories = []
            if item.status == "INVALID": categories.append("invalid")
            if item.status in {"CONFLICT", "IDENTITY_CONFLICT", "ERROR"}: categories.append("bad")
            if item.group is not None and item.group != p.selected_group: categories.append("other")
            if item.status in {"EXISTING", "IMPORTED"}: categories.append("existing")
            if item.status == "DUPLICATE": categories.append("duplicate")
            if item.group == p.selected_group and item.status in {"NEW", "EXISTING", "IMPORTED"}: categories.append("good")
            for category in categories:
                iid = f"file{n}-{category}"
                self.file_items[iid] = item
                self.files.insert(category, tk.END, iid=iid, text=item.path.name,
                                  values=(" / ".join(item.observed(k) for k in ("variant", "replicate_index", "block")),))
        for category, label in labels.items():
            self.files.item(category, text=f"{label} ({len(self.files.get_children(category))})")
        self.set_details("Wybierz archiwum, aby zobaczyć jego tożsamość i szczegóły kwalifikacji.\nGrupy są filtrami: pilot może być widoczny także w już zaimportowanych.\n\n" + "\n".join(
            f"MT — {label}: {p.mt_counts(items)[0]} archiwów z niekompletnymi dowodami; {sum(p.mt_counts(items)[1].values())} przerwanych wywołań"
            for label, items in (("cały skan", p.items), ("właściwa seria F5A", [i for i in p.items if i.is_f5a]), ("inne serie", [i for i in p.items if not i.is_f5a]))))
        for button in self.actions.winfo_children():
            button.pack_forget()
        self.back_button.pack(side=tk.LEFT)
        self.copy_button.pack(side=tk.LEFT, padx=6)
        self.cancel_button.configure(text="Anuluj", state=tk.NORMAL)
        self.cancel_button.pack(side=tk.RIGHT)
        self.import_button.configure(text=p.import_caption, state=tk.NORMAL if p.eligible_items else tk.DISABLED)
        self.import_button.pack(side=tk.RIGHT, padx=6)
        if p.finished:
            self.note.set(f"Zaimportowano: {p.count('IMPORTED')} | Już istniały: {p.count('EXISTING')} | Nie zaimportowano: {len(p.items)-p.count('IMPORTED')-p.count('EXISTING')} | Błędy: {p.count('ERROR')+p.count('INVALID')+p.count('IDENTITY_CONFLICT')}" + (". Zapis przerwany; wcześniejsze zapisy zachowano." if p.cancelled else ""))
        elif p.cancelled:
            self.note.set("Sprawdzanie anulowane. Wynik jest częściowy; wróć do wyboru folderu, aby sprawdzić ponownie.")
        elif p.selected_group is None and p.groups:
            self.note.set("Wybierz jawnie serię w tabeli i zatwierdź przyciskiem wyboru grupy. Nic nie zapisano.")
        elif other:
            self.note.set(f"Pominięto {len(other)} sesje pilotażowe / innych serii. Można jawnie wybrać ich grupę. Nic nie zapisano.")
        elif not p.eligible_items:
            self.note.set("Brak nowych zgodnych sesji do zapisu. Sprawdź grupy błędów i już zaimportowanych archiwów.")
        else:
            self.note.set("Sprawdzenie zakończone. Dopiero przycisk importu zapisze zgodne sesje wybranej serii.")

    def set_details(self, text):
        self.details.configure(state=tk.NORMAL)
        self.details.delete("1.0", tk.END)
        self.details.insert("1.0", text)
        self.details.configure(state=tk.DISABLED)

    def show_file(self, event=None):
        selected = self.files.selection()
        if selected and selected[0] in self.file_items:
            self.set_details(self.plan.item_summary(self.file_items[selected[0]]))

    def choose_group(self):
        selected = self.group_tree.selection()
        if not self.busy and selected:
            self.plan.select_group(self.group_keys[selected[0]])
            self.show_result()

    def copy_summary(self):
        if self.plan is not None:
            self.dialog.clipboard_clear()
            self.dialog.clipboard_append(self.plan.summary(completed=self.plan.finished))
            self.note.set("Skopiowano pełną diagnostykę: wszystkie grupy, zakresy MT i szczegóły plików.")

    def back(self):
        if not self.busy:
            self.phase = "selection"
            self.page.pack_forget()
            self.selection.pack(fill=tk.BOTH, expand=True)


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
        self._source_load_tokens: dict[str, int] = {}
        self._source_attempts: set[tuple[str, str, str]] = set()
        self._source_sequence = 0
        self._source_failures: dict[str, str] = {}
        self._crop_entry_rows: list[str] = []
        self._crop_photo = None
        self._science_sequence = 0
        self._science_cache: dict[tuple, dict] = {}
        self._science_error_by_key: dict[tuple, str] = {}
        self._science_requested_key = None
        self._latency_report_identity = None
        self._latency_full_x = None
        self._latency_x_view = None
        self._latency_pan_origin = None
        self._latency_effective_mode = None
        self._latency_hit_points = []
        self._latency_hover_baseline = ''
        self._latency_time_seconds = None
        self._latency_time_context = {}
        self._closing = False
        self._import_in_progress = False
        self._import_cancel_requested = False
        self._suppress_report_select = False
        # AKT87: persistent checkbox selection keyed by verified report identity.
        self._comparison_checked = set()
        self._comparison_verification = {}
        self._comparison_busy = False
        self._comparison_dialogs = []
        self._report_sort_column,self._report_sort_descending=report_sort_preference(SESSION)
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
        # Z4MOBREP029: on Windows the left Treeview can own keyboard focus;
        # route wheel events by pointer location, before Treeview class bindings.
        self._install_report_pointer_wheel()
        self._show_empty()
        self.window.bind("<FocusIn>", self._refresh_review_on_focus, add="+")
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
        width = min(1480, max(950, screen_w - 72))
        height = min(860, max(630, screen_h - 90))
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
        # Remove the per-window wheel tag; do not leave a global Tk binding.
        tag = getattr(self, "_report_wheel_tag", None)
        if tag:
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                try:
                    self.window.unbind_class(tag, sequence)
                except tk.TclError:
                    pass
            self._report_wheel_tag = None
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
        for panel in getattr(self, '_photo_reference_windows', ()):
            if not panel._closed:
                panel.close()
                if not panel._closed:
                    return
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
        button = getattr(self, "btn_cancel_import", None)
        if button is not None:
            button.grid() if running else button.grid_remove()
            button.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _cancel_mobile_import(self) -> None:
        self._import_cancel_requested = True
        self._set_status("Anulowanie — kończę bieżące archiwum. Zapisane wcześniej sesje pozostaną w magazynie.", "warning")
        button = getattr(self, "btn_cancel_import", None)
        if button is not None:
            button.configure(state=tk.DISABLED)

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

        default_dir = mobile_import_preferences.readable_directory(Path(initial_dir or Path.home())) or Path.home()
        start_dir = mobile_import_preferences.initial_directory(default_dir)

        selected_paths: list[Path] = []
        self._folder_import_handled = False
        self._folder_import_saved_id = ""
        state: dict[str, Any] = {"dir": start_dir}
        entries: list[dict[str, Any]] = []

        dialog = tk.Toplevel(self.window)
        dialog.withdraw()
        dialog.title("Wybierz raporty z telefonu")
        dialog.configure(bg=self.bg)
        try:
            dialog.geometry("1040x740")
            dialog.minsize(760, 600)
            dialog.resizable(True, True)
        except Exception:
            pass

        shell = tk.Frame(dialog, bg=self.bg, padx=14, pady=12)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_rowconfigure(3, weight=1)

        title = tk.Label(
            shell,
            text="Wybierz pliki raportów lub folder do skanowania",
            bg=self.bg,
            fg=self.fg,
            font=("Segoe UI", 13, "bold"),
            anchor=tk.W,
        )
        title.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        subtitle = tk.Label(
            shell,
            text="Pliki: .alprsession, .zip, .json (Ctrl lub Shift). Import folderu skanuje ścieżkę z pola poniżej i jej podfoldery.",
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
            text="Folder do skanowania:",
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
        remember_var = tk.BooleanVar(value=mobile_import_preferences.remember_enabled())
        tk.Checkbutton(
            path_row, text="Zapamiętaj folder ostatniego importu", variable=remember_var,
            command=lambda: mobile_import_preferences.set_remember_enabled(remember_var.get()),
            bg=self.bg, fg=self.fg, activebackground=self.bg, activeforeground=self.fg,
            selectcolor=self.panel, font=("Segoe UI", 9),
        ).grid(row=1, column=1, columnspan=2, sticky="w", pady=(5, 0))

        def _choose_folder() -> None:
            try:
                dialog.grab_release()
                selected = filedialog.askdirectory(
                    parent=dialog, title="Wybierz folder do importu", mustexist=True,
                    initialdir=str(mobile_import_preferences.readable_directory(path_var.get()) or state["dir"]))
                if selected:
                    path_var.set(selected)
                    _load_dir(selected)
            except Exception as exc:
                _set_status("Nie udało się otworzyć selektora folderów. Wklej pełną ścieżkę w polu powyżej.", "error")
                logger.warning("Selektor folderu importu: %s", exc)
            finally:
                if dialog.winfo_exists():
                    dialog.grab_set()
                    dialog.focus_force()

        ttk.Button(path_row, text="Wybierz folder…", command=_choose_folder).grid(
            row=0, column=2, sticky="e", padx=(8, 0))

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

        def _load_dir(path: Path | str) -> bool:
            try:
                target = Path(str(path or "")).expanduser()
                if not target.is_dir():
                    raise FileNotFoundError(str(target))
                target = target.resolve()
                children = list(target.iterdir())
            except Exception as exc:
                _set_status(mobile_import_path_error(exc, path), "error")
                return False

            entries.clear()
            listbox.delete(0, tk.END)
            state["dir"] = target
            path_var.set(str(target))

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
            return True

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

        def _accept_folder() -> None:
            # Only the visible editable path chooses the root. Navigating into
            # a subdirectory updates that path; list selection cannot override it.
            folder = path_var.get()
            if not _load_dir(folder):
                return
            flow.scan(Path(state["dir"]))

        def _close() -> None:
            if not flow.request_close():
                return
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
            ("Raporty", default_dir),
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
        ttk.Button(actions, text="Sprawdź folder z podfolderami", command=_accept_folder).grid(
            row=1, column=7, columnspan=2, sticky="e", pady=(10, 0), ipady=2)
        flow = _MobileFolderImportFlow(self, dialog, shell, _close)

        path_entry.bind("<Return>", lambda _event: (_load_dir(path_var.get()), "break")[-1], add="+")
        listbox.bind("<Double-Button-1>", _open_or_accept, add="+")
        listbox.bind("<Return>", _open_or_accept, add="+")
        dialog.bind("<Escape>", lambda _event: (_close(), "break")[-1], add="+")
        dialog.protocol("WM_DELETE_WINDOW", _close)

        if not _load_dir(start_dir) and start_dir != default_dir:
            _load_dir(default_dir)
        try:
            dialog.update_idletasks()
            minimum_width = max(760, int(actions.winfo_reqwidth()) + 28)
            dialog.minsize(minimum_width, 600)
            parent_x = int(self.window.winfo_rootx())
            parent_y = int(self.window.winfo_rooty())
            parent_w = int(self.window.winfo_width() or 920)
            parent_h = int(self.window.winfo_height() or 620)
            width = max(minimum_width, min(940, max(760, int(parent_w * 0.86))))
            height = min(780, max(650, int(parent_h * 0.82)))
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

    def open_final_presentation(self):
        from .final180_tv import open_from_desktop
        try:
            app=getattr(self, '_final_presentation', None)
            if app is not None and app.window.winfo_exists():app.window.lift();return
            self._final_presentation=open_from_desktop(self.window)
        except Exception as exc:
            messagebox.showerror('Pokaz fotografii', str(exc), parent=self.window)

    def open_photo_reference(self):
        from .photo_material_ui import open_material_editor
        app=getattr(self,'_final_presentation',None)
        material=getattr(app,'source_material',None) if app is not None and not getattr(app,'closed',True) else None
        panel = open_material_editor(self.window,material,on_close=app.refresh_material_reference if material is not None else None)
        if panel is not None:
            self._photo_reference_windows = [p for p in getattr(self, '_photo_reference_windows', ()) if not p._closed] + [panel]
        return panel

    def open_final_analysis(self):
        from .final180_tv import authentic_final_bundle
        bundle=getattr(self,'current_bundle',None);report=getattr(self,'current_report',None)
        if not authentic_final_bundle(bundle):return
        self.open_final_presentation()
        app=getattr(self,'_final_presentation',None)
        if app is None:return
        from .z4_mobile_sample_review import open_mobile_sample_review
        app.manual_review_callback=lambda:open_mobile_sample_review(self,report,bundle)
        app.import_android(archive=bundle.path)

    def _sync_final_action(self):
        from .final180_tv import authentic_final_bundle
        final=authentic_final_bundle(getattr(self,'current_bundle',None))
        panel=getattr(self,'science_panel',None)
        if panel is not None and hasattr(panel,'review_action_button'):
            panel.review_action_button.configure(text='Weryfikuj wyniki →')
            if final:
                panel.progress_var.set('FINAL: automatyczne powiązanie i ocena, bez ręcznych zatwierdzeń fotografii.')
                panel.progress_info_var.set('Wskaż run PC. Brak GT lub brak dowodów pozostanie jawnym N/A.')
        if hasattr(self,'sample_review_link'):
            self.sample_review_link.configure(text='Weryfikuj wyniki →')

    def _build(self) -> None:
        root = tk.Frame(self.window, bg=self.bg, padx=12, pady=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.grid_columnconfigure(0, weight=1)
        # Z4MOBREP017: one quiet page heading, workspace and a compact status line.
        root.grid_rowconfigure(1, weight=1)
        header = tk.Frame(root, bg=self.bg)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 9))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(header, text="Raporty ALPR", bg=self.bg, fg=self.fg,
                 font=("Segoe UI", 16, "bold"), anchor=tk.W).grid(row=0, column=0, sticky="w")
        tk.Label(header, text="Wybierz sesję i sprawdź wyniki eksperymentu.",
                 bg=self.bg, fg=self.muted, font=("Segoe UI", 9),
                 anchor=tk.W).grid(row=1, column=0, sticky="w", pady=(2, 0))
        # Instead of a row of competing buttons, show source evidence as quiet
        # contextual information on the right. Recovery is the only conditional action.
        self.source_status_var = tk.StringVar(value="")
        self.source_status_label = tk.Label(header, textvariable=self.source_status_var,
                 bg=self.bg, fg=self.muted, font=("Segoe UI", 9),
                 justify=tk.RIGHT, anchor=tk.E, wraplength=460)
        self.source_status_label.grid(row=0, column=1, rowspan=2, sticky="e", padx=(18, 0))
        self.btn_link_archive = ttk.Button(header, text="Podłącz archiwum",
                  command=self.link_source_archive, state="disabled")
        self.btn_link_archive.grid(row=0, column=2, rowspan=2, sticky="e", padx=(8, 0))
        materials = ttk.LabelFrame(header, text='Materiały i pokazy', padding=(6, 3))
        materials.grid(row=0,column=3,rowspan=2,sticky='e',padx=(8,0))
        self.material_gt_button=ttk.Menubutton(materials,text='Fotografie i oznaczenia ▾')
        self.material_gt_button.pack(side='left',padx=(0,6))
        material_menu=tk.Menu(self.material_gt_button,tearoff=False);self.material_gt_button.configure(menu=material_menu)
        material_menu.add_command(label='Wybierz materiał / otwórz edytor',command=self.open_photo_reference)
        from .photo_material_ui import open_recent_editor,open_material_editor
        material_menu.add_command(label='Wybierz inny katalog fotografii…',command=lambda:open_material_editor(self.window))
        material_menu.add_command(label='Wznów ostatnią sesję GT',command=lambda:open_recent_editor(self.window))
        ttk.Button(materials, text='Pokaz fotografii', command=self.open_final_presentation).pack(side='left')
        self.btn_link_archive.grid_remove()

        workspace = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg=self.bg, bd=0, relief=tk.FLAT,
                                   sashwidth=8, sashrelief=tk.FLAT, opaqueresize=True)
        workspace.grid(row=1, column=0, sticky="nsew")
        self.report_split = workspace

        left = tk.Frame(
            workspace,
            bg=self.card_bg,
            padx=9,
            pady=8,
            highlightthickness=1,
            highlightbackground=self.border,
        )
        workspace.add(left, width=650, minsize=610, padx=0, pady=0, stretch="never")
        self._report_list_pane = left
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(3, weight=1)
        left_header = tk.Frame(left, bg=self.card_bg)
        left_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        left_header.grid_columnconfigure(0, weight=1)
        tk.Label(left_header, text="Sesje", bg=self.card_bg, fg=self.fg,
                 font=("Segoe UI", 11, "bold"), anchor=tk.W).grid(row=0, column=0, sticky="w")
        self.btn_refresh_reports = tk.Button(left_header, text="Odśwież", command=self._refresh_report_list,
                relief=tk.FLAT, bd=0, highlightthickness=0, bg=self.card_bg, fg=self.accent,
                activebackground=self.card_bg, activeforeground=self.fg, cursor="hand2",
                font=("Segoe UI", 9, "underline"))
        self.btn_refresh_reports.grid(row=0, column=1, sticky="e")

        search_area = tk.Frame(left, bg=self.card_bg)
        search_area.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        search_area.grid_columnconfigure(0, weight=1)
        tk.Label(search_area, text="Szukaj sesji", bg=self.card_bg, fg=self.muted,
                 font=("Segoe UI", 9), anchor=tk.W).grid(row=0, column=0, sticky="ew")
        self.search_var = tk.StringVar(value="")
        self.search_entry = tk.Entry(search_area, textvariable=self.search_var, bg=self.panel,
                                     fg=self.fg, insertbackground=self.fg, relief=tk.FLAT,
                                     highlightthickness=1, highlightbackground=self.border,
                                     highlightcolor=self.accent, font=("Segoe UI", 10))
        self.search_entry.grid(row=1, column=0, sticky="ew", ipady=5, pady=(3, 0))
        self.search_entry.bind("<Escape>", lambda _event: self.search_var.set(""), add="+")
        self.search_var.trace_add("write", lambda *_args: self._apply_report_filter(auto_select=True))
        self.report_count_var = tk.StringVar(value="Brak zapisanych raportów")
        tk.Label(left, textvariable=self.report_count_var, bg=self.card_bg, fg=self.muted,
                 font=("Segoe UI", 9), anchor=tk.W).grid(row=2, column=0, sticky="ew", pady=(0, 5))
        self.report_table = _TreeTable(
            left, ("Wybor", "Nazwa", "Data", "Wariant", "Scena", "Powt.", "Weryfikacja", "p95"),
            ("✓", "Nazwa / notatka z telefonu", "Data", "Wariant", "Materiał", "Powt.", "Weryfikacja", "P95"),
            (28, 242, 94, 60, 96, 44, 104, 65), palette={**self.palette, "panel": self.card_bg}, height=18,
            selectmode="extended")
        self.report_table.shell.grid(row=3, column=0, sticky="nsew")
        self.report_table.tree.column('Wybor',width=28,minwidth=28)
        self.report_table.tree.configure(displaycolumns=('Wybor','Nazwa','Wariant','Powt.','Weryfikacja','Data'))
        self.report_table.tree.column('Nazwa', minwidth=210, stretch=True)
        tree=self.report_table.tree
        style=ttk.Style(tree)
        row_font=tkfont.Font(root=tree,font=style.lookup(tree.cget('style'),'font'))
        # Date/material can occupy two lines. Leave a visible gap at the
        # actual DPI rather than squeezing 40 px of text into a 42 px row.
        row_gap=max(8,round(tree.winfo_fpixels('6p')))
        style.configure(tree.cget('style'),rowheight=max(46,2*row_font.metrics('linespace')+row_gap))
        tree.tag_configure('session_even',background=self.card_bg)
        tree.tag_configure('session_odd',background=blend_hex_colors(self.card_bg,self.fg,0.07))
        style.map(tree.cget('style'),
            background=[('selected',self.palette.get('selection_bg',self.accent))],
            foreground=[('selected',self.palette.get('selection_fg',self.fg))])
        self._update_sort_headings()
        self._size_report_list_columns()
        list_xscroll = ttk.Scrollbar(self.report_table.shell, orient=tk.HORIZONTAL, command=tree.xview)
        list_xscroll.grid(row=1, column=0, sticky="ew")
        def update_list_xscroll(first, last):
            list_xscroll.set(first, last)
            if float(first) <= 0 and float(last) >= 1:
                list_xscroll.grid_remove()
            else:
                list_xscroll.grid()
        tree.configure(xscrollcommand=update_list_xscroll)
        self.report_table.tree.bind("<<TreeviewSelect>>", self._on_report_selected, add="+")
        from .mobile_report_selection import ReportListSelection
        self._report_list_selection = ReportListSelection(self)
        self.btn_import_reports = ttk.Button(left, text="Importuj raport z telefonu", command=self.import_report)
        self.btn_import_reports.grid(row=4, column=0, sticky="ew", pady=(9, 0), ipady=3)
        # AKT87: one click per reviewed session. No duplicate selection basket.
        self.compare_hint = tk.Label(left, text="Ctrl/Shift + klik: wybór wierszy · Spacja: zaznacz/odznacz grupę",
            bg=self.card_bg, fg=self.muted, anchor=tk.W, font=("Segoe UI", 8))
        self.compare_hint.grid(row=5,column=0,sticky="ew",pady=(8,3))
        self.btn_compare_open = ttk.Button(left,text="Porównaj wybrane (0)",
            command=self._open_mobile_comparison,state="disabled")
        self.btn_compare_open.grid(row=6,column=0,sticky="ew")
        self.btn_compare_clear = ttk.Button(left,text="Odznacz wszystkie",
            command=self._clear_comparison_checks)
        self.btn_compare_clear.grid(row=7,column=0,sticky="ew",pady=(5,0))
        self.btn_export_mobile_figures = ttk.Button(left,text="Wykresy tej sesji · PNG/SVG",
            command=self._export_mobile_report_figures)
        self.btn_export_mobile_figures.grid(row=8,column=0,sticky="ew",pady=(6,0))
        # Full model paths and identifiers are in 'Szczegóły', not in the session list.
        self.selected_report_info_var = tk.StringVar(value="")


        right = tk.Frame(workspace, bg=self.bg)
        workspace.add(right, minsize=510, padx=8, pady=0, stretch="always")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)
        identity = tk.Frame(right, bg=self.bg)
        identity.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        identity.grid_columnconfigure(0, weight=1)
        self.hero_title_var = tk.StringVar(value="Wybierz raport z listy")
        self.hero_meta_var = tk.StringVar(value="Najważniejsze informacje o sesji pojawią się tutaj.")
        self.hero_title_label = tk.Label(identity, textvariable=self.hero_title_var, bg=self.bg, fg=self.fg,
                         font=("Segoe UI", 13, "bold"), anchor=tk.W, justify=tk.LEFT, wraplength=650)
        self.hero_title_label.grid(row=0, column=0, sticky="ew")
        self.hero_meta_label = tk.Label(identity, textvariable=self.hero_meta_var, bg=self.bg, fg=self.muted,
                        font=("Segoe UI", 9), anchor=tk.W, justify=tk.LEFT, wraplength=650)
        self.hero_meta_label.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        phone_note = tk.Frame(identity, bg=self.bg)
        phone_note.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        phone_note.grid_columnconfigure(0, weight=1)
        tk.Label(phone_note, text="Notatka z telefonu", bg=self.bg, fg=self.fg,
                 font=("Segoe UI", 9, "bold"), anchor=tk.W).grid(row=0, column=0, sticky="w", pady=(0, 3))
        self.phone_note_text = tk.Text(phone_note, height=3, width=1, wrap="word", state="disabled",
            bg=self.panel, fg=self.fg, relief=tk.FLAT, font=("Segoe UI", 10), padx=7, pady=5,
            selectbackground=self.palette.get('selection_bg', self.accent), takefocus=True)
        self.phone_note_text.grid(row=1, column=0, sticky="ew")
        phone_scroll = ttk.Scrollbar(phone_note, orient=tk.VERTICAL, command=self.phone_note_text.yview)
        phone_scroll.grid(row=1, column=1, sticky="ns")
        self.phone_note_text.configure(yscrollcommand=phone_scroll.set)
        # No navigation buttons next to the title: tabs directly below already
        # provide navigation, and warnings are contextual to the source details.
        right.bind("<Configure>", self._resize_report_details, add="+")

        self.card_shell = tk.Frame(right, bg=self.bg)
        self.card_shell.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for column in range(4):
            self.card_shell.grid_columnconfigure(column, weight=1, uniform="report_cards")
        self.card_vars = {
            "integrity": tk.StringVar(value="Integralność: -"),
            "frames": tk.StringVar(value="Klatki: -"),
            "quality": tk.StringVar(value="Jakość: -"),
            "latency": tk.StringVar(value="p95: -"),
        }
        self.card_labels: dict[str, tk.Label] = {}
        self.card_hint_vars: dict[str, tk.StringVar] = {}
        self.metric_card_frames: dict[str, tk.Frame] = {}
        self._metric_cards = {}
        self._metric_help = HoverCardHelp(self.window, lambda key: (
            self._metric_cards[key].explanation if key in self._metric_cards else ''))
        for index, (key, title, color) in enumerate(
            (
                ("integrity", "Źródło", self.accent),
                ("frames", "Klatki ALPR", self.accent),
                ("quality", "GT i weryfikacja", self.accent),
                ("latency", "E2E p95", self.accent),
            )
        ):
            card = tk.Frame(
                self.card_shell,
                bg=self.card_bg,
                padx=11,
                pady=10,
                highlightthickness=1,
                highlightbackground=self.border,
            )
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 6, 0))
            self.metric_card_frames[key] = card
            title_label = tk.Label(card, text=title, bg=card["bg"], fg=self.muted,
                                   font=("Segoe UI", 9, "bold"), anchor=tk.W, justify=tk.LEFT)
            title_label.pack(fill=tk.X)
            label = tk.Label(
                card,
                textvariable=self.card_vars[key],
                bg=card["bg"],
                fg=self.fg,
                font=("Segoe UI", 11, "bold"),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=180,
            )
            label.pack(fill=tk.X, pady=(5, 0))
            self.card_labels[key] = label
            hint_var = tk.StringVar(value="")
            self.card_hint_vars[key] = hint_var
            hint = tk.Label(card, textvariable=hint_var, bg=card["bg"], fg=self.muted,
                            font=("Segoe UI", 8), anchor=tk.W, justify=tk.LEFT,
                            wraplength=180)
            hint.pack(fill=tk.X, pady=(3, 0))
            for widget in (card, title_label, label, hint):
                self._metric_help.bind(widget, key)

        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=2, column=0, sticky="nsew")
        self._build_tabs()
        self.review_status_var = tk.StringVar(value="Weryfikacja Desktop: wybierz sesję")
        self.review_status_label = tk.Label(right, textvariable=self.review_status_var,
               bg=self.card_bg, fg=self.fg, font=("Segoe UI", 9), anchor=tk.W, justify=tk.LEFT,
               wraplength=700, padx=10, pady=7)
        self.review_status_label.grid(row=3, column=0, sticky="ew", pady=(7, 0))
        # Z4MOBREP016: technical KPI cards duplicated the scientific dashboard.
        self.card_shell.grid_remove()
        self.review_status_label.grid_remove()
        # Keep the single status bar at the bottom; a log/progress message no
        # longer consumes an extra toolbar row across the whole screen.
        status_bar = tk.Frame(root, bg=self.bg)
        status_bar.grid(row=2, column=0, sticky="ew", pady=(7, 0))
        status_bar.grid_columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Wybierz sesję z listy lub zaimportuj raport.")
        self.status_label = tk.Label(status_bar, textvariable=self.status_var,
                    bg=self.bg, fg=self.muted, font=("Segoe UI", 8), anchor=tk.W)
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.progress = ttk.Progressbar(status_bar, mode="indeterminate", length=110)
        self.btn_cancel_import = ttk.Button(status_bar, text="Anuluj import", command=self._cancel_mobile_import)
        self.btn_cancel_import.grid(row=0, column=2, sticky="e", padx=(8, 0))
        self.btn_cancel_import.grid_remove()

    @staticmethod
    def _pointer_inside_widget(widget, x: int, y: int) -> bool:
        """Check a visible target in desktop screen coordinates."""
        try:
            return (bool(widget.winfo_ismapped())
                    and widget.winfo_rootx() <= x < widget.winfo_rootx() + widget.winfo_width()
                    and widget.winfo_rooty() <= y < widget.winfo_rooty() + widget.winfo_height())
        except tk.TclError:
            return False

    def _route_report_pointer_wheel(self, event):
        """Scroll the pane UNDER THE POINTER, not the widget retaining focus.

        A private front-of-bindtags handler intercepts the left Treeview before
        its class binding, which otherwise consumes wheel events on Windows.
        The dashboard and all of its cards scroll as one canvas. Unrelated
        notebook tabs, the timeline zoom and the phone-review window remain
        untouched by this handler.
        """
        units = _report_wheel_scroll_units(event)
        if not units:
            return None
        try:
            x, y = self.window.winfo_pointerxy()
            # Do not scroll the browser behind a different top-level dialog.
            below = self.window.winfo_containing(x, y)
            if below is None or below.winfo_toplevel() is not self.window:
                return None
            dashboard = self.science_panel
            if self._pointer_inside_widget(dashboard, x, y):
                dashboard.canvas.yview_scroll(units, "units")
                return "break"  # never scroll the still-focused session list
            left = self.report_table
            if self._pointer_inside_widget(left.shell, x, y):
                left.tree.yview_scroll(units, "units")
                return "break"  # never scroll the dashboard when pointer is left
        except (tk.TclError, AttributeError, TypeError):
            return None
        return None

    def _install_report_pointer_wheel(self) -> None:
        """Register a window-local wheel tag before widget/class default tags.

        Using bind_all here would affect other application windows. The unique
        tag is limited to widgets in this report browser and removed on close.
        """
        tag = f"AATReportPointerWheel_{id(self)}"
        self._report_wheel_tag = tag
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.window.bind_class(tag, sequence, self._route_report_pointer_wheel)

        def attach(widget) -> None:
            tags = widget.bindtags()
            if tag not in tags:
                widget.bindtags((tag,) + tags)
            for child in widget.winfo_children():
                attach(child)

        attach(self.window)

    def _build_tabs(self) -> None:
        """Four operator-facing destinations; complete legacy data stay reachable."""
        def tab_frame(parent, *, row=0):
            frame = tk.Frame(parent, bg=self.panel, padx=8, pady=8)
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(row, weight=1)
            return frame

        # 1: the main report uses canonical scientific metrics, not raw JSON.
        dashboard_host = tab_frame(self.notebook)
        self.science_panel = ScientificDashboard(
            dashboard_host, palette=self.palette, on_review=self.open_sample_review,
            on_refresh=self.refresh_scientific_dashboard,
            on_details=lambda: self._open_technical_details(),
            show_identity=False,
        )
        self.science_panel.grid(row=0, column=0, sticky="nsew")
        self.notebook.add(dashboard_host, text="Wyniki sesji")

        # 2: maintain the existing comparison guard and its full metadata.
        comparison = tab_frame(self.notebook, row=1)
        tk.Label(comparison, text="Czy te sesje wolno porównać?", bg=self.panel, fg=self.fg,
                 font=("Segoe UI", 11, "bold"), anchor=tk.W).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.comparison_table = _WrappedLabelTable(
            comparison, ("Warunek", "Ta sesja", "Seria", "Weryfikacja", "Znaczenie"),
            ("Warunek", "Ta sesja", "Seria", "Weryfikacja", "Znaczenie"),
            (170, 220, 160, 120, 330), palette={**self.palette, "panel": self.panel}, height=17)
        self.comparison_table.shell.grid(row=1, column=0, sticky="nsew")
        self.notebook.add(comparison, text="Porównywalność")

        # 3: all original crop preview functionality, plus direct review action.
        samples = tab_frame(self.notebook, row=1)
        samples.grid_rowconfigure(2, weight=0)
        head = tk.Frame(samples, bg=self.panel)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        head.grid_columnconfigure(0, weight=1)
        tk.Label(head, text="Zapisane obrazy tablic", fg=self.fg, bg=self.panel,
                 font=("Segoe UI", 11, "bold"), anchor=tk.W).grid(row=0, column=0, sticky="ew")
        # Context link, not a second prominent CTA competing with the main review.
        sample_review_link = tk.Button(head, text="Przejdź do weryfikacji zdjęć →", command=self.open_sample_review,
                        bg=self.panel, fg=self.accent, activebackground=self.panel,
                        activeforeground=self.fg, relief=tk.FLAT, bd=0,
                        highlightthickness=0, cursor="hand2", font=("Segoe UI", 9, "underline"))
        sample_review_link.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.sample_review_link=sample_review_link
        tk.Label(head, text="Brak wyciętego obrazu nie oznacza braku tablicy — sprawdź pełne zdjęcie w weryfikatorze.",
                 fg=self.muted, bg=self.panel, font=("Segoe UI", 9), anchor=tk.W).grid(
                 row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        self.crops_table = _TreeTable(
            samples, ("Próbka", "Seria", "Weryfikacja MT/MZ", "Odczyt", "GT Desktop", "Czas", "Obraz"),
            ("Próbka", "Seria", "Weryfikacja MT/MZ", "Odczyt", "GT Desktop", "Czas", "Obraz"),
            (120, 70, 185, 120, 130, 100, 180),
            palette={**self.palette, "panel": self.panel}, height=12)
        self.crops_table.shell.grid(row=1, column=0, sticky="nsew")
        self.crops_table.tree.bind("<<TreeviewSelect>>", self._on_crop_selected, add="+")
        crop_preview_frame = tk.Frame(samples, bg=self.panel)
        crop_preview_frame.grid(row=2, column=0, sticky="ew", pady=(7, 0))
        self.crop_preview_var = tk.StringVar(value="Wybierz tablicę z listy, aby zobaczyć jej obraz.")
        self.crop_preview_info = tk.Label(crop_preview_frame, textvariable=self.crop_preview_var,
                                          fg=self.muted, bg=self.panel, anchor=tk.W, justify=tk.LEFT,
                                          font=("Segoe UI", 9))
        self.crop_preview_info.pack(fill=tk.X)
        self.crop_image_label = tk.Label(crop_preview_frame, bg=self.panel)
        self.crop_image_label.pack(anchor=tk.W)
        self.notebook.add(samples, text="Próbki i weryfikacja")

        # 4: no scientific or provenance data are deleted; only regrouped.
        details = tab_frame(self.notebook)
        self.technical_notebook = ttk.Notebook(details)
        self.technical_notebook.grid(row=0, column=0, sticky="nsew")
        detail_tools = tk.Frame(details, bg=self.panel)
        detail_tools.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        detail_tools.grid_columnconfigure(0, weight=1)
        self.btn_report_warnings = ttk.Button(detail_tools, text="Informacje o integralności",
                        command=self._show_validation_details, state="disabled")
        self.btn_report_warnings.grid(row=0, column=1, sticky="e")
        self.notebook.add(details, text="Dane źródłowe")

        summary = tab_frame(self.technical_notebook)
        self.summary_table = _WrappedLabelTable(
            summary, ("Pole", "Wartość", "Opis"), ("Pole", "Wartość", "Opis"),
            (190, 250, 330), palette={**self.palette, "panel": self.panel}, height=17)
        self.summary_table.shell.grid(row=0, column=0, sticky="nsew")
        self.technical_notebook.add(summary, text="Raport źródłowy")

        config = tab_frame(self.technical_notebook)
        self.config_table = _WrappedLabelTable(
            config, ("Sekcja", "Wartość", "Opis"), ("Sekcja", "Wartość", "Opis"),
            (260, 360, 240), palette={**self.palette, "panel": self.panel}, height=17)
        self.config_table.shell.grid(row=0, column=0, sticky="nsew")
        self.technical_notebook.add(config, text="Warunki i modele")

        # Z4MOBREP019_FIX04: the scientific plot is the primary content; the
        # scrollable raw-metrics table lives in a smaller, independently
        # adjustable pane. Neither plot data nor measurement semantics change.
        latency = tab_frame(self.technical_notebook)
        self.latency_split = ttk.PanedWindow(latency, orient=tk.VERTICAL)
        self.latency_split.grid(row=0, column=0, sticky="nsew")
        latency_plot = tk.Frame(self.latency_split, bg=self.panel)
        # Display spacing is determined solely by source frame IDs; this
        # checkbox controls ONLY whether already positioned points are linked.
        self._latency_connect_neighbors_var = tk.BooleanVar(master=self.window, value=True)
        self._latency_axis_bar = tk.Frame(latency_plot, bg=self.panel)
        self._latency_axis_bar.pack(fill=tk.X, padx=6, pady=(1, 2))
        self._latency_connections_toggle = tk.Checkbutton(
            self._latency_axis_bar,
            text="Łącz punkty sąsiednich klatek",
            variable=self._latency_connect_neighbors_var,
            command=self._on_latency_connections_change,
            bg=self.panel, fg=self.fg, selectcolor=self.panel,
            activebackground=self.panel, activeforeground=self.fg,
            font=("Segoe UI", 9), borderwidth=0, highlightthickness=0,
        )
        self._latency_connections_toggle.pack(side=tk.LEFT, anchor=tk.W)
        self._latency_axis_mode_var = tk.StringVar(master=self.window, value='time')
        # Explicit axis selection is different from the line-only checkbox.
        # The checkbox must never change x/y coordinates or the current zoom.
        self._latency_frame_axis_radio = tk.Radiobutton(
            self._latency_axis_bar, text='ID klatki', value='frame',
            variable=self._latency_axis_mode_var,
            command=self._on_latency_axis_mode_change,
            bg=self.panel, fg=self.fg, selectcolor=self.panel,
            activebackground=self.panel, activeforeground=self.fg,
            font=('Segoe UI', 9), highlightthickness=0, borderwidth=0,
        )
        self._latency_frame_axis_radio.pack(side=tk.RIGHT, padx=(8, 0))
        self._latency_time_axis_radio = tk.Radiobutton(
            self._latency_axis_bar, text='Czas sesji', value='time',
            variable=self._latency_axis_mode_var,
            command=self._on_latency_axis_mode_change,
            bg=self.panel, fg=self.fg, selectcolor=self.panel,
            activebackground=self.panel, activeforeground=self.fg,
            font=('Segoe UI', 9), highlightthickness=0, borderwidth=0,
        )
        self._latency_time_axis_radio.pack(side=tk.RIGHT)
        # One fixed-height line gives session duration and hover details,
        # without changing geometry whenever the operator moves the mouse.
        self._latency_context_var = tk.StringVar(master=self.window, value='')
        self._latency_context_label = tk.Label(
            latency_plot, textvariable=self._latency_context_var,
            fg=self.muted, bg=self.panel, anchor=tk.W, justify=tk.LEFT,
            font=('Segoe UI', 8), height=1, width=1,
        )
        self._latency_context_label.pack(fill=tk.X, padx=(50, 18), pady=(0, 1))
        self.latency_canvas = tk.Canvas(latency_plot, bg=self.panel, highlightthickness=1,
                                        highlightbackground=self.border, height=320)
        self.latency_canvas.pack(fill=tk.BOTH, expand=True)
        # Keep the X-axis explanation in a separate footer. Text painted only
        # five pixels above the canvas bottom was clipped by the PanedWindow
        # sash, especially after resizing the plot/table splitter.
        self._latency_axis_footer_var = tk.StringVar(master=self.window, value="")
        self._latency_axis_footer_label = tk.Label(
            latency_plot, textvariable=self._latency_axis_footer_var,
            fg=self.muted, bg=self.panel, anchor=tk.W, justify=tk.LEFT,
            font=("Segoe UI", 8), wraplength=900,
        )
        # A PanedWindow lets the operator give more room to the table. Tk
        # allocates space to packed children in packing order: when the 320px
        # canvas was packed first, a shorter plot pane could starve this label
        # and make the entire scientific X-axis explanation disappear.
        # Reserve the bottom caption BEFORE allocating the stretchable canvas.
        # The canvas alone shrinks as the splitter moves; the caption does not.
        self._latency_axis_footer_label.pack(
            side=tk.BOTTOM, fill=tk.X, before=self.latency_canvas,
            padx=(50, 18), pady=(3, 6),
        )
        self.latency_canvas.bind("<Configure>", lambda _event: self._draw_latency_chart(), add="+")
        self.latency_canvas.bind(
            "<Configure>",
            lambda event: self._latency_axis_footer_label.configure(
                wraplength=max(220, int(event.width) - 72)),
            add="+",
        )
        self.latency_canvas.bind("<MouseWheel>", self._zoom_latency_wheel, add="+")
        self.latency_canvas.bind("<Button-4>", self._zoom_latency_wheel, add="+")
        self.latency_canvas.bind("<Button-5>", self._zoom_latency_wheel, add="+")
        self.latency_canvas.bind("<ButtonPress-1>", self._begin_latency_pan, add="+")
        self.latency_canvas.bind("<B1-Motion>", self._move_latency_pan, add="+")
        self.latency_canvas.bind("<ButtonRelease-1>", self._end_latency_pan, add="+")
        self.latency_canvas.bind("<Double-Button-1>", self._reset_latency_zoom, add="+")
        self.latency_canvas.bind('<Motion>', self._on_latency_point_hover, add='+')
        self.latency_canvas.bind('<Leave>', self._clear_latency_point_hover, add='+')
        latency_metrics = tk.Frame(self.latency_split, bg=self.panel)
        self.latency_table = _TreeTable(
            latency_metrics, ("Etap", "count", "mean", "p50", "p90", "p95", "p99", "min", "max", "std"),
            ("Etap", "n", "mean", "p50", "p90", "p95", "p99", "min", "max", "std"),
            (160, 48, 72, 72, 72, 72, 72, 72, 72, 72),
            palette={**self.palette, "panel": self.panel}, height=5)
        self.latency_table.shell.pack(fill=tk.BOTH, expand=True)
        self.latency_split.add(latency_plot, weight=3)
        self.latency_split.add(latency_metrics, weight=1)
        self.technical_notebook.add(latency, text="Czasy szczegółowe")

        artifacts = tab_frame(self.technical_notebook)
        self.artifacts_table = _WrappedLabelTable(
            artifacts, ("Artefakt", "Źródło", "Podgląd", "Status", "Opis"),
            ("Artefakt", "Źródło", "Podgląd", "Status", "Opis"),
            (170, 90, 90, 110, 430), palette={**self.palette, "panel": self.panel}, height=17)
        self.artifacts_table.shell.grid(row=0, column=0, sticky="nsew")
        self.technical_notebook.add(artifacts, text="Pliki i dowody")

        quality = tab_frame(self.technical_notebook)
        self.quality_table = _WrappedLabelTable(
            quality, ("Metryka", "Wartość", "Komentarz"), ("Metryka", "Wartość", "Komentarz"),
            (230, 160, 420), palette={**self.palette, "panel": self.panel}, height=17)
        self.quality_table.shell.grid(row=0, column=0, sticky="nsew")
        self.technical_notebook.add(quality, text="Pełne miary")

        diagnostics = tab_frame(self.technical_notebook, row=1)
        self.diagnostics_table = _WrappedLabelTable(
            diagnostics, ("Obszar", "Wartość", "Opis"), ("Obszar", "Wartość", "Opis"),
            (260, 240, 360), palette={**self.palette, "panel": self.panel}, height=8)
        self.diagnostics_table.shell.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.log_text = tk.Text(
            diagnostics, bg=blend_hex_colors(self.panel, self.bg, 0.35), fg=self.fg,
            insertbackground=self.fg, height=8, wrap=tk.WORD,
            font=("Consolas", 9), relief=tk.FLAT)
        log_scroll = WebSlimScrollbar(diagnostics, orient=tk.VERTICAL,
                                     command=self.log_text.yview, track_color=self.panel)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=1, column=0, sticky="nsew")
        log_scroll.grid(row=1, column=1, sticky="ns")
        self.technical_notebook.add(diagnostics, text="Diagnostyka")

        raw = tab_frame(self.technical_notebook)
        self.raw_text = tk.Text(raw, bg=blend_hex_colors(self.panel, self.bg, 0.35),
                                fg=self.fg, insertbackground=self.fg,
                                wrap=tk.NONE, font=("Consolas", 9), relief=tk.FLAT)
        raw_scroll_y = WebSlimScrollbar(raw, orient=tk.VERTICAL, command=self.raw_text.yview, track_color=self.panel)
        raw_scroll_x = WebSlimScrollbar(raw, orient=tk.HORIZONTAL, command=self.raw_text.xview, track_color=self.panel)
        self.raw_text.configure(yscrollcommand=raw_scroll_y.set, xscrollcommand=raw_scroll_x.set)
        self.raw_text.grid(row=0, column=0, sticky="nsew")
        raw_scroll_y.grid(row=0, column=1, sticky="ns")
        raw_scroll_x.grid(row=1, column=0, sticky="ew")
        self.technical_notebook.add(raw, text="Dane JSON")
        self.notebook.select(0)

    def _open_technical_details(self) -> None:
        self.notebook.select(3)
        self.technical_notebook.select(0)

    def _science_cache_key(self, report, bundle) -> tuple:
        sha = str(getattr(report, 'source_archive_sha256', '') or '').lower()
        review_root = Path(getattr(CONFIG, 'DIR_7_RANKINGS_MOBILE_PACKAGES', CONFIG.DIR_7_RANKINGS / 'mobile_packages'))
        sidecar = review_root / 'human_reviews' / f'{sha}.review.json'
        try:
            stat = sidecar.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = (None, None)
        from .mobile_report_canvas_view import export_stamp
        request = self._canvas_request(report, bundle)
        return (str(report.report_id), sha, str(getattr(bundle, 'path', '')), stamp,
                export_stamp(request["results_root"], request["session_id"]))

    def _canvas_request(self, report, bundle):
        from .mobile_report_canvas_view import results_root_for_source
        return dict(session_id=report_identity(report, bundle)["session_id"],
            source_sha=str(report.source_archive_sha256 or "").lower(),
            results_root=results_root_for_source(getattr(bundle, "path", "") or stored_source(report)),
            review_root=Path(getattr(CONFIG, "DIR_7_RANKINGS_MOBILE_PACKAGES", CONFIG.DIR_7_RANKINGS/"mobile_packages")))

    def refresh_scientific_dashboard(self) -> None:
        report = getattr(self, 'current_report', None)
        if report is not None:
            self._update_scientific_dashboard(report, getattr(self, 'current_bundle', None), force=True)

    def _update_scientific_dashboard(self, report, bundle, *, force: bool = False) -> None:
        """Load full canonical review outside Tk; stale selections cannot overwrite UI."""
        self._science_sequence += 1
        token = self._science_sequence
        if bundle is None or not getattr(getattr(bundle, 'validation', None), 'ok', False):
            self.science_panel.render(build_overview(report, None))
            return
        if (str(bundle.source_archive_sha256).lower() != str(report.source_archive_sha256).lower()
                or str(bundle.report.report_id) != str(report.report_id)):
            self.science_panel.render(build_overview(report, None))
            return
        cache_key = self._science_cache_key(report, bundle)
        self._science_requested_key = cache_key
        if not force and cache_key in self._science_cache:
            self.science_panel.render(build_overview(report, bundle, self._science_cache[cache_key]))
            self._populate_crops(report, bundle)
            return
        self.science_panel.render(build_overview(report, bundle, loading=True))

        def worker() -> None:
            try:
                science = science_from_verified_bundle(bundle)
                review_root = Path(getattr(CONFIG, "DIR_7_RANKINGS_MOBILE_PACKAGES", CONFIG.DIR_7_RANKINGS / "mobile_packages"))
                try:
                    # Canonical status is already computed from the same SHA-pinned
                    # review; this projection adds display-only text GT by subject_key.
                    science["_ui_desktop_samples"] = project_desktop_samples(bundle, science, review_root)
                except (OSError, ValueError, TypeError, KeyError) as overlay_error:
                    science["_ui_desktop_samples"] = {
                        "state": "error", "error": str(overlay_error)[:180], "samples": {}}
                error = ''
            except Exception as exc:
                science = None
                error = str(exc)[:220]
            # Keep protocols separate. Canvas comes only from the current
            # export selected by the same SHA/revision gate as the checkboxes.
            science = dict(science or {})
            from .mobile_report_canvas_view import load_canvas_snapshot
            request = self._canvas_request(report, bundle)
            try:
                science["_ui_canvas_snapshot"] = load_canvas_snapshot(**request)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                science["_ui_canvas_snapshot"] = dict(session_id=request["session_id"],
                    source_sha256=request["source_sha"], summary=None,
                    verification=dict(state="UNREADABLE" if isinstance(exc, OSError) else "INVALID",
                                      reason=str(exc)))
            self._run_on_ui(lambda: self._finish_scientific_dashboard(
                report, bundle, cache_key, token, science, error))

        threading.Thread(target=worker, daemon=True, name='alpr-scientific-overview').start()

    def _finish_scientific_dashboard(self, report, bundle, cache_key, token, science, error) -> None:
        if token != self._science_sequence or self._closing:
            return
        selected = getattr(self, 'current_report', None)
        current_bundle = getattr(self, 'current_bundle', None)
        if (selected is None or selected.report_id != report.report_id
                or selected.source_archive_sha256 != report.source_archive_sha256
                or current_bundle is None or current_bundle.source_archive_sha256 != bundle.source_archive_sha256):
            return
        if cache_key != self._science_cache_key(selected, current_bundle):
            # A save happened while a worker was reading the old review: never
            # mix an obsolete scientific status with a newer sidecar GT.
            self._update_scientific_dashboard(selected, current_bundle, force=True)
            return
        if science is not None:
            self._science_cache[cache_key] = science
            self._science_error_by_key.pop(cache_key, None)
        elif error:
            self._science_error_by_key[cache_key] = error
        self.science_panel.render(build_overview(selected, current_bundle, science, error=error))
        self._sync_final_action()
        self._populate_crops(selected, current_bundle)

    def _refresh_review_on_focus(self, event) -> None:
        if event.widget is not self.window or self._closing:
            return
        report, bundle = getattr(self, "current_report", None), getattr(self, "current_bundle", None)
        if report is None or bundle is None:
            return
        if self._science_cache_key(report, bundle) != self._science_requested_key:
            self._populate_crops(report, bundle)
            self._update_scientific_dashboard(report, bundle, force=True)

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
                self.progress.grid(row=0, column=1, sticky="e", padx=(8, 0))
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

    def _comparison_status_of(self, report):
        key=self._comparison_key(report)
        # Initial list is light-weight: use metadata + known original locator.
        # Full manifests are checked here; original .alprsession is not read.
        bundle=self._matching_cached_bundle(report)
        info=report_identity(report,bundle)
        sha=str(getattr(report,"source_archive_sha256","") or "").lower()
        from .mobile_report_source_recovery import stored_source
        source=str(getattr(bundle,"path","") or stored_source(report) or "")
        path=Path(source)
        results=None
        for ancestor in path.parents:
            if ancestor.name.lower()=="02_sesje":
                candidate=ancestor.parent/"03_wyniki"
                if candidate.is_dir():results=candidate
                break
        review_root=Path(getattr(CONFIG,"DIR_7_RANKINGS_MOBILE_PACKAGES",CONFIG.DIR_7_RANKINGS/"mobile_packages"))
        diagnostic = dict(browser_module=__name__, browser_module_path=__file__,
                          source_locator=source, report_id=str(report.report_id))
        if not hasattr(self, "_comparison_diagnostics"):
            self._comparison_diagnostics = {}
        self._comparison_diagnostics[key] = diagnostic
        return imported_verification(info["session_id"],sha,results,review_root,
                                     diagnostics=diagnostic)

    def _refresh_report_list(self, select_report_id: str = "", *, auto_select: bool = True) -> None:
        """Refresh operator verification and source-pinned export status."""
        try:
            self.store._load()
        except Exception:
            pass
        try:
            self.store.reports = [_compact_mobile_report_for_store(report) for report in self.store.reports]
        except Exception:
            pass
        self._ordered_reports = sorted(self.store.reports, key=lambda row: str(row.measured_at or ""), reverse=True)
        self._comparison_verification = {
            self._comparison_key(report):self._comparison_status_of(report)
            for report in self._ordered_reports
        }
        # Selection expresses intent. A GT change invalidates quality inside
        # the analyzer, not the operator's choice to inspect the source.
        self._comparison_checked = {
            key for key in self._comparison_checked
            if key in self._comparison_verification
        }
        self._update_comparison_basket_status(silent=True)
        self._apply_report_filter(select_report_id=select_report_id, auto_select=auto_select, refresh_preview=True)
        # Only legacy truncated notes need enrichment. Reuse SHA-pinned recovery
        # and its cache; filtering/refreshing must not scan every archive.
        for report in self._ordered_reports:
            if phone_session_metadata(report, self._matching_cached_bundle(report)).note.state == 'TRUNCATED':
                path = stored_source(report)
                if path:
                    self._begin_source_recovery(report, path)

    def _size_report_list_columns(self) -> None:
        """Fit this session table to its actual heading font and Windows DPI."""
        tree = self.report_table.tree
        style = ttk.Style(tree)
        style_name = tree.cget('style')
        heading_font = tkfont.Font(root=tree, font=(
            style.lookup(style_name + '.Heading', 'font') or 'TkHeadingFont'))
        row_font = tkfont.Font(root=tree, font=(style.lookup(style_name, 'font') or 'TkDefaultFont'))
        padding = max(16, round(tree.winfo_fpixels('14p')))
        samples = {
            'Wybor': ('☑',), 'Nazwa': (), 'Wariant': ('R0', 'R1', 'R2'),
            'Powt.': ('r03',), 'Data': ('09.11.2026', '23:59 +02:00'),
            'Weryfikacja': ('Nie rozpoczęto', 'Wymaga eksportu', 'Niepełna MT+MZ', 'eksport wymagany'),
        }
        total = 0
        for column in tree.cget('displaycolumns'):
            title = tree.heading(column, 'text').rstrip(' ↑↓')
            # Reserve both arrows once, so sorting does not move the separators.
            headings = (title,) if column == 'Wybor' else (title + ' ↑', title + ' ↓')
            width = max([heading_font.measure(text) for text in headings]
                        + [row_font.measure(text) for text in samples.get(column, ())]) + padding
            tree.column(column, width=width, minwidth=width, stretch=(column == 'Nazwa'))
            total += width
        # Prefer enough room for every visible column, while preserving the
        # ability to narrow the pane and scroll on a smaller display.
        self.report_split.paneconfigure(self._report_list_pane, width=total + 42)

    def _update_sort_headings(self):
        titles=dict(Nazwa='Nazwa / notatka z telefonu',Data='Data',Wariant='Wariant',Scena='Materiał',Weryfikacja='Weryfikacja',p95='P95',**{'Powt.':'Powt.'})
        for column in SORT_COLUMNS:
            arrow=(' ↓' if self._report_sort_descending else ' ↑') if column==self._report_sort_column else ''
            self.report_table.tree.heading(column,text=titles[column]+arrow,command=lambda c=column:self._sort_reports(c))

    def _sort_reports(self,column):
        if column not in SORT_COLUMNS:return
        if column==self._report_sort_column:self._report_sort_descending=not self._report_sort_descending
        else:self._report_sort_column,self._report_sort_descending=column,column=='Data'
        try:save_report_sort_preference(SESSION,self._report_sort_column,self._report_sort_descending)
        except Exception:logger.exception('Nie udało się zapisać kolejności widoku importów')
        self._update_sort_headings()
        self._apply_report_filter(auto_select=False,preserve_preview=True)

    def _apply_report_filter(self, select_report_id: str = "", *, auto_select: bool = True,
                             refresh_preview: bool = False, preserve_preview: bool = False) -> None:
        """Filter without rescanning archives; selection survives filtering."""
        all_reports = getattr(self, "_ordered_reports", [])
        query = self.search_var.get().strip() if hasattr(self, "search_var") else ""
        if select_report_id and query and not any(
            r.report_id == select_report_id and matches_report_filter(r, query, self._matching_cached_bundle(r))
            for r in all_reports
        ):
            self.search_var.set("")
            query = ""
        filtered = [r for r in all_reports if matches_report_filter(r, query, self._matching_cached_bundle(r))]
        column=getattr(self,'_report_sort_column','Data');descending=getattr(self,'_report_sort_descending',True)
        filtered=sorted_report_view(filtered,lambda r:report_sort_value(r,column,
            report_identity(r,self._matching_cached_bundle(r)),self._comparison_verification.get(self._comparison_key(r)),_pipeline_p95(r),
            phone_session_metadata(r,self._matching_cached_bundle(r))),descending)
        def table_line(r):
            info=report_identity(r, self._matching_cached_bundle(r))
            repeat=report_repeat(r)[0]
            key=self._comparison_key(r)
            state=self._comparison_verification.get(key)
            from .mobile_analysis import selection_identity
            try:selection_identity(r,self._matching_cached_bundle(r));ready=True
            except ValueError:ready=False
            status=('Zatwierdzona\npełna ocena' if state and state.ready else
                'Zatwierdzona\nogr. miary' if state and state.state=='APPROVED_LIMITED' else state.label if state else 'Nieustalona')
            status = status.replace(' · ', '\n')
            from .mobile_report_source_recovery import stored_source
            try:source_available=Path(stored_source(r)).is_file()
            except OSError:source_available=False
            if not source_available:status='Nieczytelne\nźródło'
            tick="☑" if key in self._comparison_checked else "☐" if ready else "—"
            tag="success" if state and state.approved else "warning" if state and state.state in ("STALE","NEEDS_EXPORT","IN_PROGRESS") else "muted"
            stamp=report_measurement_time(r)
            if stamp and stamp.tzinfo:
                offset=stamp.strftime('%z');offset=offset[:3]+':'+offset[3:]
                date=stamp.strftime('%d.%m.%Y\n%H:%M ')+offset
            else:date=stamp.strftime('%d.%m.%Y\n%H:%M bez strefy') if stamp else '—'
            scenario=info['scenario'] or '—'
            if len(scenario)>11 and '_' in scenario:scenario='\n'.join(scenario.rsplit('_',1))
            return (tick,phone_list_caption(phone_session_metadata(r,self._matching_cached_bundle(r))),date,
                    info["variant_short"],scenario,repeat,
                    status,_format_ms(_pipeline_p95(r)),{"tag":tag})
        tree=self.report_table.tree
        previous_selection=tree.selection()
        previous_focus=tree.focus()
        mapping={'report_'+hashlib.sha256(json.dumps(report_view_key(r),ensure_ascii=False).encode()).hexdigest():r for r in filtered}
        self._suppress_report_select=True
        try:
            for iid in tree.get_children():
                if iid not in mapping:tree.delete(iid)
            self.report_by_iid=mapping
            for index,(iid,r) in enumerate(mapping.items()):
                row=table_line(r);values=tuple(str(v) for v in row[:-1])
                tags=(row[-1]['tag'],'session_odd' if index%2 else 'session_even')
                if tree.exists(iid):tree.item(iid,values=values,tags=tags)
                else:tree.insert('',tk.END,iid=iid,values=values,tags=tags)
                tree.move(iid,'',index)
        finally:self._suppress_report_select=False
        ready_count=sum(1 for state in self._comparison_verification.values() if state.approved)
        self.report_count_var.set(f"{len(filtered)} z {len(all_reports)} raportów · {ready_count} zatwierdzonych"
            +f" · koszyk: {len(self._comparison_checked)}")
        if not filtered:
            if getattr(self,'current_report',None) is not None and all_reports:
                self.report_count_var.set(self.report_count_var.get()+' · podgląd poza filtrem')
                return
            if all_reports:
                self._show_report_hint(len(all_reports))
                self.summary_table.set_rows([("Brak dopasowań", query or "—", "Wyczyść filtr, aby wrócić do pełnej listy.")])
            else:
                self._show_empty()
            return
        current=getattr(self,'current_report',None)
        if not (auto_select or select_report_id or preserve_preview or current):
            self._show_report_hint(len(all_reports))
            return
        selected_iid=next((iid for iid,r in mapping.items() if select_report_id and r.report_id==select_report_id),None)
        if selected_iid is None and current is not None:
            selected_iid=next((iid for iid,r in mapping.items() if report_view_key(r)==report_view_key(current)),None)
            if selected_iid is None:
                selected_iid=next((iid for iid,r in mapping.items() if self._comparison_key(r)==self._comparison_key(current)),None)
            if selected_iid is None:
                self.report_count_var.set(self.report_count_var.get()+' · podgląd poza filtrem');return
        if selected_iid is None:
            if preserve_preview or not auto_select:return
            selected_iid=next(iter(mapping))
        self._suppress_report_select = True
        try:
            preserve_group=not select_report_id and bool(previous_selection or preserve_preview or current is not None)
            surviving=[iid for iid in previous_selection if iid in mapping]
            tree.selection_set(surviving if preserve_group else selected_iid)
            focus=previous_focus if preserve_group and previous_focus in mapping else selected_iid
            tree.focus(focus)
            tree.see(focus)
        except Exception:
            pass
        finally:
            self._suppress_report_select = False
        selected_report=mapping[selected_iid]
        if refresh_preview or select_report_id or current is None or report_view_key(selected_report)!=report_view_key(current):
            self._show_report(selected_report,self._matching_cached_bundle(selected_report))

    def _ux_reset_selection(self) -> None:
        if hasattr(self, 'phone_note_text'):
            self._replace_text(self.phone_note_text, '')
        for name, text in (("hero_title_var", "Wybierz raport z listy"),
                           ("hero_meta_var", ""), ("selected_report_info_var", "Wybierz sesję, aby zobaczyć jej identyfikatory."),
                           ("review_status_var", "Weryfikacja Desktop: brak wybranej sesji")):
            var = getattr(self, name, None)
            if var is not None:
                var.set(text)
        self._ux_review_snapshot = {}
        self._metric_cards = {}
        for var in getattr(self, 'card_hint_vars', {}).values():
            var.set('')
        if hasattr(self, '_metric_help'):
            self._metric_help.hide()
        try:
            self.btn_report_warnings.configure(state="disabled", text="Ostrzeżenia")
            self.btn_report_warnings.grid_remove()
        except Exception:
            pass

    def _resize_report_details(self, event) -> None:
        try:
            width = max(400, int(event.width))
            self.hero_title_label.configure(wraplength=max(260, width - 40))
            self.hero_meta_label.configure(wraplength=max(260, width - 40))
            self.review_status_label.configure(wraplength=max(260, width - 30))
            # Two rows when the right panel is narrow; four cards in one row otherwise.
            narrow = width < 830
            for index, key in enumerate(("integrity", "frames", "quality", "latency")):
                frame = getattr(self, "metric_card_frames", {}).get(key)
                if frame is None:
                    continue
                column = index % 2 if narrow else index
                row = index // 2 if narrow else 0
                frame.grid_configure(row=row, column=column,
                                     sticky="nsew", padx=(0 if column == 0 else 6, 0),
                                     pady=(0 if row == 0 else 6, 0))
                wrap = max(110, int(width / (2 if narrow else 4)) - 44)
                label = self.card_labels.get(key)
                if label is not None:
                    label.configure(wraplength=wrap)
                for child in frame.winfo_children():
                    if isinstance(child, tk.Label):
                        child.configure(wraplength=wrap)
            for col in range(4):
                self.card_shell.grid_columnconfigure(
                    col, weight=1 if (col < 2 or not narrow) else 0,
                    uniform='report_cards' if (col < 2 or not narrow) else '')
        except Exception:
            pass

    def _show_validation_details(self) -> None:
        bundle = getattr(self, "current_bundle", None)
        if not bundle:
            messagebox.showinfo("Integralność źródła", "Aby odczytać ostrzeżenia, otwórz oryginalne archiwum z telefonu.", parent=self.window)
            return
        validation = bundle.validation
        errors = [str(message) for message in validation.errors]
        warnings = [str(message) for message in validation.warnings]
        if not errors and not warnings:
            body = "Brak ostrzeżeń i błędów walidacji archiwum."
        else:
            body = "\n\n".join([*("BŁĄD: " + m for m in errors[:8]), *("UWAGA: " + m for m in warnings[:12])])
        messagebox.showinfo("Kontrola integralności archiwum", body, parent=self.window)

    def _on_report_selected(self, _event=None) -> None:
        if self._suppress_report_select:
            return
        tree=self.report_table.tree
        selection=tree.selection()
        if not selection:
            return
        focus=tree.focus()
        report=self.report_by_iid.get(str(focus if focus in self.report_by_iid else selection[0]))
        if report:
            current=getattr(self,'current_report',None)
            if current is not None and self._comparison_key(current)==self._comparison_key(report):return
            self._show_report(report,self._matching_cached_bundle(report))

    @staticmethod
    def _comparison_key(report) -> tuple[str,str]:
        return (str(report.report_id),str(report.source_archive_sha256).lower())



    def _toggle_comparison_checkbox(self,event):
        tree=self.report_table.tree
        if tree.identify_region(event.x,event.y)!='cell' or tree.identify_column(event.x)!='#1':return None
        iid=tree.identify_row(event.y);report=self.report_by_iid.get(iid)
        if report is None:return 'break'
        from .mobile_report_selection import toggle_checks
        return toggle_checks(self,[iid])

    def _clear_comparison_checks(self):
        from .mobile_analysis import selection_identity
        self._comparison_checked.clear()
        for iid,report in self.report_by_iid.items():
            try:selection_identity(report,self._matching_cached_bundle(report));symbol='☐'
            except ValueError:symbol='—'
            self.report_table.tree.set(iid,'Wybor',symbol)
        self._update_comparison_basket_status()

    def _update_comparison_basket_status(self,*,silent=False):
        count=len(self._comparison_checked)
        self.btn_compare_open.configure(text=f'Porównaj wybrane ({count})',state='normal' if count>=2 and not self._comparison_busy else 'disabled')
        if hasattr(self,'report_count_var'):
            text=self.report_count_var.get().split(' · koszyk:')[0]
            self.report_count_var.set(text+f' · koszyk: {count}')
        if not silent:self._set_status(f'Wybrano {count} sesji do analizy. Dostępność źródeł i każdej miary zostanie sprawdzona w module.')

    def _open_mobile_comparison(self):
        reports=[r for r in self._ordered_reports if self._comparison_key(r) in self._comparison_checked]
        if len(reports)<2:self._set_status('Zaznacz co najmniej dwie różne sesje.','warning');return
        self._launch_unified_analysis(reports)

    def _launch_unified_analysis(self,reports):
        if self._comparison_busy:return
        from .mobile_analysis import make_input,analyze
        from .mobile_report_source_recovery import stored_source
        try:
            inputs=[make_input(r,source_path=str(getattr(self._matching_cached_bundle(r),'path','') or stored_source(r)),
                bundle=self._matching_cached_bundle(r)) for r in reports]
        except ValueError as exc:self._set_status(str(exc),'warning');return
        self._comparison_busy=True;self._update_comparison_basket_status(silent=True)
        self._set_status('Sprawdzam źródła i dostępność poszczególnych miar…')
        def worker():
            try:
                cohort=analyze(inputs,'F4')
                families = {s.metadata.get('experiment') for s in cohort.sessions if s.source_status == 'VERIFIED'}
                if families == {'F5A'}:
                    cohort = analyze(inputs, 'F5A')
                elif not any(s.metadata.get('experiment')=='F4' for s in cohort.sessions):cohort=analyze(inputs,'DIAGNOSTIC')
                failure=None
            except Exception as exc:cohort=None;failure=str(exc)
            def done():
                self._comparison_busy=False;self._update_comparison_basket_status(silent=True)
                if failure:self._set_status('Nie otwarto analizy: '+failure,'warning');return
                from .mobile_analysis_gui import UnifiedAnalysisWindow
                roots={i.results_root for i in inputs if i.results_root}
                root=next(iter(roots)) if len(roots)==1 else None
                dialog=UnifiedAnalysisWindow(self.window,cohort,root);self._comparison_dialogs.append(dialog)
                self._set_status(f'Analiza {len(cohort.sessions)} źródeł. Wyniki opisowe; n i braki osobno dla każdej miary.')
            self._run_on_ui(done)
        threading.Thread(target=worker,daemon=True,name='alpr-unified-analysis').start()

    def _export_mobile_report_figures(self) -> None:
        current=getattr(self,"current_report",None)
        if current is None:
            self._set_status("Wybierz pojedynczą sesję do eksportu wykresów.","warning")
            return
        self._launch_mobile_comparison([current],comparison_scope='single')

    def _launch_mobile_comparison(self, reports, *, comparison_scope=None) -> None:
        """Resolve source archives in a worker, only use verified imported identities."""
        if self._comparison_busy:
            return
        if not reports:
            self._set_status("Nie wybrano żadnej sesji do porównania.","warning")
            return
        from .mobile_comparison_scope import scope_value,assess_import,validate_comparison
        scope=scope_value(comparison_scope,len(reports))
        self._comparison_busy=True
        for name in ("btn_compare_open","btn_export_mobile_figures"):
            getattr(self,name).configure(state="disabled")
        self._set_status(f"Sprawdzam źródła {len(reports)} sesji i przygotowuję wykresy…")

        from .mobile_report_source_recovery import restore_source, stored_source
        from .mobile_import_comparison import record_from_import, infer_results_root

        def worker() -> None:
            try:
                return prepare_worker()
            except Exception as error:
                description=f"Nie udało się przygotować porównania: {type(error).__name__}: {error}"
                def failed():
                    self._comparison_busy=False
                    self.btn_export_mobile_figures.configure(state="normal")
                    self._update_comparison_basket_status(silent=True)
                    self._set_status(description[:220],"warning")
                    messagebox.showerror("Błąd porównywarki",description[:500],parent=self.window)
                self._run_on_ui(failed)

        def prepare_worker() -> None:
            resolved=[]; failures=[]
            for report in reports:
                if self._closing:
                    return
                bundle=self._matching_cached_bundle(report)
                if bundle is None:
                    path=stored_source(report)
                    if not path:
                        failures.append(f"{report.report_id}: nie podłączono źródłowego archiwum.")
                        continue
                    try:
                        result=restore_source(report,path)
                        if not result.ok or result.bundle is None:
                            failures.append(f"{report.report_id}: {result.detail[:180]}")
                            continue
                        bundle=result.bundle
                    except (OSError,ValueError,RuntimeError,KeyError) as exc:
                        failures.append(f"{report.report_id}: {exc}")
                        continue
                resolved.append((report,bundle))
            root=infer_results_root(resolved) if resolved else None
            prepared=[]
            for report,bundle in resolved:
                try:
                    if scope=='single':session,traces,warnings=record_from_import(report,bundle,root)
                    else:
                        access=assess_import(report,bundle.path,root)
                        if not access.allowed(scope):raise ValueError(access.reason(scope))
                        session,traces,warnings=access.record,list(access.traces),access.record.comparison_source['warnings']
                    prepared.append((report,bundle,session,traces,warnings))
                except (OSError,ValueError,RuntimeError,KeyError) as exc:
                    failures.append(f"{report.report_id}: {exc}")
            if prepared and not failures:
                try:validate_comparison([v[2] for v in prepared],[v[3] for v in prepared],comparison_scope=scope,require_sources=True)
                except (OSError,ValueError,RuntimeError,KeyError) as exc:failures.append(str(exc))
            def done():
                self._comparison_busy=False
                self.btn_export_mobile_figures.configure(state="normal")
                self._update_comparison_basket_status(silent=True)
                if failures:
                    messagebox.showwarning("Niepełne źródła porównania",
                        "Wybór nie został otwarty. Uzupełnij/ponownie podłącz archiwa:\n\n"+"\n".join(failures[:8]),parent=self.window)
                    self._set_status("Porównanie wstrzymane — brak potwierdzenia tożsamości wszystkich źródeł.","warning")
                    return
                if not prepared:
                    self._set_status("Brak zweryfikowanych danych do wykresów.","warning")
                    return
                for report,bundle,_,_,_ in prepared:
                    self.bundle_by_report_id[report.report_id]=bundle
                from .mobile_import_comparison_gui import ImportedMobileComparisonWindow
                dialog=ImportedMobileComparisonWindow(self.window,
                    [v[2] for v in prepared],[v[3] for v in prepared],root,
                    imported_sources=[(v[0],v[1]) for v in prepared],comparison_scope=scope)
                self._comparison_dialogs.append(dialog)
                all_warnings=[x for v in prepared for x in v[4]]
                if all_warnings:
                    self._set_status("Porównywarka otwarta. Uwagi źródłowe: "+"; ".join(all_warnings[:2]),"warning")
                else:
                    self._set_status(f"Porównywarka: {len(prepared)} sesji. Wyniki źródłowe bez zmian.","success")
            self._run_on_ui(done)
        threading.Thread(target=worker,daemon=True,name="alpr-mobile-compare-source-read").start()

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
            if getattr(self, "_folder_import_handled", False):
                saved_id = getattr(self, "_folder_import_saved_id", "")
                if saved_id:
                    self._refresh_report_list(saved_id)
                self._set_status("Zamknięto dialog importu. Zapisane sesje pozostają w magazynie." if saved_id else "Zamknięto sprawdzanie folderu. Magazyn raportów bez zmian.", "info")
                return
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
            successful_paths: list[Path] = []
            saved = 0
            already_existing = 0
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
                                    validate_known_mobile_experiment(bundle)
                                    existing = self.store.existing_import(bundle.report)
                                    if existing is not None:
                                        already_existing += 1
                                        successful_paths.append(report_path)
                                        bundles[-1] = (report_path, replace(bundle, report=existing))
                                        continue
                                    self.store.reports = [
                                        _compact_mobile_report_for_store(report)
                                        for report in self.store.reports
                                    ]
                                    self.store.add_report(_compact_mobile_report_for_store(bundle.report, bundle=bundle), save=False)
                                except Exception as exc:
                                    failures.append((report_path, f"Raport odczytany, ale nie udało się go przygotować do zapisu: {exc}"))
                                    continue
                                saved += 1
                                successful_paths.append(report_path)
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
                        mobile_import_preferences.remember_successful_import(successful_paths)
                        self._refresh_report_list(last_saved_report_id)
                    elif bundles:
                        self._show_report(bundles[-1][1].report, bundles[-1][1])
                    else:
                        self._show_empty()

                    failed_count = len(failures)
                    invalid_count = len(invalid)
                    if already_existing and not failed_count and not invalid_count:
                        if not saved:
                            mobile_import_preferences.remember_successful_import(successful_paths)
                        self._set_status(f"Zaimportowano {saved} nowych raportów. Już istniejące: {already_existing}; zachowano weryfikację.", "success")
                        return
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
        self._science_sequence += 1
        if hasattr(self, "science_panel"):
            self.science_panel.render({"status": "Wybierz sesję", "progress": "", "progress_subtitle": "", "cards": (), "performance": {}, "warnings": (), "source": "", "frames": ""})
        self._ux_reset_selection()
        self.current_bundle = None
        if hasattr(self, "btn_link_archive"):
            self.btn_link_archive.configure(state="disabled")
            self.btn_link_archive.grid_remove()
        if hasattr(self, "source_status_var"):
            self.source_status_var.set("")
        self.current_report = None
        self._latency_report_identity = None
        self._latency_x_view = None
        self._latency_full_x = None
        self._latency_pan_origin = None
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
        self._science_sequence += 1
        if hasattr(self, "science_panel"):
            self.science_panel.render({"status": "Wybierz sesję", "progress": "", "progress_subtitle": "", "cards": (), "performance": {}, "warnings": (), "source": "", "frames": ""})
        self._ux_reset_selection()
        self.current_bundle = None
        if hasattr(self, "btn_link_archive"):
            self.btn_link_archive.configure(state="disabled")
            self.btn_link_archive.grid_remove()
        if hasattr(self, "source_status_var"):
            self.source_status_var.set("")
        self.current_report = None
        self._latency_report_identity = None
        self._latency_x_view = None
        self._latency_full_x = None
        self._latency_pan_origin = None
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

    def _matching_cached_bundle(self, report):
        """Do not show old details for another archive with a reused report ID."""
        cached = self.bundle_by_report_id.get(report.report_id)
        return cached if (cached is not None and
                          str(cached.source_archive_sha256).lower() == str(report.source_archive_sha256).lower() and
                          str(cached.report.report_id) == str(report.report_id)) else None

    def link_source_archive(self) -> None:
        """Let the operator relocate a missing archive; never silently match names."""
        report = getattr(self, "current_report", None)
        if report is None:
            return
        original = stored_source(report)
        initial = Path(original).parent if original else Path.home()
        if not initial.is_dir():
            initial = Path.home()
        filename = filedialog.askopenfilename(
            parent=self.window, title="Wskaż oryginalne archiwum tej sesji ALPR",
            initialdir=str(initial),
            filetypes=[("Sesje ALPR", "*.alprsession *.zip"), ("Wszystkie pliki", "*.*")],
        )
        if filename:
            self._begin_source_recovery(report, filename, manual=True)

    def _begin_source_recovery(self, report: MobileBenchmarkReport, path: str, *, manual: bool = False) -> None:
        """SHA/manifest verification can be expensive; never block the Tk thread."""
        if self._closing:
            return
        report_id = str(report.report_id)
        key = (report_id, str(report.source_archive_sha256 or ""), str(path))
        if not manual and key in self._source_attempts:
            return
        self._source_attempts.add(key)
        self._source_sequence += 1
        token = self._source_sequence
        self._source_load_tokens[report_id] = token
        if getattr(getattr(self, "current_report", None), "report_id", None) == report_id:
            self.source_status_var.set("Sprawdzam SHA-256 oraz tożsamość archiwum…")
        def worker() -> None:
            result = restore_source(report, path)
            self._run_on_ui(lambda result=result: self._finish_source_recovery(report, result, token, manual=manual))
        threading.Thread(target=worker, daemon=True, name="alpr-source-verify").start()

    def _finish_source_recovery(self, report, result, token: int, *, manual: bool) -> None:
        report_id = str(report.report_id)
        if self._source_load_tokens.get(report_id) != token:
            return  # A newer, explicitly selected source takes precedence.
        selected_report = getattr(self, "current_report", None)
        active = (selected_report is not None and selected_report.report_id == report_id
                  and selected_report.source_archive_sha256 == report.source_archive_sha256)
        if not result.ok:
            self._source_failures[report_id] = result.detail
            if active:
                self.source_status_var.set(result.detail + "  Użyj «Podłącz archiwum».")
            if manual and active:
                messagebox.showwarning("Nie połączono archiwum", result.detail, parent=self.window)
            return
        self._source_failures.pop(report_id, None)
        self.bundle_by_report_id[report_id] = result.bundle
        if (hasattr(self, '_ordered_reports')
                and phone_session_metadata(report, result.bundle) != phone_session_metadata(report)):
            self._apply_report_filter(auto_select=False, preserve_preview=True)
        linked_report = report
        if manual and str(report.source_path or "") != result.path:
            linked_report = relinked_report(report, result.path)
            try:
                self.store.reports = [
                    linked_report if row.report_id == report_id and row.source_archive_sha256 == report.source_archive_sha256
                    else row for row in self.store.reports
                ]
                self.store.save()  # Only report locator metadata, not research artifacts.
                self._ordered_reports = [
                    linked_report if row.report_id == report_id and row.source_archive_sha256 == report.source_archive_sha256
                    else row for row in getattr(self, "_ordered_reports", [])
                ]
                self.report_by_iid = {
                    key: linked_report if row.report_id == report_id and row.source_archive_sha256 == report.source_archive_sha256
                    else row for key, row in self.report_by_iid.items()
                }
            except Exception as exc:
                # Verified details remain usable, but do not pretend the new location was saved.
                self._source_failures[report_id] = "Archiwum zweryfikowano, lecz nie zapisano nowej ścieżki: " + str(exc)[:140]
        if active:
            self._show_report(linked_report, result.bundle)
            n = len(result.bundle.trace_rows)
            crops = sum(entry.name.startswith("samples/crops/") for entry in result.bundle.entries)
            if report_id in self._source_failures:
                self.source_status_var.set(self._source_failures[report_id])
            else:
                self.source_status_var.set(
                    f"Integralność źródła potwierdzona (ZIP/SHA-256)  ·  {n} pomiarów czasu  ·  {crops} wyciętych tablic")

    def _show_report(self, report: MobileBenchmarkReport | None, bundle: MobileReportBundle | None = None) -> None:
        if report is None:
            self._show_empty()
            return
        if bundle is not None and (
            str(bundle.source_archive_sha256).lower() != str(report.source_archive_sha256).lower()
            or bundle.report.report_id != report.report_id
        ):
            bundle = None  # Legacy report-id cache must not cross archive identities.
        identity = (str(report.report_id), str(report.source_archive_sha256).lower())
        if self._latency_report_identity != identity:
            self._latency_report_identity = identity
            self._latency_x_view = None
            self._latency_full_x = None
            self._latency_pan_origin = None
            self._latency_effective_mode = None
            axis_var = getattr(self, '_latency_axis_mode_var', None)
            if axis_var is not None:
                axis_var.set('time')  # new session: prefer its real clock, if documented
        self.current_bundle = bundle
        self.current_report = report
        if bundle is not None:
            n = len(bundle.trace_rows)
            c = sum(e.name.startswith("samples/crops/") for e in bundle.entries)
            self.source_status_var.set(f"Integralność źródła potwierdzona (ZIP/SHA-256)  ·  {n} pomiarów czasu  ·  {c} wyciętych tablic")
        if hasattr(self, "btn_link_archive"):
            self.btn_link_archive.configure(state="normal")
            if bundle is None:
                self.btn_link_archive.grid()
            else:
                self.btn_link_archive.grid_remove()
        if bundle is None:
            self.source_status_var.set(self._source_failures.get(report.report_id,
                "Dane zbiorcze są dostępne. Do wykresów i zdjęć potrzebne jest oryginalne archiwum."))
        if hasattr(self, "btn_review_samples"):
            self.btn_review_samples.configure(state="normal" if (bundle or report.source_path) else "disabled")
        self._update_cards(report, bundle)
        self._populate_summary(report, bundle)
        self._populate_scientific_comparison(report, bundle)
        self._populate_config(report)
        self._populate_latency(report, bundle)
        self._populate_artifacts(report, bundle)
        self._populate_quality(report)
        self._populate_diagnostics(report, bundle)
        self._populate_crops(report, bundle)
        self._populate_raw(report, bundle)
        self._update_scientific_dashboard(report, bundle)
        self._sync_final_action()
        if bundle is None:
            original = stored_source(report)
            if original:
                self._begin_source_recovery(report, original)
            else:
                self.source_status_var.set("Nie zapisano ścieżki źródła. Użyj «Podłącz archiwum».")

    def open_sample_review(self) -> None:
        report = getattr(self, "current_report", None)
        if report is None:
            self._set_status("Wybierz sesję z listy raportów.", "warning")
            return
        from .final180_tv import authentic_final_bundle
        if authentic_final_bundle(getattr(self,'current_bundle',None)):
            self.open_final_analysis();return
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
        info = report_identity(report, bundle)
        from .mobile_model_identity import identity_from_bundle
        models = identity_from_bundle(bundle)
        from .mobile_measurement_availability import polish_model_caption
        phone = phone_session_metadata(report, bundle)
        self.hero_title_var.set(phone_name_caption(phone))
        if hasattr(self, 'phone_note_text'):
            self._replace_text(self.phone_note_text, phone_note_caption(phone))
        meta = (f"Seria: {info['series'] or 'brak danych'} · Wariant: {info['variant']} · "
                f"Powtórzenie: {report_repeat(report)[0]}\n"
                f"Scenariusz: {info['scenario'] or 'brak danych'} · {local_measurement_time(report.measured_at, compact=True)}")
        if bundle and bundle.validation.warnings:
            meta += f"  ·  {len(bundle.validation.warnings)} uwaga o archiwum (Szczegóły)"
        self.hero_meta_var.set(meta)
        self.selected_report_info_var.set(selected_report_caption(report, bundle))
        review_root = Path(getattr(CONFIG, "DIR_7_RANKINGS_MOBILE_PACKAGES", CONFIG.DIR_7_RANKINGS / "mobile_packages"))
        snapshot = desktop_status(report, bundle, review_root)
        self._ux_review_snapshot = snapshot
        self.review_status_var.set(f"{snapshot['title']}  •  {snapshot['detail']}")
        if bundle:
            warnings_count = len(bundle.validation.warnings) + len(bundle.validation.errors)
            self.btn_report_warnings.configure(state="normal" if warnings_count else "disabled",
                                               text=f"Wyjaśnij uwagi ({warnings_count})" if warnings_count else "Bez ostrzeżeń")
            if warnings_count:
                self.btn_report_warnings.grid()
            else:
                self.btn_report_warnings.grid_remove()
        else:
            self.btn_report_warnings.configure(state="disabled", text="Brak walidacji")
            self.btn_report_warnings.grid_remove()
        self.card_vars["integrity"].set(integrity)
        self.card_vars["frames"].set(f"{trace_total or '-'} klatek")
        self.card_vars["quality"].set(_quality_value(report))
        self.card_vars["latency"].set(_format_ms(_pipeline_p95(report)))
        try:
            self.card_labels["integrity"].configure(fg=tone)
        except Exception:
            pass

        self._metric_cards = cards_for_report(
            report, bundle, getattr(self, '_ux_review_snapshot', {}))
        for key, metric in self._metric_cards.items():
            self.card_vars[key].set(metric.value)
            self.card_hint_vars[key].set(metric.subtitle)
        try:
            self._resize_report_details(type('_Width', (), {'width': self.card_shell.winfo_width()})())
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
            ("Profil techniczny", report.variant_id, "Nazwa profilu runtime/kwantyzacji, inna niż wariant eksperymentu."),
            ("Pomiar", _format_datetime(report.measured_at), "Data i czas zapisane przez klienta mobilnego."),
            ("Urządzenie", _device_label(report), device_details(report)),
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
                    ("Wariant eksperymentu", str(index.get("experiment_variant") or "-"), "Wariant A/B/C/D w kampanii, nie techniczny profil runtime."),
                    ("Replika", str(index.get("replicate_index") or "-"), "Numer powtórzenia tego samego wariantu."),
                ]
            )
        review = getattr(self, "_ux_review_snapshot", {}) or {}
        q = dict(report.quality or {})
        phone_status = ("Nieustalone (raport obejmuje dane Desktop)"
                        if q.get("quality_source") == "human_review" else
                        "Brak GT w archiwum" if q.get("available") is False else
                        "Obecne w raporcie" if q.get("available") is True else "Nieustalone")
        rows = [
            ("GT z telefonu", phone_status, "Dotyczy tylko jakości zapisanej w źródłowym archiwum Androida."),
            ("Weryfikacja Desktop", review.get("title", "Nieustalone"), review.get("detail", "Brak odczytu sidecara.")),
        ] + rows
        from .mobile_model_identity import identity_from_bundle
        identity = identity_from_bundle(bundle)
        rows.append(("Nazwa pakietu MT+MZ", identity["package_name"] or "brak danych źródłowych",
                     "Metadane zapisane w źródłowym pomiarze telefonu."))
        for role, title in (("plate", "MT"), ("character", "MZ")):
            model = identity["models"][role]
            for key in ("model_display_id", "model_id", "variant_id", "checkpoint_sha256", "package_sha256"):
                rows.append((f"{title} · {key}", model[key] if model["verified"] and model[key] else
                             "brak danych źródłowych", "Zapisane odniesienie modelu telefonu."))
        self.summary_table.set_rows(validation_rows + rows[:2] + summary_measure_rows(report, bundle) + rows[2:])

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
        if isinstance(value, str):
            return value.strip() or "-"
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
        sizes = model_input_sizes(report)
        frame = f"klatka {width}×{height}" if width and height else ""
        if frame and fps:
            frame += f" @ {fps} fps"
        return "; ".join(item for item in (frame, sizes) if item) or "-"

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
        if stage_enabled(report, role_key) is False:
            return "Wyłączony"
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
            if value.get("enabled") is False:
                return "Wyłączony"
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
            # An enabled/disabled descriptor is not a cryptographic model ID.
            return "-"
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
                    "Wspólna seria" if scenario_id else "Niepełne",
                    "Ta sama seria/scenariusz to filtr kandydatów, nie zgoda na porównanie naukowe. Warunki wykonania wymagają osobnej kontroli.",
                    {"tag": "warning"},
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
                "Rozdzielczość modeli / kamery",
                lambda candidate, index: self._guard_resolution_value(candidate, index),
                "Rozróżniamy wejścia MT/MZ od klatki kamery. Nieustalone wymiary blokują bezwarunkowe porównanie.",
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
                "Model pojazdów (MP)",
                lambda candidate, index: self._guard_model_fingerprint_value(candidate, index, "mp"),
                "Wyłączony MP to brak użytego modelu pojazdów, nie jego fingerprint.",
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
            if label == "Model pojazdów (MP)" and values and all(value == "Wyłączony" for value in values):
                status = "Nie używano"
                tag = "muted"
                series_value = "MP wyłączony"
            elif selected_value in ("", "-"):
                status = "Brak danych"
                tag = "warning"
                series_value = "-"
            elif require_same and len(non_empty_values) != len(values):
                status = "Brak danych"
                tag = "warning"
                series_value = f"{len(non_empty_values)}/{len(values)} raportów"
            elif require_same and len(cohort) < 2:
                status = "Jedna sesja"
                tag = "muted"
                series_value = selected_value
            elif require_same and len(distinct) > 1:
                status = "Różne"
                tag = "error"
                series_value = f"{len(distinct)} wartości"
            elif require_same:
                status = "Spójne konfiguracje"
                tag = "muted"
                series_value = selected_value
            else:
                status = "Jest" if selected_value not in ("", "-") else "Brak"
                tag = "success" if selected_value not in ("", "-") else "warning"
                series_value = f"{len(non_empty_values)}/{len(cohort)} raportów"
            rows.append((label, selected_value, series_value, status, description, {"tag": tag}))

        self.comparison_table.set_rows(rows)

    def _populate_scientific_comparison(self, report: MobileBenchmarkReport, bundle: MobileReportBundle | None) -> None:
        """Evidence-sensitive comparisons; matching series alone never approves research parity.

        The older configuration-guard remains available for compatibility, but
        the operator's comparison tab uses the stricter, read-only analysis.
        """
        try:
            analysis = assess_cohort(report, getattr(self.store, 'reports', ()), bundle=bundle)
            self.comparison_table.set_rows(list(analysis['rows']))
        except Exception:
            logger.exception('Cannot assess scientific session comparability')
            self.comparison_table.set_rows([
                ('Weryfikacja porównywalności', 'Nie ustalono', '—', 'BRAK POTWIERDZENIA',
                 'Błąd odczytu warunków. Nie traktuj raportu jako porównywalnego z innymi sesjami.',
                 {'tag': 'error'})
            ])

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
        for key, label in (("mt", "MT: model tablic"), ("mz", "MZ: model znaków"),
                           ("pipeline", "ALPR: cały proces")):
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
        desktop = getattr(self, "_ux_review_snapshot", {}) or {}
        desktop_rows = [
            ("Weryfikacja Desktop", desktop.get("title", "Nieustalone"),
             desktop.get("detail", "Stan weryfikacji jest niezależny od GT w archiwum.")),
        ]
        if quality.get("quality_source") == "human_review":
            def ratio(key):
                value = quality.get(key)
                return "—" if value is None else f"{float(value) * 100:.2f}%"
            self.quality_table.set_rows(desktop_rows + [
                ("Źródło jakości", "Weryfikacja człowieka", "Ukończona weryfikacja; surowe pomiary czasu i pamięci zachowane."),
                ("Tryb weryfikacji", "Zaślepiona GT" if quality.get("review_mode") == "blinded_gt_v1" else "Asystowana", "Tryb zapisany w sidecarze weryfikacji."),
                ("Poprawne odczyty MZ", ratio("exact_read_rate"), "Jednostka: zweryfikowany obraz tablicy."),
                ("Brak odczytu MZ", ratio("no_read_rate"), "Zweryfikowany obraz tablicy z GT, bez użytecznego odczytu."),
                ("CER MZ", ratio("cer"), "Suma błędów / suma znaków GT. Wartość może przekraczać 100%."),
                ("Skuteczność ALPR dla tablic", ratio("subject_success_rate"), "Co najmniej jeden poprawny odczyt lub wynik konsensusu."),
                ("Skuteczność lokalizacji MT", ratio("mt_localization_success_rate"), "Zweryfikowane wywołania z jedną widoczną tablicą; nie jest to mAP."),
                ("SHA-256 źródła", quality.get("source_archive_sha256", ""), "Tożsamość archiwum powiązanego z weryfikacją."),
            ])
            return
        if not quality:
            self.quality_table.set_rows(desktop_rows + [("Brak jakości", "-", "Raport nie zawiera sekcji quality.")])
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
        self.quality_table.set_rows(desktop_rows + rows)

    def _populate_diagnostics(self, report: MobileBenchmarkReport, bundle: MobileReportBundle | None) -> None:
        rows: list[tuple[str, str, str]] = []
        errors = dict(report.errors or {})
        historical_crash = crash_description(errors)
        for field, value, extra in _flatten_rows(errors, prefix="errors", limit=24):
            if historical_crash and field == "errors.crash_count":
                rows.append((historical_crash[0], value,
                             "Odzyskane poprzednie, niekontrolowane pomiary. Nie jest to liczba awarii wybranego eksperymentu."))
            elif historical_crash and field == "errors.crash_semantics":
                rows.append(("Znaczenie licznika odzyskania", "Poprzednie sesje",
                             "Ta informacja jest metadanymi historii uruchomień. Nie przypisuj jej bieżącej sesji."))
            else:
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
        """Display verified *Desktop* review without rewriting Android history."""
        rows: list[tuple[str, str, str, str, str, str, str]] = []
        selected = self.crops_table.tree.selection()
        selected_id = str(selected[0]) if selected else ""
        self._crop_entry_rows = []
        self._crop_photo = None
        self.crop_image_label.configure(image="")
        if bundle is None:
            self.crop_preview_var.set("Połącz oryginalne archiwum, aby zobaczyć wycięte tablice.")
            rows.append(("—", "—", "—", "—", "—", "—", "Podłącz oryginalne archiwum"))
        else:
            key = self._science_cache_key(report, bundle)
            science = self._science_cache.get(key, {})
            overlay = science.get("_ui_desktop_samples", {}) if isinstance(science, dict) else {}
            state = overlay.get("state") if isinstance(overlay, dict) else None
            preview = overlay.get("samples", {}) if state == "current" else {}
            entry_names = {str(e.name) for e in bundle.entries}
            for row in bundle.sample_rows[:500]:
                entry = crop_entry_for_row(row, entry_names)
                capture_id = str(row.get("capture_id") or row.get("sample_id") or row.get("id") or "")
                decision = preview.get(capture_id) if isinstance(preview, dict) else None
                if isinstance(decision, dict):
                    status, gt = decision["status"], decision["gt"] or "—"
                elif state == "not_started":
                    status, gt = "Nie rozpoczęto w Desktop", "—"
                elif state == "error":
                    status, gt = "Weryfikacja wymaga sprawdzenia", "—"
                elif state == "current":
                    status, gt = "Brak powiązania weryfikacji", "—"
                elif key in self._science_error_by_key:
                    status, gt = "Nie można odczytać weryfikacji", "—"
                else:
                    status, gt = "Wczytuję weryfikację Desktop…", "—"
                rows.append((
                    capture_id or "—", row.get("track_id") or "—", status,
                    row.get("prediction") or row.get("text") or "—", gt,
                    row.get("captured_at_ms") or row.get("timestamp_ms") or "—",
                    "Obraz dostępny" if entry else "Brak obrazu",
                ))
                self._crop_entry_rows.append(entry)
            if not rows:
                for entry in sorted(e for e in entry_names if e.startswith("samples/crops/")
                                    and e.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))[:500]:
                    rows.append((Path(entry).stem, "—", "Brak identyfikatora GT", "—", "—", "—", "Obraz dostępny"))
                    self._crop_entry_rows.append(entry)
            if rows:
                if state == "error":
                    self.crop_preview_var.set("Nie można powiązać weryfikacji Desktop: " + str(overlay.get("error") or "błąd weryfikacji")[:180])
                elif key in self._science_error_by_key:
                    self.crop_preview_var.set("Nie można odczytać weryfikacji Desktop: " + self._science_error_by_key[key][:180])
                elif state == "not_started":
                    self.crop_preview_var.set("Brak zapisanej weryfikacji Desktop. Oryginalny status Androida pozostaje w archiwum.")
                elif state != "current":
                    self.crop_preview_var.set("Trwa odczyt weryfikacji Desktop…")
                else:
                    self.crop_preview_var.set("Status i GT: weryfikacja Desktop • obraz oraz odczyt: niezmienne archiwum Androida.")
            else:
                self.crop_preview_var.set("W zweryfikowanym archiwum nie znaleziono wyciętych tablic.")
                rows.append(("—", "—", "—", "—", "—", "—", "Brak wyciętych tablic"))
        self.crops_table.set_rows(rows)
        if selected_id and bundle is not None and selected_id in self.crops_table.tree.get_children(""):
            self.crops_table.tree.selection_set(selected_id)
            self._on_crop_selected()

    def _on_crop_selected(self, _event=None) -> None:
        """Render selected image from the archive, bounded by the shared reader."""
        try:
            selected = self.crops_table.tree.selection()
            if not selected or self.current_bundle is None:
                return
            index = int(str(selected[0]).removeprefix("row_"))
            if index < 0 or index >= len(self._crop_entry_rows):
                return
            entry = self._crop_entry_rows[index]
            if not entry:
                self.crop_preview_var.set("W tej próbie nie zapisano obrazu wyciętej tablicy.")
                self._crop_photo = None
                self.crop_image_label.configure(image="")
                return
            from PIL import ImageTk
            image = read_mobile_sample_image(self.current_bundle, entry, max_entry_bytes=5 * 1024 * 1024,
                                              max_dimension=8000, max_pixels=24_000_000)
            scale = min(560 / max(1, image.width), 142 / max(1, image.height), 3.0)
            if scale != 1.0:
                from PIL import Image
                image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                                     Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
            self._crop_photo = ImageTk.PhotoImage(image, master=self.window)
            self.crop_image_label.configure(image=self._crop_photo)
            self.crop_preview_var.set(f"Wycięta tablica · {Path(entry).name} · oryginał w zweryfikowanym archiwum")
        except Exception as exc:
            self._crop_photo = None
            self.crop_image_label.configure(image="")
            self.crop_preview_var.set("Nie można wyświetlić obrazu: " + str(exc)[:180])

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

    def _on_latency_connections_change(self) -> None:
        """Toggle line segments ONLY; never reset zoom, pan, axes or data."""
        self._draw_latency_chart()

    def _on_latency_axis_mode_change(self) -> None:
        # Only an EXPLICIT coordinate-system change can reset the x zoom.
        # The connect/disconnect checkbox does not touch this state.
        self._latency_x_view = None
        self._latency_pan_origin = None
        self._latency_hit_points = []
        self._draw_latency_chart()

    def _clear_latency_point_hover(self, _event=None):
        context_label = getattr(self, '_latency_context_var', None)
        if context_label is not None:
            baseline = getattr(self, '_latency_hover_baseline', '')
            if context_label.get() != baseline:
                context_label.set(baseline)

    def _on_latency_point_hover(self, event):
        candidates = getattr(self, '_latency_hit_points', [])
        if not candidates:
            return
        x, y = float(event.x), float(event.y)
        nearest = min(candidates, key=lambda item: (item[0] - x)**2 + (item[1] - y)**2)
        if (nearest[0] - x)**2 + (nearest[1] - y)**2 > 12**2:
            self._clear_latency_point_hover()
            return
        _, _, index, key, value = nearest
        bundle = getattr(self, 'current_bundle', None)
        if bundle is None or index >= len(bundle.trace_rows):
            return
        stamps = getattr(self, '_latency_time_seconds', None)
        offset = stamps[index] if stamps is not None and index < len(stamps) else None
        title = STAGE_DESCRIPTIONS.get(key, str(key))
        label = point_caption(bundle.trace_rows[index], offset, title, value,
                              getattr(self, '_latency_time_context', {}))
        context_label = getattr(self, '_latency_context_var', None)
        if context_label is not None and context_label.get() != label:
            context_label.set(label)

    def _zoom_latency_wheel(self, event):
        """Mouse wheel scales the X axis around the cursor, not the original data."""
        full = self._latency_full_x
        if full is None or full[1] <= full[0]:
            return "break"
        number = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        inward = number == 4 or (number not in (4, 5) and delta > 0)
        outward = number == 5 or (number not in (4, 5) and delta < 0)
        if not (inward or outward):
            return "break"
        plot_width = max(1, int(self.latency_canvas.winfo_width()) - 68)
        fraction = (float(event.x) - 50.0) / plot_width
        self._latency_x_view = zoom_x_view(
            full, self._latency_x_view, fraction, zoom_in=inward,
            integer_axis=getattr(self, "_latency_axis_label", "") == "ID klatki")
        self._latency_pan_origin = None
        self._draw_latency_chart()
        return "break"

    def _begin_latency_pan(self, event):
        self._latency_press_xy = (float(event.x), float(event.y))
        self._latency_pan_origin = (float(event.x), self._latency_x_view) if self._latency_x_view is not None else None

    def _move_latency_pan(self, event):
        if self._latency_pan_origin is None or self._latency_full_x is None:
            return
        initial_x, initial_view = self._latency_pan_origin
        plot_width = max(1, int(self.latency_canvas.winfo_width()) - 68)
        self._latency_x_view = pan_x_view(self._latency_full_x, initial_view,
                                           float(event.x) - initial_x, plot_width)
        self._draw_latency_chart()
        return "break"

    def _end_latency_pan(self, _event):
        self._latency_pan_origin = None
        origin = getattr(self, '_latency_press_xy', None)
        link = self._evidence_timing_link() if getattr(self, '_linked_evidence_timing', None) else None
        if not origin or not link or (float(_event.x)-origin[0])**2+(float(_event.y)-origin[1])**2 > 16:
            return
        points = getattr(self, '_latency_hit_points', ())
        if not points:
            return
        point = min(points, key=lambda p: (p[0]-_event.x)**2+(p[1]-_event.y)**2)
        if (point[0]-_event.x)**2+(point[1]-_event.y)**2 <= 144:
            row = self.current_bundle.trace_rows[point[2]]
            if row.get('frame_id') not in (None, ''):
                link['callback'](str(row['frame_id']), point[3])

    def _evidence_timing_link(self):
        link = getattr(self, '_linked_evidence_timing', None)
        report = getattr(self, 'current_report', None)
        if link and report and str(report.source_archive_sha256).lower() == link['sha']:
            return link
        return None

    def link_evidence_timing(self, archive_sha, trace_ids, callback, *, open_tab=False):
        """Optional source-ID bridge; ordinary Z4 charts keep their own clock/zoom."""
        report = getattr(self, 'current_report', None)
        if report is None or str(report.source_archive_sha256).lower() != str(archive_sha).lower():
            return False
        ids = frozenset(str(v) for v in trace_ids)
        previous = self._evidence_timing_link()
        self._linked_evidence_timing = dict(sha=str(archive_sha).lower(), ids=ids, callback=callback)
        if open_tab:
            self.notebook.select(3)
            self.technical_notebook.select(self.latency_split.master)
            # The selected source points must not remain outside an old zoom.
            self._latency_x_view = None
            self.window.lift()
            self.window.after_idle(self._reveal_linked_timing_points)
        if open_tab or previous is None or previous['ids'] != ids:
            self._draw_latency_chart()
        return True

    def _reveal_linked_timing_points(self):
        if not self._evidence_timing_link() or not self.window.winfo_exists():
            return
        split = self.latency_split
        # The existing splitter can otherwise leave only a few pixels below
        # the legend on first opening a hidden tab. Keep the points clickable.
        if len(split.panes()) > 1 and split.winfo_height() > 320 and self.latency_canvas.winfo_height() < 260:
            split.sashpos(0, int(split.winfo_height() * .74))
        self._draw_latency_chart()

    def _reset_latency_zoom(self, _event=None):
        self._latency_x_view = None
        self._latency_pan_origin = None
        self._draw_latency_chart()
        return "break"

    def _draw_latency_chart(self) -> None:
        """Scientific per-frame plot: labeled X axis and lossless visible extrema.

        For short sessions every measured point is shown, using the real frame
        ID (or the honest timestamp/sample-number fallback). Long sessions
        use pixel-column min/max envelopes, never stride-skipping spikes.
        """
        canvas = getattr(self, "latency_canvas", None)
        if canvas is None:
            return
        try:
            width = max(320, int(canvas.winfo_width() or 760))
            height = max(150, int(canvas.winfo_height() or 180))
            self._latency_hit_points = []
            canvas.delete("all")
            canvas.create_rectangle(0, 0, width, height, fill=self.panel, outline=self.border)
            footer = getattr(self, "_latency_axis_footer_var", None)
            bundle = self.current_bundle
            if bundle is None or not bundle.trace_rows:
                self._latency_hit_points = []
                self._latency_time_seconds = None
                self._latency_hover_baseline = ''
                clock_radio = getattr(self, '_latency_time_axis_radio', None)
                if clock_radio is not None:
                    clock_radio.configure(state='disabled')
                context_label = getattr(self, '_latency_context_var', None)
                if context_label is not None:
                    context_label.set('')
                if footer is not None:
                    footer.set("")
                canvas.create_text(width / 2, height / 2,
                                   text=("Połącz oryginalne archiwum, aby zobaczyć czasy klatek."
                                         if bundle is None else "W archiwum nie ma szczegółowego śladu czasu."),
                                   fill=self.muted, font=("Segoe UI", 10, "bold"))
                return
            # Distinct, constant scientific colors and marker shapes are
            # independent of the rest of the olive application theme.
            specs = tuple((key, style) for key, style in LATENCY_SERIES_STYLES.items())
            timeline = timeline_stages(bundle.trace_rows, tuple(item[0] for item in specs))
            context = session_time_context(bundle, getattr(self, 'current_report', None))
            stamps = session_elapsed_seconds(bundle.trace_rows, context)
            self._latency_time_context = context
            self._latency_time_seconds = stamps
            caption = session_caption(context)
            self._latency_hover_baseline = caption
            label = getattr(self, '_latency_context_var', None)
            if label is not None and label.get() != caption:
                label.set(caption)
            radio = getattr(self, '_latency_time_axis_radio', None)
            if radio is not None:
                radio.configure(state='normal' if stamps is not None else 'disabled')
            if stamps is None:
                axis_choice = getattr(self, '_latency_axis_mode_var', None)
                if axis_choice is not None and axis_choice.get() == 'time':
                    axis_choice.set('frame')
            xs = timeline["x"]
            stage_values = timeline["stages"]
            mp_off = stage_enabled(getattr(self, "current_report", None), "mp") is False
            present = [(key, style, stage_values[key]) for key, style in specs
                       if (key != "vehicle_inference_ms" or not mp_off)
                       and any(value is not None for value in stage_values[key])]
            if not present:
                if footer is not None:
                    footer.set("")
                canvas.create_text(width / 2, height / 2,
                                   text="W archiwum nie ma czasów poszczególnych etapów.",
                                   fill=self.muted, font=("Segoe UI", 10))
                return
            max_y = max(1.0, max(value for _, _, seq in present for value in seq if value is not None))
            # Sparse frame-ID series use categorical positions automatically.
            # Display X mapping and zoom are independent of line visibility.
            # Source frame IDs remain intact and visible as axis tick labels.
            use_even_spacing = has_sparse_frame_gap(xs, timeline["axis_label"])
            connections = getattr(self, "_latency_connect_neighbors_var", None)
            # Adjacency is ALWAYS checked against original frame IDs, even
            # when points are arranged by actual elapsed session time.
            connect_consecutive = (bool(connections.get()) if connections is not None else True) and (timeline["axis_label"] == "ID klatki")
            mode_var = getattr(self, '_latency_axis_mode_var', None)
            prefer_time = mode_var is not None and mode_var.get() == 'time'
            time_axis = prefer_time and stamps is not None
            effective_mode = 'time' if time_axis else 'frame'
            previous_mode = getattr(self, '_latency_effective_mode', None)
            if previous_mode is not None and previous_mode != effective_mode:
                self._latency_x_view = None
                self._latency_pan_origin = None
            self._latency_effective_mode = effective_mode
            if time_axis:
                plot_xs = tuple(stamps)
                source_min_x, source_max_x = 0.0, max(stamps[-1], context['duration_ms'] / 1000.0)
                self._latency_axis_label = 'Czas sesji [s]'
            else:
                plot_xs = display_positions(xs, even_spacing=use_even_spacing)
                source_min_x, source_max_x = min(plot_xs), max(plot_xs)
                self._latency_axis_label = timeline['axis_label']
            self._latency_full_x = (source_min_x, source_max_x)
            self._latency_display_even_spacing = use_even_spacing and not time_axis
            self._latency_x_view = clamp_x_view(self._latency_full_x, self._latency_x_view)
            min_x, max_x = self._latency_x_view or self._latency_full_x
            span_x = max_x - min_x
            # Z4MOBREP036: full scientific series names need their own space.
            # The legend and gesture instructions are two separate rows;
            # no legend label may share a pixel row with the right-aligned hint.
            # For narrow panes, wrap the legend into two or three rows instead
            # of drawing several texts on top of one another.
            pad_l, pad_r, pad_b = 50, 18, 46
            plot_w = max(1, width - pad_l - pad_r)
            legend_cols = 3 if plot_w >= 1000 else 2 if plot_w >= 580 else 1
            legend_rows = max(1, (len(present) + legend_cols - 1) // legend_cols)
            legend_row_height = 20
            legend_first_y = 13
            hint_y = legend_first_y + legend_rows * legend_row_height + 2
            pad_t = hint_y + 20
            plot_h = max(1, height - pad_t - pad_b)

            def draw_x(source_x: float) -> float:
                return pad_l + plot_w * ((source_x - min_x) / span_x if span_x > 0 else 0.5)

            def draw_y(value: float) -> float:
                return pad_t + plot_h * (1.0 - value / max_y)

            for tick in range(5):
                fraction = tick / 4.0
                y = pad_t + plot_h * fraction
                canvas.create_line(pad_l, y, width - pad_r, y,
                                   fill=blend_hex_colors(self.panel, self.fg, 0.12))
                canvas.create_text(7, y, text=f"{max_y * (1.0 - fraction):.0f}",
                                   fill=self.muted, font=("Segoe UI", 7), anchor=tk.W)
            canvas.create_text(7, 10, text="ms", fill=self.muted, font=("Segoe UI", 8), anchor=tk.W)

            # X labels scale with available pixels; short sessions show real
            # measured IDs, longer sessions show a sparse numeric ruler.
            if time_axis:
                marks = time_ticks(min_x, max_x, max_labels=(6 if plot_w >= 700 else 4))
            elif use_even_spacing:
                # Each tick is attached to an actually measured sample, and
                # prints its original frame ID instead of the display index.
                marks = source_id_ticks(xs, min_x, max_x, plot_w)
            else:
                marks = (axis_ticks((min_x, max_x), plot_w, timeline["axis_label"])
                         if self._latency_x_view is not None else axis_ticks(xs, plot_w, timeline["axis_label"]))
            for index, (tick_value, label) in enumerate(marks):
                x = draw_x(tick_value)
                canvas.create_line(x, pad_t, x, pad_t + plot_h,
                                   fill=blend_hex_colors(self.panel, self.fg, 0.075))
                canvas.create_line(x, pad_t + plot_h, x, pad_t + plot_h + 5,
                                   fill=blend_hex_colors(self.panel, self.fg, 0.30), width=1)
                align = (tk.W if index == 0 else tk.E if index == len(marks) - 1 else tk.CENTER)
                canvas.create_text(x, pad_t + plot_h + 17, text=label,
                                   fill=self.muted, font=("Segoe UI", 8), anchor=align)
            if time_axis:
                axis_caption = ('Oś X: czas od startu sesji · chwila zapisu pomiaru, '
                                'nie moment ekspozycji kamery'
                                + (' · linie: sąsiednie ID klatek' if connect_consecutive else ''))
            elif use_even_spacing:
                axis_caption = ("Oś X: ID klatki z pomiarem czasu · punkty w równych odstępach "
                                "(nie jest to oś czasu)")
            elif timeline["axis_label"] == "ID klatki":
                axis_caption = "Oś X: ID klatki z pomiarem czasu · odstępy według numerów klatek"
            else:
                axis_caption = f"Oś X: {timeline['axis_label']}"
            footer_notes = [axis_caption]

            # No sampling at 100 frames. For >350 real trace rows, plot every
            # screen-column's measured min/max instead of silently dropping
            # spikes with stride sampling. The source values remain unmodified.
            left_hidden, visible_count, right_hidden = viewport_counts(plot_xs, min_x, max_x)
            dense = visible_count > 350
            picked_points = []
            for series_index, (key, style, values) in enumerate(present):
                measured = sum(value is not None for value in values)
                measured_visible = sum(value is not None and min_x <= x <= max_x
                                       for x, value in zip(plot_xs, values))
                # Each series has an exclusive column and row. The layout
                # changes with pane width; neither data nor zoom state does.
                legend_col = series_index % legend_cols
                legend_row = series_index // legend_cols
                legend_column_width = plot_w / legend_cols
                legend_x = pad_l + legend_col * legend_column_width
                legend_y = legend_first_y + legend_row * legend_row_height
                # The legend reflects the same line-only toggle as the plot.
                if connect_consecutive and not dense:
                    canvas.create_line(legend_x, legend_y, legend_x + 11, legend_y,
                                       fill=style.color, width=style.line_width)
                draw_chart_marker(canvas, legend_x + 5.5, legend_y, style, radius=4.0,
                                  background=self.panel)
                legend_name = (STAGE_DESCRIPTIONS if legend_column_width >= 280 else STAGE_SHORT).get(key, style.label)
                canvas.create_text(legend_x + 15, legend_y, text=(f"{legend_name} ({measured_visible}/{measured})"
                                         if measured_visible != measured else f"{legend_name} (n={measured})"),
                                   fill=self.fg, font=("Segoe UI", 8), anchor=tk.W)
                if dense:
                    # Dense view: a vertical min-max range is NOT one sample.
                    if self._latency_x_view is None:
                        bins = pixel_column_extrema(plot_xs, values, min_x, max_x, plot_w)
                    else:
                        sub_x, sub_v = visible_xy(plot_xs, values, min_x, max_x)
                        bins = pixel_column_extrema(sub_x, sub_v, min_x, max_x, plot_w)
                    for col, low, high, count in bins:
                        px = pad_l + col
                        if low < high:
                            canvas.create_line(px, draw_y(low), px, draw_y(high),
                                               fill=style.color, width=max(1.5, style.line_width * 0.75))
                        else:
                            draw_chart_marker(canvas, px, draw_y(low), style,
                                              radius=2.1, background=self.panel)
                else:
                    if connect_consecutive:
                        for i, j in continuous_segments(values, xs, require_consecutive=True):
                            if not (min_x <= plot_xs[i] <= max_x and min_x <= plot_xs[j] <= max_x):
                                continue
                            canvas.create_line(draw_x(plot_xs[i]), draw_y(values[i]),
                                               draw_x(plot_xs[j]), draw_y(values[j]),
                                               fill=style.color, width=style.line_width, smooth=False)
                    for i, value in enumerate(values):
                        if value is None or not min_x <= plot_xs[i] <= max_x:
                            continue
                        point_x, point_y = draw_x(plot_xs[i]), draw_y(value)
                        draw_chart_marker(canvas, point_x, point_y, style,
                                          radius=3.3, background=self.panel)
                        picked_points.append((point_x, point_y, i, key, value))
            self._latency_hit_points = picked_points
            link = self._evidence_timing_link() if getattr(self, '_linked_evidence_timing', None) else None
            if link:
                for px, py, i, _, _ in picked_points:
                    if str(bundle.trace_rows[i].get('frame_id', '')) in link['ids']:
                        canvas.create_oval(px-6, py-6, px+6, py+6, outline=self.fg, width=2,
                                           tags=('linked_evidence_timing',))
            if dense and plot_w >= 450:
                footer_notes.append("Zagęszczony widok: zakres min–max na kolumnę")
            # Guidance is below all legend rows, never behind MZ label.
            hint = ("Rolka: oś X  ·  przeciągnij: przesuń  ·  dwuklik: reset"
                    if plot_w >= 470 else "Rolka: zoom  ·  przesuń myszą  ·  dwuklik: reset")
            canvas.create_text(width - pad_r, hint_y, text=hint,
                               fill=self.muted, font=("Segoe UI", 8), anchor=tk.E)
            if self._latency_x_view is not None and plot_w > 350:
                # Explain which *measured* points the zoom hides; do not show a
                # misleading numeric frame span for a categorical axis.
                if use_even_spacing or time_axis:
                    zoom_note = f"Widoczne pomiary: {visible_count}/{len(xs)} · poza: ◀{left_hidden}  ▶{right_hidden}"
                else:
                    zoom_note = (f"ID {min_x:.1f}–{max_x:.1f} · {visible_count}/{len(xs)} pomiarów"
                                 f" · poza: ◀{left_hidden}  ▶{right_hidden}")
                footer_notes.append(zoom_note)
            if footer is not None:
                footer.set("   |   ".join(footer_notes))
        except Exception:
            logger.exception("Nie udało się wyświetlić śladu czasów ALPR")
            try:
                canvas.delete("all")
                canvas.create_text(canvas.winfo_width() / 2, canvas.winfo_height() / 2,
                                   text="Nie można wyświetlić wykresu — sprawdź szczegóły diagnostyczne.",
                                   fill=self.muted)
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
