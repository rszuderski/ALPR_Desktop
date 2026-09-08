from __future__ import annotations

import csv
import datetime
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from ..validators import read_model_metadata_sidecar
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors


def _format_model_value(value, *, percent: bool = False) -> str:
    if value is None or value == "":
        return "brak danych"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if percent:
        if 0.0 <= number <= 1.0:
            number *= 100.0
        return f"{number:.2f}%"
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _first_metric(info: dict, *keys: str):
    for key in keys:
        if key in info and info.get(key) not in (None, ""):
            return info.get(key)
    return None


def _float_or_none(value):
    try:
        return float(value)
    except Exception:
        return None


def _format_file_timestamp(value) -> str:
    try:
        return datetime.datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def _model_file_timestamp_rows(model_path: Path) -> list[tuple[str, str]]:
    try:
        stat = model_path.stat()
    except Exception:
        return []

    rows: list[tuple[str, str]] = []
    created_raw = getattr(stat, "st_birthtime", None)
    if created_raw is None:
        created_raw = getattr(stat, "st_ctime", None)
    created = _format_file_timestamp(created_raw)
    modified = _format_file_timestamp(getattr(stat, "st_mtime", None))
    if created:
        rows.append(("Utworzono plik", created))
    if modified:
        rows.append(("Modyfikacja pliku", modified))
    return rows


def _read_results_csv_metrics(model_path: Path) -> dict:
    candidates = []
    try:
        candidates.append(model_path.parent.parent / "results.csv")
        candidates.append(model_path.parent / "results.csv")
    except Exception:
        pass
    for csv_path in candidates:
        if not csv_path.exists():
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
            score = _float_or_none(row.get("metrics/mAP50-95(B)") or row.get("metrics/mAP50-95(P)") or row.get("metrics/mAP50-95"))
            if score is None:
                score = _float_or_none(row.get("metrics/mAP50(B)") or row.get("metrics/mAP50(P)") or row.get("metrics/mAP50"))
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_row = row
        best_row = best_row or rows[-1]
        return {
            "map50": _float_or_none(best_row.get("metrics/mAP50(B)") or best_row.get("metrics/mAP50(P)") or best_row.get("metrics/mAP50")),
            "map50_95": _float_or_none(best_row.get("metrics/mAP50-95(B)") or best_row.get("metrics/mAP50-95(P)") or best_row.get("metrics/mAP50-95")),
            "precision": _float_or_none(best_row.get("metrics/precision(B)") or best_row.get("metrics/precision(P)") or best_row.get("metrics/precision")),
            "recall": _float_or_none(best_row.get("metrics/recall(B)") or best_row.get("metrics/recall(P)") or best_row.get("metrics/recall")),
            "epoch": _float_or_none(best_row.get("epoch")),
            "metrics_source": str(csv_path),
        }
    return {}


def _build_model_rows(host, model_path: str) -> tuple[list[tuple[str, str]], str]:
    safe_path = Path(str(model_path or "").strip())
    rows: list[tuple[str, str]] = [("Plik", safe_path.name or "brak")]
    try:
        rows.append(("Ścieżka", str(safe_path.resolve())))
    except Exception:
        rows.append(("Ścieżka", str(safe_path)))
    rows.extend(_model_file_timestamp_rows(safe_path))

    try:
        version, size = host._infer_yolo_arch_from_model_path(str(safe_path))
    except Exception:
        version, size = "", ""
    if version or size:
        rows.append(("Architektura z nazwy", f"YOLOv{version}{size}".strip()))

    metadata = read_model_metadata_sidecar(safe_path) if safe_path else None
    if metadata is None:
        rows.append(("Metadata JSON", "nie znaleziono pasującego sidecara"))
        return rows, "Podgląd korzysta wyłącznie z lekkiego pliku JSON obok modelu. Plik .pt nie jest ładowany."

    validation_ok, validation_message, info = metadata
    info = dict(info or {})
    csv_metrics = _read_results_csv_metrics(safe_path)
    for key, value in csv_metrics.items():
        if value not in (None, "") and info.get(key) in (None, ""):
            info[key] = value
    rows.append(("Status metadata", "OK" if validation_ok else "uwaga"))
    if validation_message:
        rows.append(("Walidacja", str(validation_message)))

    metadata_json = str(info.get("metadata_json") or "").strip()
    if metadata_json:
        rows.append(("Plik JSON", metadata_json))
    metrics_source = str(info.get("metrics_source") or "").strip()
    if metrics_source:
        rows.append(("Metryki", metrics_source))

    arch = (
        str(info.get("architecture_label") or "").strip()
        or str(info.get("yolo_variant") or "").strip()
        or str(info.get("source_architecture_label") or "").strip()
    )
    if arch:
        rows.append(("Architektura", arch))

    task = str(info.get("task") or info.get("type") or "").strip()
    if task:
        rows.append(("Zadanie", task))

    rows.extend(
        [
            (
                "mAP50",
                _format_model_value(
                    _first_metric(
                        info,
                        "map50",
                        "best_map50",
                        "box_map50",
                        "pose_map50",
                        "metrics/mAP50(B)",
                        "metrics/mAP50(P)",
                        "metrics/mAP50",
                    ),
                    percent=True,
                ),
            ),
            (
                "mAP50-95",
                _format_model_value(
                    _first_metric(
                        info,
                        "map50_95",
                        "best_map50_95",
                        "box_map50_95",
                        "pose_map50_95",
                        "metrics/mAP50-95(B)",
                        "metrics/mAP50-95(P)",
                        "metrics/mAP50-95",
                    ),
                    percent=True,
                ),
            ),
            (
                "Precision",
                _format_model_value(
                    _first_metric(
                        info,
                        "precision",
                        "box_precision",
                        "pose_precision",
                        "metrics/precision(B)",
                        "metrics/precision(P)",
                    ),
                    percent=True,
                ),
            ),
            (
                "Recall",
                _format_model_value(
                    _first_metric(
                        info,
                        "recall",
                        "box_recall",
                        "pose_recall",
                        "metrics/recall(B)",
                        "metrics/recall(P)",
                    ),
                    percent=True,
                ),
            ),
            ("Epoka", _format_model_value(_first_metric(info, "epoch", "best_epoch", "Epoch"))),
        ]
    )

    classes = info.get("classes") or []
    if isinstance(classes, (list, tuple)):
        class_preview = ", ".join(str(item) for item in classes[:12])
        if len(classes) > 12:
            class_preview += f" ... (+{len(classes) - 12})"
        rows.append(("Klasy", f"{len(classes)} | {class_preview}" if class_preview else str(len(classes))))
    elif info.get("num_classes") not in (None, ""):
        rows.append(("Liczba klas", _format_model_value(info.get("num_classes"))))

    source_model = str(info.get("source_model_name") or info.get("source_model") or "").strip()
    if source_model:
        rows.append(("Model źródłowy", source_model))

    dataset_path = str(info.get("dataset_path") or info.get("training_dataset") or "").strip()
    if dataset_path:
        rows.append(("Dataset treningowy", dataset_path))

    return rows, "Podgląd został odczytany z lekkich metadanych JSON/CSV, bez ładowania ciężkiego modelu .pt."


def show_yolo_model_metadata_dialog(host, model_path: str | None = None, *, title: str | None = None) -> None:
    resolved_path = str(model_path or host._get_effective_yolo_model_path() or "").strip()
    parent = getattr(host, "_detection_pipeline_modal", None) or getattr(host, "frame", None)
    if not resolved_path:
        messagebox.showinfo(
            "Parametry modelu detekcji",
            "Najpierw wskaż model detekcji znaków .pt. Parametry zostaną odczytane z pliku JSON obok modelu.",
            parent=parent,
        )
        return

    rows, footer = _build_model_rows(host, resolved_path)
    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    accent = palette.get("accent", "#4ade80")

    dialog = tk.Toplevel(parent)
    dialog_title = title or "Parametry modelu detekcji"
    dialog.title(dialog_title)
    try:
        styler = getattr(host.app, "style_dialog_window", None)
        if callable(styler):
            styler(dialog, title=dialog_title, geometry="760x520", parent=parent)
        else:
            dialog.geometry("760x520")
            dialog.transient(parent)
            dialog.grab_set()
    except Exception:
        try:
            dialog.geometry("760x520")
        except Exception:
            pass
    dialog.configure(bg=panel_bg)

    body_builder = getattr(host.app, "_build_themed_dialog_surface", None)
    if callable(body_builder):
        body = body_builder(dialog, tone="info")
    else:
        body = tk.Frame(dialog, bg=panel_bg, padx=18, pady=16)
        body.pack(fill=tk.BOTH, expand=True)

    body.grid_columnconfigure(0, weight=1)
    body.grid_rowconfigure(1, weight=1)
    tk.Label(
        body,
        text=dialog_title,
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 13, "bold"),
        anchor="w",
    ).grid(row=0, column=0, sticky="ew", pady=(0, 10))

    table_shell = tk.Frame(body, bg=panel_bg, bd=0, highlightthickness=0)
    table_shell.grid(row=1, column=0, sticky="nsew")
    table_shell.grid_rowconfigure(0, weight=1)
    table_shell.grid_columnconfigure(0, weight=1)

    table_canvas = tk.Canvas(
        table_shell,
        bg=panel_alt,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
    )
    table_scroll = WebSlimScrollbar(table_shell, orient=tk.VERTICAL, command=table_canvas.yview)
    table_canvas.configure(yscrollcommand=table_scroll.set)
    table_canvas.grid(row=0, column=0, sticky="nsew")
    table_scroll.grid(row=0, column=1, sticky="ns")

    table = tk.Frame(table_canvas, bg=panel_alt)
    table_window = table_canvas.create_window((0, 0), window=table, anchor="nw")
    table.grid_columnconfigure(0, weight=0)
    table.grid_columnconfigure(1, weight=1)

    def _sync_table_scrollregion(_event=None):
        try:
            table_canvas.configure(scrollregion=table_canvas.bbox("all"))
        except Exception:
            pass

    def _sync_table_width(event=None):
        try:
            table_canvas.itemconfigure(table_window, width=max(1, int(event.width)))
        except Exception:
            pass

    def _on_table_mousewheel(event):
        try:
            delta = -1 if int(getattr(event, "delta", 0) or 0) > 0 else 1
            table_canvas.yview_scroll(delta * 3, "units")
            return "break"
        except Exception:
            return None

    table.bind("<Configure>", _sync_table_scrollregion, add="+")
    table_canvas.bind("<Configure>", _sync_table_width, add="+")
    table_canvas.bind("<MouseWheel>", _on_table_mousewheel, add="+")
    table.bind("<MouseWheel>", _on_table_mousewheel, add="+")

    for idx, (label, value) in enumerate(rows):
        bg = panel_alt if idx % 2 == 0 else blend_hex_colors(panel_alt, panel_bg, 0.35)
        tk.Label(
            table,
            text=label,
            bg=bg,
            fg=accent,
            font=("Segoe UI", 9),
            anchor="w",
            padx=10,
            pady=6,
        ).grid(row=idx, column=0, sticky="nsew")
        tk.Label(
            table,
            text=value,
            bg=bg,
            fg=fg,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=520,
            padx=10,
            pady=6,
        ).grid(row=idx, column=1, sticky="nsew")

    footer_row = tk.Frame(body, bg=panel_bg)
    footer_row.grid(row=2, column=0, sticky="ew", pady=(12, 0))
    footer_row.grid_columnconfigure(0, weight=1)
    tk.Label(
        footer_row,
        text=footer,
        bg=panel_bg,
        fg=muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=560,
    ).grid(row=0, column=0, sticky="ew", padx=(0, 10))
    ttk.Button(footer_row, text="Zamknij", command=dialog.destroy).grid(row=0, column=1, sticky="e")
