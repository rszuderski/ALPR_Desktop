#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Porównanie przebiegów treningów w Z4/PZ2."""

from __future__ import annotations

import csv
import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from ..campaign_manager import CAMPAIGN
from ..config import logger
from ..training import TrainingStatus
from .dataset_display import build_dataset_display_ref
from .run_display import build_run_display_ref
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors


COMPARE_METRICS: tuple[dict[str, object], ...] = (
    {
        "key": "map50_95",
        "label": "mAP50-95",
        "hint": "Najsurowsza metryka jakości. Dla Pose preferujemy punkty/narożniki, a dla Detect klasyczne boxy.",
        "kind": "score",
        "keys": (
            "pose_map50_95",
            "metrics/mAP50-95(P)",
            "map50_95",
            "metrics/mAP50-95",
            "box_map50_95",
            "metrics/mAP50-95(B)",
        ),
    },
    {
        "key": "map50",
        "label": "mAP50",
        "hint": "Łagodniejsza metryka jakości. Dobrze pokazuje, czy model w ogóle łapie obiekty.",
        "kind": "score",
        "keys": (
            "pose_map50",
            "metrics/mAP50(P)",
            "map50",
            "metrics/mAP50",
            "box_map50",
            "metrics/mAP50(B)",
        ),
    },
    {
        "key": "precision",
        "label": "Precision",
        "hint": "Czystość predykcji: ile wskazań modelu było trafnych.",
        "kind": "score",
        "keys": (
            "pose_precision",
            "metrics/precision(P)",
            "precision",
            "metrics/precision",
            "box_precision",
            "metrics/precision(B)",
        ),
    },
    {
        "key": "recall",
        "label": "Recall",
        "hint": "Czułość: ile właściwych obiektów model odnalazł.",
        "kind": "score",
        "keys": (
            "pose_recall",
            "metrics/recall(P)",
            "recall",
            "metrics/recall",
            "box_recall",
            "metrics/recall(B)",
        ),
    },
    {
        "key": "train_loss",
        "label": "Train loss",
        "hint": "Błąd na materiale treningowym. Powinien raczej maleć, ale sam nie wystarcza do wyboru modelu.",
        "kind": "loss",
        "loss_prefix": "train",
    },
    {
        "key": "val_loss",
        "label": "Val loss",
        "hint": "Błąd na walidacji. Rozjazd względem train loss może sugerować przeuczenie.",
        "kind": "loss",
        "loss_prefix": "val",
    },
)


COMPARE_COLORS = (
    "#2f80ed",
    "#27ae60",
    "#f2994a",
    "#eb5757",
    "#9b51e0",
    "#00a6a6",
    "#b7791f",
    "#d946ef",
)


def _training_compare_dialog_geometry(widget) -> tuple[int, int, int, int, bool]:
    """Return a screen-safe initial geometry for the comparison modal."""
    try:
        screen_w = int(widget.winfo_screenwidth() or 1360)
        screen_h = int(widget.winfo_screenheight() or 860)
    except Exception:
        screen_w, screen_h = 1360, 860

    width = min(1360, max(940, screen_w - 96))
    height = min(860, max(600, screen_h - 128))
    width = min(width, max(720, screen_w - 48))
    height = min(height, max(520, screen_h - 72))
    x = max(0, int((screen_w - width) / 2))
    y = max(0, int((screen_h - height) / 2) - 8)
    compact = bool(width < 1180 or height < 760)
    return int(width), int(height), int(x), int(y), compact


def _apply_training_compare_dialog_geometry(dialog) -> bool:
    try:
        width, height, x, y, compact = _training_compare_dialog_geometry(dialog)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        dialog.minsize(min(980, width), min(560, height))
        try:
            screen_w = int(dialog.winfo_screenwidth() or width)
            screen_h = int(dialog.winfo_screenheight() or height)
            dialog.maxsize(max(width, screen_w), max(height, screen_h))
        except Exception:
            pass
        return compact
    except Exception:
        try:
            dialog.geometry("1180x760")
            dialog.minsize(940, 560)
        except Exception:
            pass
        return False


def _toggle_training_compare_maximized(self):
    dialog = getattr(self, "_training_compare_dialog", None)
    if dialog is None:
        return
    try:
        if str(dialog.state()) == "zoomed":
            dialog.state("normal")
        else:
            dialog.state("zoomed")
    except Exception:
        try:
            is_zoomed = bool(dialog.attributes("-zoomed"))
            dialog.attributes("-zoomed", not is_zoomed)
        except Exception:
            return
    try:
        dialog.after_idle(self._draw_training_compare_chart)
    except Exception:
        pass


def _training_compare_metric_by_key(key: str) -> dict[str, object]:
    raw = str(key or "").strip()
    for metric in COMPARE_METRICS:
        if metric["key"] == raw or metric["label"] == raw:
            return metric
    return COMPARE_METRICS[0]


def _format_compare_datetime(value: str | None, *, long: bool = False) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "-"
    try:
        fmt = "%Y-%m-%d %H:%M:%S" if long else "%d.%m %H:%M"
        return datetime.datetime.fromisoformat(raw).strftime(fmt)
    except Exception:
        return raw.replace("T", " ")[:19 if long else 16] or "-"


def _compare_dataset_label(self, run) -> str:
    dataset_path = str(getattr(run, "dataset_path", "") or "").strip()
    if not dataset_path:
        return "-"
    try:
        target_hint = self._infer_history_run_target(run)
    except Exception:
        target_hint = ""
    try:
        return build_dataset_display_ref(dataset_path, target_hint=target_hint).id
    except Exception:
        return Path(dataset_path).name or "-"


def _compare_float(value) -> float | None:
    raw = str(value if value is not None else "").strip()
    if not raw:
        return None
    try:
        numeric = float(raw.replace(",", "."))
    except Exception:
        return None
    if numeric != numeric:
        return None
    return numeric


def _compare_row_value(row: dict, keys: tuple[str, ...]) -> float | None:
    if not isinstance(row, dict):
        return None
    normalized = {str(k or "").strip(): v for k, v in row.items()}
    lower_map = {str(k or "").strip().lower(): v for k, v in row.items()}
    for key in keys:
        if key in normalized:
            value = _compare_float(normalized.get(key))
            if value is not None:
                return value
        low = key.lower()
        if low in lower_map:
            value = _compare_float(lower_map.get(low))
            if value is not None:
                return value
    return None


def _compare_loss_value(row: dict, prefix: str) -> float | None:
    keys = (
        f"{prefix}/box_loss",
        f"{prefix}/pose_loss",
        f"{prefix}/kobj_loss",
        f"{prefix}/cls_loss",
        f"{prefix}/dfl_loss",
    )
    values = []
    for key in keys:
        value = _compare_row_value(row, (key,))
        if value is not None:
            values.append(value)
    if values:
        return sum(values)
    fallback_keys = (f"{prefix}/loss", f"{prefix}_loss")
    if prefix == "train":
        fallback_keys = (*fallback_keys, "loss")
    return _compare_row_value(row, fallback_keys)


def _compare_epoch_value(row: dict, fallback: int) -> int:
    value = _compare_row_value(row, ("epoch", "Epoch", "epoka"))
    if value is None:
        return fallback
    try:
        return max(1, int(round(value)))
    except Exception:
        return fallback


def _read_compare_results_csv(run) -> list[dict[str, str]]:
    run_dir = Path(str(getattr(run, "output_dir", "") or ""))
    csv_path = run_dir / "train" / "results.csv"
    if not csv_path.exists():
        return []
    rows: list[dict[str, str]] = []
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if isinstance(row, dict):
                    rows.append({str(k or "").strip(): str(v or "").strip() for k, v in row.items()})
    except Exception as exc:
        logger.debug(f"Nie udało się odczytać results.csv dla porównania runów: {exc}")
        return []
    return rows


def _build_compare_curve_from_rows(rows: list[dict], metric: dict[str, object]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    metric_key = str(metric.get("key") or "")
    keys = tuple(str(k) for k in metric.get("keys", ()) or ())
    loss_prefix = str(metric.get("loss_prefix") or "")
    for index, row in enumerate(rows, start=1):
        epoch = _compare_epoch_value(row, index)
        if metric_key in {"train_loss", "val_loss"}:
            value = _compare_loss_value(row, loss_prefix)
        else:
            value = _compare_row_value(row, keys)
        if value is not None:
            points.append((float(epoch), float(value)))
    return points


def _collect_compare_curves_for_run(run) -> dict[str, list[tuple[float, float]]]:
    csv_rows = _read_compare_results_csv(run)
    history_rows = list(getattr(run, "metrics_history", []) or [])
    source_rows = csv_rows or history_rows
    curves: dict[str, list[tuple[float, float]]] = {}
    for metric in COMPARE_METRICS:
        curves[str(metric["key"])] = _build_compare_curve_from_rows(source_rows, metric)
    return curves


def _format_compare_run_label(self, run, index: int) -> str:
    run_id = str(getattr(run, "id", "") or "").strip()
    dataset = _compare_dataset_label(self, run)
    try:
        base = build_run_display_ref(run, kind_hint="training").id
    except Exception:
        base = run_id or f"run {index + 1}"
    suffix = dataset
    if suffix:
        return f"{index + 1}. {base} | {suffix}"
    return f"{index + 1}. {base}"


def _training_compare_run_key(run) -> str:
    run_id = str(getattr(run, "id", "") or "").strip()
    if run_id:
        return run_id.lower()
    for attr in ("output_dir", "run_dir", "project_dir"):
        raw = str(getattr(run, attr, "") or "").strip()
        if not raw:
            continue
        try:
            return str(Path(raw).resolve()).lower()
        except Exception:
            return str(Path(raw)).lower()
    return ""


def _selected_training_compare_runs(self) -> list:
    tree = getattr(self, "tree", None)
    if tree is None:
        return []
    try:
        selected_ids = list(tree.selection() or [])
    except Exception:
        selected_ids = []
    if not selected_ids:
        try:
            focused = str(tree.focus() or "").strip()
        except Exception:
            focused = ""
        if focused:
            selected_ids = [focused]
    runs = []
    try:
        self._reload_history_snapshot_from_disk()
    except Exception:
        pass
    seen: set[str] = set()
    for run_id in selected_ids:
        try:
            run = self.history.get_run(str(run_id))
        except Exception:
            run = None
        if run is not None:
            key = _training_compare_run_key(run)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            runs.append(run)
    return runs


def _draw_training_compare_chart(self):
    canvas = getattr(self, "_training_compare_canvas", None)
    if canvas is None:
        return
    try:
        width = int(canvas.winfo_width() or 0)
        height = int(canvas.winfo_height() or 0)
    except Exception:
        return
    if width < 200 or height < 160:
        try:
            canvas.after(80, self._draw_training_compare_chart)
        except Exception:
            pass
        return

    canvas.delete("all")
    palette = getattr(getattr(self, "app", None), "palette", {}) or {}
    bg = palette.get("panel", "#252526")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#9ca3af")
    grid_color = blend_hex_colors(muted, bg, 0.74)
    axis_color = blend_hex_colors(muted, bg, 0.38)

    try:
        canvas.configure(bg=bg)
    except Exception:
        pass

    runs_data = list(getattr(self, "_training_compare_runs_data", []) or [])
    if not runs_data:
        canvas.create_text(width / 2, height / 2, text="Brak runów do porównania.", fill=muted, font=("Segoe UI", 11))
        return

    metric_choice = str(getattr(getattr(self, "_training_compare_metric_var", None), "get", lambda: "mAP50-95")() or "mAP50-95")
    metric_key = str(_training_compare_metric_by_key(metric_choice).get("key") or "map50_95")
    metric = _training_compare_metric_by_key(metric_key)
    x_mode = str(getattr(getattr(self, "_training_compare_xmode_var", None), "get", lambda: "Epoka")() or "Epoka")
    smooth = bool(getattr(getattr(self, "_training_compare_smooth_var", None), "get", lambda: False)())

    left = 74
    right = 26
    top = 34
    bottom = 54
    plot_w = max(1, width - left - right)
    plot_h = max(1, height - top - bottom)

    visible_series = []
    for item in runs_data:
        run_id = str(item.get("run_id") or "")
        var = getattr(self, "_training_compare_visible_vars", {}).get(run_id)
        if var is not None and not bool(var.get()):
            continue
        points = list((item.get("curves") or {}).get(metric_key) or [])
        if not points:
            continue
        total_epochs = max(1, int(item.get("epochs") or 0) or int(max((p[0] for p in points), default=1)))
        if x_mode == "Postęp %":
            display_points = [(min(100.0, max(0.0, (x / float(total_epochs)) * 100.0)), y) for x, y in points]
        else:
            display_points = points
        visible_series.append((item, display_points))

    if not visible_series:
        canvas.create_text(width / 2, height / 2, text="Wybrane runy nie mają tej krzywej metryk.", fill=muted, font=("Segoe UI", 11))
        return

    all_x = [x for _item, points in visible_series for x, _y in points]
    all_y = [y for _item, points in visible_series for _x, y in points]
    x_min = 0.0 if x_mode == "Postęp %" else min(all_x)
    x_max = 100.0 if x_mode == "Postęp %" else max(all_x)
    if x_max <= x_min:
        x_max = x_min + 1.0
    y_min = min(all_y)
    y_max = max(all_y)
    if str(metric.get("kind")) == "score":
        y_min = 0.0
        y_max = max(1.0, y_max)
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_pad = (y_max - y_min) * 0.08
    if str(metric.get("kind")) != "score":
        y_min = max(0.0, y_min - y_pad)
        y_max = y_max + y_pad

    def sx(x: float) -> float:
        return left + ((x - x_min) / (x_max - x_min)) * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - ((y - y_min) / (y_max - y_min))) * plot_h

    # Siatka i osie.
    for i in range(6):
        ratio = i / 5.0
        y = top + ratio * plot_h
        value = y_max - ratio * (y_max - y_min)
        canvas.create_line(left, y, width - right, y, fill=grid_color)
        canvas.create_text(left - 10, y, text=f"{value:.2f}", fill=muted, anchor=tk.E, font=("Segoe UI", 8))
    for i in range(6):
        ratio = i / 5.0
        x = left + ratio * plot_w
        value = x_min + ratio * (x_max - x_min)
        label = f"{value:.0f}%" if x_mode == "Postęp %" else f"{value:.0f}"
        canvas.create_line(x, top, x, height - bottom, fill=grid_color)
        canvas.create_text(x, height - bottom + 18, text=label, fill=muted, anchor=tk.N, font=("Segoe UI", 8))
    canvas.create_line(left, top, left, height - bottom, fill=axis_color, width=2)
    canvas.create_line(left, height - bottom, width - right, height - bottom, fill=axis_color, width=2)
    canvas.create_text(left, 12, text=str(metric["label"]), fill=fg, anchor=tk.W, font=("Segoe UI Semibold", 10))
    canvas.create_text(width - right, height - 14, text=x_mode, fill=muted, anchor=tk.E, font=("Segoe UI", 8))

    hover_points = []
    for item, points in visible_series:
        color = str(item.get("color") or "#2f80ed")
        pixel_points = []
        for x, y in points:
            px = sx(x)
            py = sy(y)
            pixel_points.extend((px, py))
            hover_points.append((px, py, item, x, y))
        if len(pixel_points) >= 4:
            canvas.create_line(
                *pixel_points,
                fill=color,
                width=2,
                smooth=bool(smooth),
                splinesteps=18 if smooth else 1,
            )
        for px, py in zip(pixel_points[0::2], pixel_points[1::2]):
            canvas.create_oval(px - 2, py - 2, px + 2, py + 2, fill=color, outline=color)

        raw_points = list((item.get("curves") or {}).get(metric_key) or [])
        if raw_points:
            best_epoch, best_value = max(raw_points, key=lambda p: p[1])
            x_best = (best_epoch / float(max(1, int(item.get("epochs") or best_epoch))) * 100.0) if x_mode == "Postęp %" else best_epoch
            px = sx(x_best)
            py = sy(best_value)
            canvas.create_oval(px - 5, py - 5, px + 5, py + 5, outline=color, width=2)
            canvas.create_line(px, top, px, height - bottom, fill=blend_hex_colors(color, bg, 0.55), dash=(3, 4))

    self._training_compare_hover_points = hover_points
    hint_label = getattr(self, "_training_compare_chart_hint_lbl", None)
    if hint_label is not None:
        try:
            hint_label.configure(text=str(metric.get("hint") or ""))
        except Exception:
            pass


def _on_training_compare_canvas_motion(self, event):
    points = list(getattr(self, "_training_compare_hover_points", []) or [])
    label = getattr(self, "_training_compare_hover_lbl", None)
    if label is None or not points:
        return
    ex = float(getattr(event, "x", 0) or 0)
    ey = float(getattr(event, "y", 0) or 0)
    best = min(points, key=lambda p: (p[0] - ex) ** 2 + (p[1] - ey) ** 2)
    distance2 = (best[0] - ex) ** 2 + (best[1] - ey) ** 2
    if distance2 > 18 ** 2:
        try:
            label.configure(text="Najedź na punkt krzywej, aby zobaczyć run, epokę i wartość.")
        except Exception:
            pass
        return
    _px, _py, item, x, y = best
    try:
        label.configure(text=f"{item.get('short_label')}: {x:.0f} | {y:.4f}")
    except Exception:
        pass


def _refresh_training_compare_legend(self):
    host = getattr(self, "_training_compare_legend_frame", None)
    if host is None:
        return
    try:
        for child in host.winfo_children():
            child.destroy()
    except Exception:
        pass
    palette = getattr(getattr(self, "app", None), "palette", {}) or {}
    panel = palette.get("panel", "#252526")
    fg = palette.get("fg", "#f3f3f3")
    self._training_compare_visible_vars = {}
    for row, item in enumerate(list(getattr(self, "_training_compare_runs_data", []) or [])):
        run_id = str(item.get("run_id") or "")
        color = str(item.get("color") or "#2f80ed")
        var = tk.BooleanVar(value=True)
        self._training_compare_visible_vars[run_id] = var
        swatch = tk.Label(host, text="■", fg=color, bg=panel, font=("Segoe UI", 12), width=2)
        swatch.grid(row=row, column=0, sticky="w", padx=(2, 0), pady=1)
        chk = ttk.Checkbutton(
            host,
            text=str(item.get("short_label") or run_id),
            variable=var,
            command=self._draw_training_compare_chart,
        )
        chk.grid(row=row, column=1, sticky="ew", padx=(2, 6), pady=1)
        try:
            chk.configure(cursor="hand2")
        except Exception:
            pass
    host.grid_columnconfigure(1, weight=1)


def _build_training_compare_runs_data(self, runs: list) -> list[dict[str, object]]:
    data = []
    seen: set[str] = set()
    for run in list(runs or []):
        run_id = str(getattr(run, "id", "") or "").strip()
        if not run_id:
            continue
        key = _training_compare_run_key(run)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        index = len(data)
        curves = _collect_compare_curves_for_run(run)
        label = _format_compare_run_label(self, run, index)
        short_label = label if len(label) <= 44 else label[:41].rstrip() + "..."
        data.append(
            {
                "run": run,
                "run_id": run_id,
                "label": label,
                "short_label": short_label,
                "color": COMPARE_COLORS[index % len(COMPARE_COLORS)],
                "curves": curves,
                "epochs": int(getattr(run, "epochs", 0) or 0),
            }
        )
        if len(data) >= 8:
            break
    return data


def _populate_training_compare_summary(self):
    tree = getattr(self, "_training_compare_summary_tree", None)
    if tree is None:
        return
    try:
        tree.delete(*tree.get_children())
    except Exception:
        pass
    for item in list(getattr(self, "_training_compare_runs_data", []) or []):
        run = item.get("run")
        if run is None:
            continue
        run_id = str(item.get("run_id") or "")
        try:
            target = self._format_history_run_target_label(self._infer_history_run_target(run))
        except Exception:
            target = "-"
        dataset = _compare_dataset_label(self, run)
        started = _format_compare_datetime(getattr(run, "started_at", None) or getattr(run, "created_at", None))
        try:
            best95 = max((p[1] for p in (item.get("curves") or {}).get("map50_95", []) or []), default=float(getattr(run, "best_map50_95", 0.0) or 0.0))
        except Exception:
            best95 = float(getattr(run, "best_map50_95", 0.0) or 0.0)
        try:
            best50 = max((p[1] for p in (item.get("curves") or {}).get("map50", []) or []), default=float(getattr(run, "best_map50", 0.0) or 0.0))
        except Exception:
            best50 = float(getattr(run, "best_map50", 0.0) or 0.0)
        status = self._format_history_run_status_label(run)
        values = (
            "■",
            target,
            started,
            build_run_display_ref(run, kind_hint="training").id,
            dataset,
            f"{int(getattr(run, 'current_epoch', 0) or 0)}/{int(getattr(run, 'epochs', 0) or 0)}",
            f"{best95:.3f}",
            f"{best50:.3f}",
            status,
        )
        try:
            tree.insert("", tk.END, iid=run_id, values=values, tags=(f"run_{run_id}",))
            tree.tag_configure(f"run_{run_id}", foreground=str(item.get("color") or "#2f80ed"))
        except Exception:
            continue


def _selected_training_compare_modal_run(self):
    tree = getattr(self, "_training_compare_summary_tree", None)
    if tree is None:
        return None
    try:
        selection = list(tree.selection() or [])
    except Exception:
        selection = []
    run_id = str(selection[0] if selection else "").strip()
    if not run_id:
        return None
    for item in list(getattr(self, "_training_compare_runs_data", []) or []):
        if str(item.get("run_id") or "") == run_id:
            return item.get("run")
    return None


def _open_training_compare_selected_details(self):
    run = _selected_training_compare_modal_run(self)
    if run is None:
        return messagebox.showwarning("Brak runu", "Najpierw wskaż run w tabeli porównania.")
    return self._open_run_details_modal(run)


def _promote_training_compare_selected_run(self):
    run = _selected_training_compare_modal_run(self)
    if run is None:
        return messagebox.showwarning("Brak runu", "Najpierw wskaż run w tabeli porównania.")
    run_id = str(getattr(run, "id", "") or "").strip()
    tree = getattr(self, "tree", None)
    if tree is not None and run_id:
        try:
            tree.selection_set(run_id)
            tree.focus(run_id)
            tree.see(run_id)
        except Exception:
            pass
    return self._promote_selected_run_model_to_campaign()


def _open_training_compare_modal(self, runs: list):
    if len(runs) < 2:
        return messagebox.showinfo(
            "Za mało runów",
            "Zaznacz co najmniej dwa treningi w historii, aby porównać ich krzywe.",
        )

    self._training_compare_runs_data = _build_training_compare_runs_data(self, runs)
    if len(self._training_compare_runs_data) < 2:
        return messagebox.showinfo(
            "Brak danych",
            "Nie udało się przygotować co najmniej dwóch runów do porównania.",
        )

    palette = getattr(getattr(self, "app", None), "palette", {}) or {}
    dialog = getattr(self, "_training_compare_dialog", None)
    dialog_exists = False
    if dialog is not None:
        try:
            dialog_exists = bool(dialog.winfo_exists())
        except Exception:
            dialog_exists = False

    if not dialog_exists:
        dialog = tk.Toplevel(self.frame)
        dialog.title("Porównanie treningów")
        dialog.resizable(True, True)
        compact_dialog = _apply_training_compare_dialog_geometry(dialog)
        dialog.protocol("WM_DELETE_WINDOW", self._close_training_compare_dialog)
        dialog.bind("<F11>", lambda _event: _toggle_training_compare_maximized(self), add="+")
        self._training_compare_dialog = dialog

        shell = ttk.Frame(dialog, padding=9 if compact_dialog else 12, style="Panel.TFrame")
        shell.pack(fill=tk.BOTH, expand=True)
        self._training_compare_shell = shell

        header = ttk.Frame(shell, style="Panel.TFrame")
        header.pack(fill=tk.X, pady=(0, 10))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="Porównanie treningów",
            style="PanelTitle.TLabel",
            anchor=tk.W,
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(
            header,
            text="Krzywe są nakładane na jedną oś. Wybierz metrykę, tryb osi X i ukrywaj modele w legendzie.",
            style="PanelMuted.TLabel",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=760 if compact_dialog else 980,
        ).grid(row=1, column=0, sticky="ew", pady=(3, 0))
        ttk.Button(
            header,
            text="Minimalizuj",
            command=dialog.iconify,
        ).grid(row=0, column=1, rowspan=2, sticky="ne", padx=(10, 6))
        ttk.Button(
            header,
            text="Maksymalizuj",
            command=lambda: _toggle_training_compare_maximized(self),
        ).grid(row=0, column=2, rowspan=2, sticky="ne")

        summary_box = ttk.LabelFrame(shell, text=" Tabela decyzyjna ", padding=7)
        summary_box.pack(fill=tk.X, pady=(0, 10))
        columns = ("Kolor", "Tor", "Start", "Run", "Dataset", "Epoki", "Best mAP50-95", "Best mAP50", "Status")
        self._training_compare_summary_tree = ttk.Treeview(
            summary_box,
            columns=columns,
            show="headings",
            height=4 if compact_dialog else 5,
            selectmode="browse",
        )
        heading_labels = {
            "Best mAP50-95": "mAP50-95",
            "Best mAP50": "mAP50",
        }
        for name in columns:
            self._training_compare_summary_tree.heading(name, text=heading_labels.get(name, name))
        self._training_compare_summary_tree.column("Kolor", width=38, stretch=False, anchor=tk.CENTER)
        self._training_compare_summary_tree.column("Tor", width=64, stretch=False, anchor=tk.CENTER)
        self._training_compare_summary_tree.column("Start", width=86, stretch=False, anchor=tk.CENTER)
        self._training_compare_summary_tree.column("Run", width=210, stretch=True)
        self._training_compare_summary_tree.column("Dataset", width=150, stretch=True)
        self._training_compare_summary_tree.column("Epoki", width=54, stretch=False, anchor=tk.CENTER)
        self._training_compare_summary_tree.column("Best mAP50-95", width=82, stretch=False, anchor=tk.CENTER)
        self._training_compare_summary_tree.column("Best mAP50", width=72, stretch=False, anchor=tk.CENTER)
        self._training_compare_summary_tree.column("Status", width=98, stretch=False)
        summary_scroll = WebSlimScrollbar(summary_box, orient=tk.VERTICAL, command=self._training_compare_summary_tree.yview)
        self._training_compare_summary_tree.configure(yscrollcommand=summary_scroll.set)
        self._training_compare_summary_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        controls = ttk.Frame(shell, style="Panel.TFrame")
        controls.pack(fill=tk.X, pady=(0, 8))
        self._training_compare_metric_var = tk.StringVar(value="mAP50-95")
        metric_values = tuple(str(metric["label"]) for metric in COMPARE_METRICS)
        self._training_compare_metric_combo = ttk.Combobox(
            controls,
            textvariable=self._training_compare_metric_var,
            values=metric_values,
            state="readonly",
            width=18,
        )
        ttk.Label(controls, text="Krzywa:", style="PanelMuted.TLabel").pack(side=tk.LEFT, padx=(0, 5))
        self._training_compare_metric_combo.pack(side=tk.LEFT, padx=(0, 12))
        self._training_compare_metric_combo.bind("<<ComboboxSelected>>", lambda _event: self._draw_training_compare_chart())

        self._training_compare_xmode_var = tk.StringVar(value="Epoka")
        ttk.Label(controls, text="Oś X:", style="PanelMuted.TLabel").pack(side=tk.LEFT, padx=(0, 5))
        xmode_combo = ttk.Combobox(
            controls,
            textvariable=self._training_compare_xmode_var,
            values=("Epoka", "Postęp %"),
            state="readonly",
            width=12,
        )
        xmode_combo.pack(side=tk.LEFT, padx=(0, 12))
        xmode_combo.bind("<<ComboboxSelected>>", lambda _event: self._draw_training_compare_chart())

        self._training_compare_smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Wygładź krzywe",
            variable=self._training_compare_smooth_var,
            command=self._draw_training_compare_chart,
        ).pack(side=tk.LEFT, padx=(0, 12))

        self._training_compare_chart_hint_lbl = ttk.Label(
            controls,
            text="",
            style="PanelMuted.TLabel",
            anchor=tk.W,
            justify=tk.LEFT,
        )
        self._training_compare_chart_hint_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        chart_pane = ttk.PanedWindow(shell, orient=tk.HORIZONTAL)
        chart_pane.pack(fill=tk.BOTH, expand=True)
        chart_box = ttk.LabelFrame(chart_pane, text=" Krzywe ", padding=7)
        legend_box = ttk.LabelFrame(chart_pane, text=" Legenda ", padding=7)
        chart_pane.add(chart_box, weight=5)
        chart_pane.add(legend_box, weight=2)

        self._training_compare_canvas = tk.Canvas(
            chart_box,
            bg=palette.get("panel", "#252526"),
            highlightthickness=1,
            highlightbackground=palette.get("border", "#3c3c3c"),
        )
        self._training_compare_canvas.pack(fill=tk.BOTH, expand=True)
        self._training_compare_canvas.bind("<Configure>", lambda _event: self._draw_training_compare_chart())
        self._training_compare_canvas.bind("<Motion>", self._on_training_compare_canvas_motion)
        self._training_compare_hover_lbl = ttk.Label(
            chart_box,
            text="Najedź na punkt krzywej, aby zobaczyć run, epokę i wartość.",
            style="PanelMuted.TLabel",
            anchor=tk.W,
        )
        self._training_compare_hover_lbl.pack(fill=tk.X, pady=(6, 0))

        legend_canvas = tk.Canvas(
            legend_box,
            highlightthickness=0,
            bg=palette.get("panel", "#252526"),
            height=140 if compact_dialog else 180,
        )
        legend_scroll = WebSlimScrollbar(legend_box, orient=tk.VERTICAL, command=legend_canvas.yview)
        legend_canvas.configure(yscrollcommand=legend_scroll.set)
        legend_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        legend_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._training_compare_legend_frame = tk.Frame(legend_canvas, bg=palette.get("panel", "#252526"))
        legend_canvas.create_window((0, 0), window=self._training_compare_legend_frame, anchor="nw")
        self._training_compare_legend_frame.bind(
            "<Configure>",
            lambda _event: legend_canvas.configure(scrollregion=legend_canvas.bbox("all")),
        )

        actions = ttk.Frame(shell, style="Panel.TFrame")
        actions.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(
            actions,
            text="Szczegóły wskazanego runu",
            command=self._open_training_compare_selected_details,
            style="WorkflowCard.TButton",
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._training_compare_promote_btn = ttk.Button(
            actions,
            text="Wybierz wskazany model jako wynik T06",
            command=self._promote_training_compare_selected_run,
            style="WorkflowCard.TButton",
        )
        self._training_compare_promote_btn.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="Zamknij", command=self._close_training_compare_dialog).pack(side=tk.RIGHT)

    try:
        if getattr(self, "_training_compare_promote_btn", None) is not None:
            state = tk.NORMAL if CAMPAIGN.get_active_project_name() else tk.DISABLED
            self._training_compare_promote_btn.configure(state=state)
    except Exception:
        pass

    _populate_training_compare_summary(self)
    _refresh_training_compare_legend(self)
    try:
        first = self._training_compare_runs_data[0]["run_id"]
        self._training_compare_summary_tree.selection_set(first)
        self._training_compare_summary_tree.focus(first)
    except Exception:
        pass
    try:
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
        dialog.after_idle(self._draw_training_compare_chart)
    except Exception:
        pass


def _close_training_compare_dialog(self):
    dialog = getattr(self, "_training_compare_dialog", None)
    if dialog is not None:
        try:
            dialog.destroy()
        except Exception:
            pass
    self._training_compare_dialog = None
    self._training_compare_canvas = None
    self._training_compare_summary_tree = None
    self._training_compare_legend_frame = None
    self._training_compare_hover_points = []
    self._training_compare_runs_data = []


def _open_selected_runs_compare(self):
    runs = _selected_training_compare_runs(self)
    return _open_training_compare_modal(self, runs)
