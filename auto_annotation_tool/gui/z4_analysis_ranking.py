#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zakładka Treningu: trening YOLO + analiza modeli.

W trybie swobodnym Z4 konsumuje gotowy dataset z Z2 lub Z3.
Pomost datasetowy pozostaje tylko na potrzeby kampanii.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import datetime
import time
import webbrowser
import csv
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ..campaign_manager import CAMPAIGN
from ..config import (
    CONFIG,
    YOLO_AVAILABLE,
    AVAILABLE_POSE_MODELS,
    AVAILABLE_DETECT_MODELS,
    PIL_AVAILABLE,
    get_torch_module,
    get_yolo_class,
    is_cuda_available,
    logger,
)
from ..icons import IconManager
from ..validators import validate_model_file, format_yolo_model_identity
from ..training import YOLOPoseTrainer, TrainingHistory, TrainingStatus, DatasetCreator, DatasetSplitter
from ..training.training_report import TrainingReportGenerator
from ..ranking import ModelRanking, format_ranking_model_label, is_plate_pose_model_path
from ..utils import cleanup_gpu_memory, safe_load_yaml, get_image_files
from .help_manager import HELP
from .inertial_scroll import InertialScrollController
from .dataset_display import build_dataset_display_ref
from .model_display import build_model_display_ref
from .run_display import build_run_display_ref
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .zoomable_canvas import ZoomableCanvas
from .z4_train_progress_bar import TrainProgressBar
from .z4_campaign_flow import (
    build_step4_campaign_navigation_view_model,
    build_step4_dataset_workflow_view_model,
    build_step4_training_inputs_view_model,
    clear_campaign_context,
    complete_campaign_project,
    finish_campaign_step4,
    get_campaign_training_target,
    open_campaign_step4_entry,
    poll_training_completion,
    restore_step4_campaign_project_state,
    set_campaign_context,
    set_campaign_training_target,
)
from .z4_flow_models import (
    CharYoloDatasetSourceAdapter,
    PlateXmlImagesSourceAdapter,
    TrainingSource,
    TrainingSourceStats,
)
from .z4_free_mode_flow import (
    refresh_free_training_route_cards,
    refresh_free_training_route_ui,
    update_step4_notebook_mode,
)
from .z4_shared_ui import (
    accept_training_input_context,
    clear_step4_guidance,
    guide_step4_builder_action,
    guide_step4_finish_action,
    guide_step4_next_action,
    guide_step4_route_selection,
    mark_step4_dataset_ready,
    open_step4_dataset_stage,
    refresh_step4_campaign_builder_inputs_ui,
    refresh_step4_analysis_tab_visibility,
    refresh_step4_campaign_navigation_ui,
    refresh_step4_dataset_mode_ui,
    refresh_step4_training_inputs_mode_ui,
    set_step4_dataset_mode,
    sync_step4_analysis_nav_buttons,
    step4_dataset_go_back,
    step4_dataset_go_next,
    step4_train_go_back,
)
from . import z4_dataset_sources
from . import z4_training_metrics
from . import z4_dataset_builder
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None

RANKING_TASK_PLATE = "Tablice (Pose)"
RANKING_TASK_CHAR = "Znaki (Detect)"


def _ranking_task_label_for_target(target: str | None) -> str:
    normalized = CONFIG.normalize_task_target(target)
    return RANKING_TASK_CHAR if normalized == "char" else RANKING_TASK_PLATE


def _is_char_detect_model_path(path_like) -> bool:
    raw = str(path_like or "").strip()
    if not raw:
        return False
    try:
        path = Path(raw)
    except Exception:
        path = Path(str(raw))
    name = path.name.lower()
    parts = {str(part).lower() for part in path.parts}
    if not name.endswith(".pt"):
        return False
    if is_plate_pose_model_path(path):
        return False
    if "pose" in name or "plates" in parts or "plate" in parts:
        return False
    return True


def _ranking_metric_percent(value) -> float:
    try:
        numeric = float(value or 0.0)
    except Exception:
        return 0.0
    if 0.0 <= numeric <= 1.0:
        return numeric * 100.0
    return numeric


def _extract_yolo_ranking_metrics(metrics, *, split_name: str, total_images: int) -> dict:
    results_dict = {}
    try:
        if hasattr(metrics, "results_dict") and isinstance(metrics.results_dict, dict):
            results_dict = dict(metrics.results_dict)
    except Exception:
        results_dict = {}

    lowered = {str(k or "").strip().lower(): v for k, v in results_dict.items()}

    def from_dict(*keys: str) -> float | None:
        for key in keys:
            raw_key = str(key or "").strip().lower()
            if raw_key in lowered:
                try:
                    return float(lowered[raw_key])
                except Exception:
                    return None
        return None

    precision = from_dict("metrics/precision(b)", "metrics/precision", "precision")
    recall = from_dict("metrics/recall(b)", "metrics/recall", "recall")
    map50 = from_dict("metrics/map50(b)", "metrics/map50", "map50")
    map50_95 = from_dict("metrics/map50-95(b)", "metrics/map50-95", "map50_95", "map")

    try:
        box = getattr(metrics, "box", None)
    except Exception:
        box = None
    if box is not None:
        if precision is None:
            precision = getattr(box, "mp", None)
        if recall is None:
            recall = getattr(box, "mr", None)
        if map50 is None:
            map50 = getattr(box, "map50", None)
        if map50_95 is None:
            map50_95 = getattr(box, "map", None)

    return {
        "total_images": int(total_images or 0),
        "accuracy": _ranking_metric_percent(map50_95),
        "precision": _ranking_metric_percent(precision),
        "recall": _ranking_metric_percent(recall),
        "map50": _ranking_metric_percent(map50),
        "map50_95": _ranking_metric_percent(map50_95),
        "split_name": str(split_name or "").strip(),
        "metrics_source": "YOLO val",
    }
def _collect_run_analysis_paths(self, run) -> list[Path]:
    if run is None:
        return []

    run_dir = Path(getattr(run, "output_dir", "") or "")
    if not run_dir.exists():
        return []

    train_dir = run_dir / "train"
    try:
        if (train_dir / "results.csv").exists():
            TrainingReportGenerator.generate_csv_charts(train_dir, run_dir / "plots")
    except Exception as e:
        logger.debug(f"Nie udało się odświeżyć czytelnych wykresów results.csv: {e}")

    priority_order = (
        "results_metrics_from_csv",
        "train_val_losses_from_csv",
        "train_learning_rate_from_csv",
        "results",
        "confusion_matrix",
        "pr_curve",
        "f1_curve",
        "p_curve",
        "r_curve",
        "labels",
        "val",
        "train",
    )

    candidates: list[Path] = []
    for path in run_dir.rglob("*"):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        lower_name = path.name.lower()
        if any(token in lower_name for token in priority_order):
            candidates.append(path)

    def sort_key(path: Path):
        lower_name = path.name.lower()
        priority = next((idx for idx, token in enumerate(priority_order) if token in lower_name), len(priority_order))
        return (priority, lower_name)

    native_paths = sorted(candidates, key=sort_key)
    if native_paths:
        return native_paths

    return self._build_fallback_run_analysis_paths(run)

def _load_run_results_csv_rows(self, run) -> list[dict[str, str]]:
    if run is None:
        return []

    run_dir = Path(getattr(run, "output_dir", "") or "")
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
    except Exception as e:
        logger.debug(f"Nie udało się odczytać results.csv dla analizy runu: {e}")
        return []
    return rows

def _run_analysis_font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def _wrap_run_analysis_line(self, label: str, value: str, width: int = 110) -> list[str]:
    base = f"{label}: {value}".strip()
    if not base:
        return [""]
    return textwrap.wrap(base, width=width, break_long_words=False, break_on_hyphens=False) or [base]

def _render_run_analysis_sheet(
    self,
    title: str,
    sections: list[tuple[str, list[str]]],
    out_path: Path,
    *,
    width: int = 1500,
) -> Path | None:
    if not PIL_AVAILABLE:
        return None

    font = self._run_analysis_font()
    line_height = 24
    section_gap = 18
    top_pad = 28
    left_pad = 30
    right_pad = 30
    bottom_pad = 28

    total_lines = 2
    for heading, lines in sections:
        total_lines += 1
        total_lines += max(1, len(lines))
        total_lines += 1

    height = max(420, top_pad + bottom_pad + total_lines * line_height + max(0, len(sections) - 1) * section_gap)
    img = Image.new("RGB", (int(width), int(height)), color="#1f2933")
    draw = ImageDraw.Draw(img)

    title_color = "#e8f6ef"
    heading_color = "#8fd19e"
    text_color = "#d8dee9"
    muted_color = "#94a3b8"
    accent_color = "#2d6a4f"

    y = top_pad
    draw.text((left_pad, y), title, fill=title_color, font=font)
    y += line_height + 8
    draw.line((left_pad, y, width - right_pad, y), fill=accent_color, width=2)
    y += 16

    for heading, lines in sections:
        draw.text((left_pad, y), heading, fill=heading_color, font=font)
        y += line_height
        section_lines = lines or ["Brak danych."]
        for line in section_lines:
            color = muted_color if str(line or "").strip() == "Brak danych." else text_color
            draw.text((left_pad + 10, y), str(line or ""), fill=color, font=font)
            y += line_height
        y += section_gap

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return out_path
    except Exception as e:
        logger.debug(f"Nie udało się zapisać syntetycznej planszy analizy runu: {e}")
        return None

def _analysis_plot_info(self, path: Path | str) -> tuple[str, str]:
    name = Path(path).name
    lower = name.lower()
    if lower.startswith("00_podsumowanie") or "podsumowanie" in lower:
        return (
            "Podsumowanie runu",
            "Syntetyczna karta z najważniejszymi ustawieniami i metrykami. To pierwszy punkt kontroli: czy run dotyczy właściwego datasetu, modelu i toru.",
        )
    if "results_metrics_from_csv" in lower:
        return (
            "Czytelne metryki walidacyjne",
            "Wykres z results.csv. B oznacza ramki obiektów, P punkty/narożniki modelu pose. Precision mówi, ile predykcji było trafnych, recall ile obiektów model odnalazł. mAP50 jest łagodniejszą oceną, a mAP50-95 surowszą miarą jakości.",
        )
    if "train_val_losses_from_csv" in lower:
        return (
            "Czytelne straty train/val",
            "Wykres z results.csv. Train to błąd na zbiorze treningowym, val na walidacyjnym. Box dotyczy położenia ramki, cls klasy, dfl granic ramki, pose narożników. Loss powinien maleć; rozjazd train/val może oznaczać przeuczenie.",
        )
    if "train_learning_rate_from_csv" in lower:
        return (
            "Learning rate w czasie treningu",
            "Wykres pokazuje harmonogram współczynnika uczenia. pg0, pg1 i pg2 to grupy parametrów optymalizatora. To nie jest miara jakości modelu, tylko kontekst wyjaśniający tempo zmian metryk i ewentualne skoki loss.",
        )
    if lower.startswith("01_przebieg") or "results" in lower:
        return (
            "Przebieg treningu: loss i mAP",
            "Patrz na trend: loss powinien maleć, a mAP50 oraz mAP50-95 rosnąć lub stabilizować się. Nagłe skoki, spadki albo rozjazd metryk sugerują za mały zbiór, przeuczenie albo niestabilny trening.",
        )
    if "confusion_matrix" in lower:
        return (
            "Macierz pomyłek",
            "Pokazuje, które klasy model myli ze sobą. Najlepiej, gdy dominują wartości na przekątnej. Mocne pola poza przekątną wskazują klasy wymagające lepszych danych lub korekty etykiet.",
        )
    if "pr_curve" in lower:
        return (
            "Krzywa Precision-Recall",
            "Pokazuje kompromis między precyzją a czułością. Im bliżej prawego górnego obszaru, tym stabilniejszy model. Słaba krzywa oznacza, że model gubi obiekty albo generuje dużo fałszywych trafień.",
        )
    if "f1_curve" in lower:
        return (
            "F1 względem progu pewności",
            "Pomaga dobrać próg confidence. Szczyt krzywej pokazuje okolice najlepszego kompromisu między precision i recall.",
        )
    if "p_curve" in lower:
        return (
            "Precision względem progu pewności",
            "Pokazuje, jak rośnie czystość predykcji po podnoszeniu progu confidence. Wysoka precyzja oznacza mniej fałszywych trafień.",
        )
    if "r_curve" in lower:
        return (
            "Recall względem progu pewności",
            "Pokazuje, ile właściwych obiektów model odnajduje przy różnych progach confidence. Spadek recall przy wysokim progu jest normalny.",
        )
    if "labels" in lower:
        return (
            "Rozkład etykiet w datasecie",
            "Kontrola danych wejściowych: liczność klas, położenia i rozmiary anotacji. Nierówny rozkład może tłumaczyć słabsze wyniki wybranych klas.",
        )
    if "val" in lower and "pred" in lower:
        return (
            "Predykcje na walidacji",
            "Podgląd tego, co model faktycznie przewiduje na obrazach walidacyjnych. Szukaj przesuniętych ramek, braków i podwójnych detekcji.",
        )
    if "val" in lower and ("label" in lower or "labels" in lower):
        return (
            "Etykiety walidacyjne",
            "Materiał odniesienia dla predykcji walidacyjnych. Porównaj z widokiem predykcji, aby zrozumieć, czy problem leży w modelu czy w danych.",
        )
    if "train" in lower:
        return (
            "Próbka treningowa",
            "Podgląd obrazów używanych w treningu. Sprawdź, czy anotacje wyglądają poprawnie i czy augmentacje nie zniekształcają materiału.",
        )
    return (
        name,
        "Artefakt zapisany przez trening. Jeśli nie jest jasny, porównaj go z results.csv i podglądem predykcji walidacyjnych.",
    )

def _analysis_plot_list_label(self, path: Path | str) -> str:
    title, _description = _analysis_plot_info(self, path)
    name = Path(path).name
    if title == name:
        return name
    return f"{title}  |  {name}"

def _build_fallback_run_analysis_paths(self, run) -> list[Path]:
    if run is None or not PIL_AVAILABLE:
        return []

    run_dir = Path(getattr(run, "output_dir", "") or "")
    if not run_dir.exists():
        return []

    cache_dir = run_dir / "_analysis_cache"
    detail_rows = self._build_training_run_detail_rows(run)
    metric_rows = self._build_training_run_metric_rows(run)
    csv_rows = self._load_run_results_csv_rows(run)

    sections_summary: list[tuple[str, list[str]]] = [
        (
            "Podsumowanie runu",
            [
                line
                for label, value in detail_rows
                for line in self._wrap_run_analysis_line(label, value, width=118)
            ] or ["Brak danych."],
        ),
        (
            "Najważniejsze metryki",
            [
                f"{label}: ostatnia {current} | najlepsza {best} | ocena {band}"
                for label, current, best, band in metric_rows
            ] or ["Brak danych."],
        ),
    ]

    epoch_lines: list[str] = []
    if csv_rows:
        for row in csv_rows[-12:]:
            epoch_value = str(row.get("epoch", "") or row.get("Epoch", "") or "-").strip()
            loss_value = str(
                row.get("train/box_loss", "")
                or row.get("train/loss", "")
                or row.get("loss", "")
                or "-"
            ).strip()
            map50_value = str(
                row.get("metrics/mAP50(B)", "")
                or row.get("metrics/mAP50", "")
                or row.get("map50", "")
                or "-"
            ).strip()
            map95_value = str(
                row.get("metrics/mAP50-95(B)", "")
                or row.get("metrics/mAP50-95", "")
                or row.get("map50_95", "")
                or "-"
            ).strip()
            epoch_lines.append(
                f"Epoka {epoch_value}: loss {loss_value} | mAP50 {map50_value} | mAP50-95 {map95_value}"
            )
    else:
        history = list(getattr(run, "metrics_history", []) or [])
        for item in history[-12:]:
            epoch_lines.append(
                "Epoka "
                f"{int(item.get('epoch', 0) or 0)}: "
                f"loss {float(item.get('loss', 0.0) or 0.0):.4f} | "
                f"mAP50 {float(item.get('map50', 0.0) or 0.0):.4f} | "
                f"mAP50-95 {float(item.get('map50_95', 0.0) or 0.0):.4f}"
            )

    sections_epochs: list[tuple[str, list[str]]] = [
        (
            "Przebieg epok",
            epoch_lines or ["Brak danych epok w results.csv ani metrics_history."],
        ),
        (
            "Artefakty runu",
            [
                f"Folder runu: {self._format_workspace_relative_path(run_dir)}",
                f"Plik wyników CSV: {self._format_workspace_relative_path(run_dir / 'train' / 'results.csv') if (run_dir / 'train' / 'results.csv').exists() else 'brak'}",
                f"Najlepsze wagi: {self._format_workspace_relative_path(getattr(run, 'best_weights', '') or '-')}",
                f"Checkpoint last.pt: {self._format_workspace_relative_path(getattr(run, 'last_weights', '') or '-')}",
            ],
        ),
    ]

    generated: list[Path] = []
    summary_path = cache_dir / "00_podsumowanie_runu.png"
    epochs_path = cache_dir / "01_przebieg_epok.png"

    rendered_summary = self._render_run_analysis_sheet(
        "Analiza runu treningowego",
        sections_summary,
        summary_path,
    )
    if rendered_summary is not None:
        generated.append(rendered_summary)

    rendered_epochs = self._render_run_analysis_sheet(
        "Przebieg treningu i artefakty",
        sections_epochs,
        epochs_path,
    )
    if rendered_epochs is not None:
        generated.append(rendered_epochs)

    return generated

def _close_analysis_dialog(self):
    dialog = getattr(self, "_analysis_dialog", None)
    if dialog is not None:
        try:
            dialog.destroy()
        except Exception:
            pass
    self._analysis_dialog = None
    self._analysis_dialog_shell = None
    self._analysis_plot_paths = []
    self._analysis_plot_canvas = None
    self._analysis_plots_list = None
    self._analysis_plot_title_lbl = None
    self._analysis_plot_hint_lbl = None

def _style_analysis_dialog(self):
    dialog = getattr(self, "_analysis_dialog", None)
    if dialog is None:
        return
    try:
        if not dialog.winfo_exists():
            return
    except Exception:
        return

    palette = getattr(self.app, "palette", {})
    try:
        dialog.configure(bg=palette.get("bg", "#1e1e1e"))
    except Exception:
        pass
    try:
        self.app.style_panel_surface(
            getattr(self, "_analysis_dialog_shell", None),
            background=palette.get("panel", "#252526"),
        )
    except Exception:
        pass
    try:
        self.app.style_listbox_widget(
            getattr(self, "_analysis_plots_list", None),
            bordercolor=palette.get("console_border", palette.get("border", "#3c3c3c")),
        )
    except Exception:
        pass
    try:
        self.app.style_canvas_widget(
            getattr(self, "_analysis_plot_canvas", None),
            background=palette.get("panel", "#252526"),
            bordercolor=palette.get("console_border", palette.get("border", "#3c3c3c")),
        )
    except Exception:
        pass

def _populate_analysis_dialog(self, plot_paths: list[Path]):
    self._analysis_plot_paths = list(plot_paths or [])
    listbox = getattr(self, "_analysis_plots_list", None)
    if listbox is None:
        return

    try:
        listbox.delete(0, tk.END)
    except Exception:
        pass

    for path in self._analysis_plot_paths:
        try:
            listbox.insert(tk.END, self._analysis_plot_list_label(path))
        except Exception:
            continue

    if self._analysis_plot_paths:
        try:
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(0)
            listbox.activate(0)
        except Exception:
            pass
        self._show_analysis_plot(self._analysis_plot_paths[0])

def _open_run_analysis_window(self, run):
    palette = getattr(self.app, "palette", {})
    plot_paths = self._collect_run_analysis_paths(run)
    if not plot_paths:
        return messagebox.showinfo(
            "Brak wykresow",
            "Dla wybranego runu nie znaleziono artefaktow analitycznych Ultralytics.",
        )

    if not PIL_AVAILABLE:
        self._open_run_folder()
        return messagebox.showinfo(
            "Brak podgladu obrazów",
            "Brakuje biblioteki PIL, wiec otworzylem folder runu zamiast podgladu wykresow.",
        )

    dialog = getattr(self, "_analysis_dialog", None)
    dialog_exists = False
    if dialog is not None:
        try:
            dialog_exists = bool(dialog.winfo_exists())
        except Exception:
            dialog_exists = False

    if not dialog_exists:
        dialog = tk.Toplevel(self.frame)
        dialog.title("Analiza treningu")
        dialog.geometry("1360x820")
        dialog.minsize(1120, 680)
        dialog.transient(self.frame.winfo_toplevel())
        dialog.resizable(True, True)
        dialog.protocol("WM_DELETE_WINDOW", self._close_analysis_dialog)
        self._analysis_dialog = dialog

        shell = ttk.Frame(dialog, padding=10, style="Panel.TFrame")
        shell.pack(fill=tk.BOTH, expand=True)
        self._analysis_dialog_shell = shell

        ttk.Label(
            shell,
            text="Artefakty treningu Ultralytics dla wybranego runu. Po lewej wybierasz wykres, po prawej masz podgląd i krótką interpretację.",
            style="PanelMuted.TLabel",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=1180,
        ).pack(anchor=tk.W, fill=tk.X, pady=(0, 10))

        pane = ttk.PanedWindow(shell, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(pane, text=" Wykresy ", padding=8)
        right = ttk.LabelFrame(pane, text=" Podgląd i interpretacja ", padding=8)
        pane.add(left, weight=2)
        pane.add(right, weight=5)

        list_shell = ttk.Frame(left, style="Panel.TFrame")
        list_shell.pack(fill=tk.BOTH, expand=True)
        self._analysis_plots_list = tk.Listbox(
            list_shell,
            height=16,
            font=("Consolas", 9),
            activestyle="none",
            exportselection=False,
        )
        self._analysis_plots_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll = WebSlimScrollbar(list_shell, orient=tk.VERTICAL, command=self._analysis_plots_list.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._analysis_plots_list.configure(yscrollcommand=list_scroll.set)
        self._analysis_plots_list.bind("<<ListboxSelect>>", self._on_analysis_plot_selected)

        canvas_shell = ttk.Frame(right, style="Panel.TFrame")
        canvas_shell.pack(fill=tk.BOTH, expand=True)

        plot_info = ttk.Frame(canvas_shell, style="Panel.TFrame")
        plot_info.pack(fill=tk.X, pady=(0, 8))
        self._analysis_plot_title_lbl = ttk.Label(
            plot_info,
            text="Wybierz wykres",
            style="PanelTitle.TLabel",
            anchor=tk.W,
        )
        self._analysis_plot_title_lbl.pack(anchor=tk.W, fill=tk.X)
        self._analysis_plot_hint_lbl = ttk.Label(
            plot_info,
            text="Po wyborze wykresu pokażę krótki opis i podpowiedź interpretacji.",
            style="PanelMuted.TLabel",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=860,
        )
        self._analysis_plot_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(3, 0))

        controls = ttk.Frame(canvas_shell, style="Panel.TFrame")
        controls.pack(fill=tk.X, pady=(0, 6))

        def _control_analysis_canvas(action: str):
            canvas = getattr(self, "_analysis_plot_canvas", None)
            if canvas is None or getattr(canvas, "original_image", None) is None:
                return "break"
            try:
                canvas.focus_set()
            except Exception:
                pass
            if action == "fit":
                canvas.fit_to_view()
            elif action == "reset":
                canvas.reset_view()
            elif action in ("zoom_in", "zoom_out"):
                try:
                    factor = 1.22 if action == "zoom_in" else (1.0 / 1.22)
                    state = canvas.get_view_state()
                    state["zoom_level"] = max(
                        float(getattr(canvas, "min_zoom", 0.1) or 0.1),
                        min(
                            float(getattr(canvas, "max_zoom", 5.0) or 5.0),
                            float(state.get("zoom_level", getattr(canvas, "zoom_level", 1.0)) or 1.0) * factor,
                        ),
                    )
                    canvas.set_view_state(state, redraw=True)
                except Exception:
                    pass
            elif action == "info":
                try:
                    canvas.show_info = not bool(getattr(canvas, "show_info", True))
                    canvas.refresh_overlay_only(skip_info=False)
                except Exception:
                    pass
            return "break"

        ttk.Label(
            controls,
            text="Podgląd:",
            style="PanelMuted.TLabel",
        ).pack(side=tk.LEFT, padx=(0, 6))
        for label, action in (
            ("Dopasuj", "fit"),
            ("Reset", "reset"),
            ("Zoom +", "zoom_in"),
            ("Zoom -", "zoom_out"),
            ("Info", "info"),
        ):
            ttk.Button(
                controls,
                text=label,
                command=lambda a=action: _control_analysis_canvas(a),
                style="WorkflowCard.TButton",
            ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Label(
            controls,
            text="Rolka: zoom | LPM + drag: przesuwanie | R/Home: reset | I: info",
            style="PanelMuted.TLabel",
        ).pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)

        self._analysis_plot_canvas = ZoomableCanvas(
            canvas_shell,
            bg=palette.get("panel", "#252526"),
            highlightthickness=0,
        )
        try:
            self._analysis_plot_canvas.resampling_quality = Image.Resampling.LANCZOS
        except Exception:
            pass
        self._analysis_plot_canvas.pack(fill=tk.BOTH, expand=True)
        self._analysis_plot_canvas.bind("<Double-1>", lambda _event: _control_analysis_canvas("fit"), add="+")
        dialog.bind("<r>", lambda _event: _control_analysis_canvas("reset"), add="+")
        dialog.bind("<R>", lambda _event: _control_analysis_canvas("reset"), add="+")
        dialog.bind("<Home>", lambda _event: _control_analysis_canvas("reset"), add="+")
        dialog.bind("<i>", lambda _event: _control_analysis_canvas("info"), add="+")
        dialog.bind("<I>", lambda _event: _control_analysis_canvas("info"), add="+")

    try:
        self._analysis_dialog.title(f"Analiza treningu | {self._shorten_training_text(getattr(run, 'name', ''), 48)}")
        self._analysis_dialog.deiconify()
        self._analysis_dialog.lift()
        self._analysis_dialog.focus_force()
    except Exception:
        pass

    self._style_analysis_dialog()
    self._populate_analysis_dialog(plot_paths)

def _open_selected_run_analysis(self, event=None):
    if event is not None and hasattr(self, "tree") and getattr(event, "y", None) is not None:
        try:
            row_id = self.tree.identify_row(event.y)
        except Exception:
            row_id = ""
        if not row_id:
            return
        try:
            self.tree.selection_set(row_id)
            self.tree.focus(row_id)
        except Exception:
            pass
        self._on_run_selected()

    run = self._selected_run()
    if run is None:
        return
    self._open_run_analysis_window(run)

def _collect_project_ranking_model_candidates(self, target: str | None = None) -> list[Path]:
    if not CAMPAIGN.get_active_project_name():
        return []
    normalized_target = CONFIG.normalize_task_target(target or self._get_ranking_task_target())
    if normalized_target not in {"plate", "char"}:
        normalized_target = "plate"

    histories: list[TrainingHistory] = []
    try:
        project_root = CAMPAIGN.get_active_project_root_dir()
        if project_root is not None:
            project_runs_dir = Path(project_root) / "5_training_runs"
            if project_runs_dir.exists():
                histories.append(TrainingHistory(history_dir=project_runs_dir))
    except Exception:
        pass

    current_history = getattr(self, "history", None)
    if current_history is not None:
        try:
            current_dir = Path(getattr(current_history, "history_dir", ""))
            known_dirs = {str(Path(getattr(item, "history_dir", "")).resolve()) for item in histories}
            if str(current_dir.resolve()) not in known_dirs:
                histories.append(current_history)
        except Exception:
            histories.append(current_history)

    candidates: list[Path] = []
    seen: set[str] = set()

    def add_path(path_like) -> None:
        if not path_like:
            return
        try:
            path = Path(path_like)
        except Exception:
            return
        if not path.exists() or not path.is_file():
            return
        try:
            key = str(path.resolve()).lower()
            resolved = path.resolve()
        except Exception:
            key = str(path).lower()
            resolved = path
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    for history in histories:
        try:
            runs = list(history.get_all_runs() or [])
        except Exception:
            runs = []
        for run in runs:
            status = str(getattr(run, "status", "") or "").strip().lower()
            if status != TrainingStatus.COMPLETED.value:
                continue
            try:
                target = str(self._infer_history_run_target(run) or "").strip().lower()
            except Exception:
                target = ""
            if target != normalized_target:
                continue
            try:
                add_path(self._resolve_history_run_best_weights(run))
            except Exception:
                pass

    try:
        explicit_project_model = str(CAMPAIGN.get_global_model(normalized_target) or "").strip()
        if explicit_project_model:
            add_path(explicit_project_model)
    except Exception:
        pass

    return candidates


def _collect_project_plate_ranking_model_candidates(self) -> list[Path]:
    return _collect_project_ranking_model_candidates(self, "plate")


def _collect_ranking_model_candidates(self, models_dir: Path, target: str | None = None) -> list[Path]:
    normalized_target = CONFIG.normalize_task_target(target or self._get_ranking_task_target())
    if normalized_target not in {"plate", "char"}:
        normalized_target = "plate"
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_path(path_like, *, require_domain_name: bool = True) -> None:
        if not path_like:
            return
        try:
            path = Path(path_like)
        except Exception:
            return
        if not path.exists() or not path.is_file():
            return
        if require_domain_name:
            if normalized_target == "plate" and not is_plate_pose_model_path(path):
                return
            if normalized_target == "char" and not _is_char_detect_model_path(path):
                return
        try:
            resolved = path.resolve()
            key = str(resolved).lower()
        except Exception:
            resolved = path
            key = str(path).lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    try:
        for path in sorted(Path(models_dir).rglob("*.pt")):
            add_path(path, require_domain_name=True)
    except Exception:
        pass

    for path in _collect_project_ranking_model_candidates(self, normalized_target):
        # Projektowe runy zwykle zapisują wagę jako best.pt, więc domenę
        # bierzemy z historii treningu, a nie z nazwy pliku.
        add_path(path, require_domain_name=False)

    return candidates


def _get_ranking_scope(self) -> str:
    raw = ""
    try:
        scope_var = getattr(self, "rank_scope_var", None)
        if scope_var is not None:
            raw = str(scope_var.get() or "").strip()
    except Exception:
        raw = ""
    if raw in {"Projekt", "Globalne", "Wszystkie"}:
        return raw
    try:
        return "Projekt" if CAMPAIGN.get_active_project_name() else "Wszystkie"
    except Exception:
        return "Wszystkie"


def _format_ranking_scope_label(self, scope: str | None = None, target: str | None = None) -> str:
    selected_scope = scope or _get_ranking_scope(self)
    normalized_target = CONFIG.normalize_task_target(target or self._get_ranking_task_target())
    short_name = "MZ" if normalized_target == "char" else "MT"
    if selected_scope == "Projekt":
        return f"Projektowe {short_name}"
    if selected_scope == "Globalne":
        return f"Globalne {short_name}"
    return f"Wszystkie {short_name}"


def _ranking_model_candidate_scope(self, path_like, target: str | None = None) -> str:
    raw = str(path_like or "").strip()
    if not raw:
        return "Globalne"
    normalized_target = CONFIG.normalize_task_target(target or self._get_ranking_task_target())
    project_keys: set[str] = set()
    try:
        if CAMPAIGN.get_active_project_name():
            for candidate in _collect_project_ranking_model_candidates(self, normalized_target):
                key = _ranking_path_key(candidate)
                if key:
                    project_keys.add(key)
    except Exception:
        project_keys = set()

    try:
        model_path = Path(raw).resolve()
        model_key = str(model_path).lower()
    except Exception:
        model_path = Path(raw)
        model_key = str(model_path).lower()
    if model_key in project_keys:
        return "Projekt"

    try:
        project_root = CAMPAIGN.get_active_project_root_dir()
        project_root = Path(project_root).resolve() if project_root is not None else None
        if project_root is not None and (model_path == project_root or project_root in model_path.parents):
            return "Projekt"
    except Exception:
        pass
    return "Globalne"


def _collect_ranking_participant_candidates(
    self,
    models_dir: Path | None,
    target: str | None = None,
    scope: str | None = None,
) -> list[Path]:
    normalized_target = CONFIG.normalize_task_target(target or self._get_ranking_task_target())
    if normalized_target not in {"plate", "char"}:
        normalized_target = "plate"
    selected_scope = scope or _get_ranking_scope(self)
    if selected_scope not in {"Projekt", "Globalne", "Wszystkie"}:
        selected_scope = "Wszystkie"

    candidates: list[Path] = []
    seen: set[str] = set()

    def add_path(path_like, *, require_domain_name: bool = True, force_scope: str | None = None) -> None:
        if not path_like:
            return
        try:
            path = Path(path_like)
        except Exception:
            return
        if not path.exists() or not path.is_file():
            return
        if require_domain_name:
            if normalized_target == "plate" and not is_plate_pose_model_path(path):
                return
            if normalized_target == "char" and not _is_char_detect_model_path(path):
                return
        candidate_scope = force_scope or _ranking_model_candidate_scope(self, path, normalized_target)
        if selected_scope in {"Projekt", "Globalne"} and candidate_scope != selected_scope:
            return
        try:
            resolved = path.resolve()
            key = str(resolved).lower()
        except Exception:
            resolved = path
            key = str(path).lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    if selected_scope in {"Projekt", "Wszystkie"}:
        for path in _collect_project_ranking_model_candidates(self, normalized_target):
            # Wyniki projektu często mają nazwę best.pt; domenę znamy z historii runu.
            add_path(path, require_domain_name=False, force_scope="Projekt")

    if selected_scope in {"Globalne", "Wszystkie"} and models_dir is not None:
        try:
            root = Path(models_dir)
            if root.exists() and root.is_dir():
                for path in sorted(root.rglob("*.pt")):
                    add_path(path, require_domain_name=True)
        except Exception:
            pass

    return candidates


def _collect_plate_ranking_model_candidates(self, models_dir: Path) -> list[Path]:
    return _collect_ranking_model_candidates(self, models_dir, "plate")


def _ranking_path_key(path_like) -> str:
    raw = str(path_like or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).resolve()).lower()
    except Exception:
        return str(Path(raw)).lower()


def _is_ranking_participant_enabled(self, path_like) -> bool:
    key = _ranking_path_key(path_like)
    if not key:
        return True
    enabled_map = getattr(self, "_ranking_participant_enabled_by_key", None)
    if not isinstance(enabled_map, dict):
        self._ranking_participant_enabled_by_key = {}
        enabled_map = self._ranking_participant_enabled_by_key
    return bool(enabled_map.get(key, True))


def _set_ranking_participant_enabled(self, path_like, enabled: bool) -> None:
    key = _ranking_path_key(path_like)
    if not key:
        return
    enabled_map = getattr(self, "_ranking_participant_enabled_by_key", None)
    if not isinstance(enabled_map, dict):
        self._ranking_participant_enabled_by_key = {}
        enabled_map = self._ranking_participant_enabled_by_key
    enabled_map[key] = bool(enabled)


def _filter_enabled_ranking_participants(self, paths: list[Path]) -> list[Path]:
    return [Path(path) for path in list(paths or []) if _is_ranking_participant_enabled(self, path)]


def _ranking_report_escape(value) -> str:
    text = str(value if value is not None else "")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _ranking_report_percent(value) -> float:
    try:
        numeric = float(value or 0.0)
    except Exception:
        return 0.0
    if 0.0 < numeric <= 1.0:
        return numeric * 100.0
    return numeric


def _ranking_report_percent_text(value) -> str:
    numeric = _ranking_report_percent(value)
    if numeric <= 0:
        return "-"
    return f"{numeric:.2f}%"


def _ranking_report_datetime_text(value) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "-"
    try:
        return datetime.datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return raw.replace("T", " ")


def _ranking_report_model_label(self, entry, target_task: str) -> tuple[str, str]:
    model_path = str(getattr(entry, "model_path", "") or "").strip()
    model_file = Path(model_path).name if model_path else str(getattr(entry, "model_name", "") or "-")
    run = None
    resolver = getattr(self, "_resolve_training_run_from_model_path", None)
    if callable(resolver) and model_path:
        try:
            run = resolver(Path(model_path))
        except Exception:
            run = None
    if run is not None:
        try:
            return build_run_display_ref(run, kind_hint="training").id, model_file or "best.pt"
        except Exception:
            try:
                return self._format_training_model_run_label(run), model_file or "best.pt"
            except Exception:
                pass
    return (
        format_ranking_model_label(
            getattr(entry, "model_name", ""),
            model_path,
            getattr(entry, "task_type", target_task),
        ),
        model_file or "-",
    )


def _collect_current_ranking_report_context(self) -> dict:
    self._ensure_plate_ranking_engine()
    target = self._get_ranking_task_target()
    target_task = self._get_ranking_task_label(target)
    selected_scope = _get_ranking_scope(self)
    selected_reference = self._resolve_ranking_reference_source()
    selected_reference_path = str(selected_reference.get("reference_dir") or "").strip()
    selected_reference_raw = str(selected_reference.get("selected_path") or "").strip()
    selected_split = str(selected_reference.get("split_name") or "").strip()
    models_dir_raw = str(getattr(getattr(self, "rank_models_dir", None), "get", lambda: "")() or "").strip()
    try:
        models_dir = Path(models_dir_raw) if models_dir_raw else None
    except Exception:
        models_dir = None

    def normalize_path(path_like) -> str:
        raw = str(path_like or "").strip()
        if not raw:
            return ""
        try:
            return str(Path(raw).resolve()).lower()
        except Exception:
            return str(Path(raw)).lower()

    def entry_scope(entry) -> str:
        try:
            return _ranking_model_candidate_scope(self, getattr(entry, "model_path", ""), target)
        except Exception:
            return "Globalne"

    get_unique_entries = getattr(self.ranking_engine, "get_unique_entries", None)
    ranking_entries = get_unique_entries() if callable(get_unique_entries) else getattr(self.ranking_engine, "entries", [])
    entries = [
        entry for entry in ranking_entries
        if str(getattr(entry, "task_type", "") or "").strip() == target_task
    ]
    if selected_reference_raw and not selected_reference.get("ok"):
        entries = []
    elif selected_reference_path:
        reference_key = normalize_path(selected_reference_path)
        entries = [
            entry for entry in entries
            if normalize_path(getattr(entry, "reference_path", "")) == reference_key
        ]
        if target == "char" and selected_split:
            entries = [
                entry for entry in entries
                if str(getattr(entry, "split_name", "") or "").strip() == selected_split
            ]
    if selected_scope in {"Projekt", "Globalne"}:
        entries = [entry for entry in entries if entry_scope(entry) == selected_scope]
    entries.sort(key=lambda item: float(getattr(item, "ranking_score", getattr(item, "f1_score", 0)) or 0), reverse=True)

    try:
        participants = list(_collect_ranking_participant_candidates(self, models_dir, target, selected_scope) or [])
        participants = _filter_enabled_ranking_participants(self, participants)
    except Exception:
        participants = []
    participant_keys = {_ranking_path_key(path) for path in participants if _ranking_path_key(path)}
    evaluated_keys = {_ranking_path_key(getattr(entry, "model_path", "")) for entry in entries}
    pending_count = len([key for key in participant_keys if key and key not in evaluated_keys])

    rows: list[dict] = []
    for index, entry in enumerate(entries, 1):
        label, model_file = _ranking_report_model_label(self, entry, target_task)
        precision = _ranking_report_percent(getattr(entry, "precision", 0))
        recall = _ranking_report_percent(getattr(entry, "recall", 0))
        f1 = float(getattr(entry, "f1_score", 0) or 0)
        rows.append(
            {
                "rank": index,
                "label": label,
                "model_file": model_file,
                "scope": entry_scope(entry),
                "score": _ranking_report_percent(getattr(entry, "ranking_score", 0)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": _ranking_report_percent(getattr(entry, "accuracy", 0)),
                "map50": _ranking_report_percent(getattr(entry, "map50", 0)),
                "map50_95": _ranking_report_percent(getattr(entry, "map50_95", 0)),
                "sample": int(getattr(entry, "total_images", 0) or 0),
                "total_auto_plates": int(getattr(entry, "total_auto_plates", 0) or 0),
                "total_corrected_plates": int(getattr(entry, "total_corrected_plates", 0) or 0),
                "plates_unchanged": int(getattr(entry, "plates_unchanged", 0) or 0),
                "plates_minor_fix": int(getattr(entry, "plates_minor_fix", 0) or 0),
                "plates_major_fix": int(getattr(entry, "plates_major_fix", 0) or 0),
                "plates_added": int(getattr(entry, "plates_added", 0) or 0),
                "plates_removed": int(getattr(entry, "plates_removed", 0) or 0),
                "reference": str(getattr(entry, "reference_name", "") or selected_reference.get("reference_name") or "-"),
                "split": str(getattr(entry, "split_name", "") or selected_split or "-"),
                "metrics_source": str(getattr(entry, "metrics_source", "") or ("YOLO val" if target == "char" else "CVAT IoU")),
                "model_path": str(getattr(entry, "model_path", "") or ""),
                "evaluated_at": _ranking_report_datetime_text(getattr(entry, "date_evaluated", "")),
            }
        )

    return {
        "target": target,
        "target_task": target_task,
        "scope": selected_scope,
        "scope_label": _format_ranking_scope_label(self, selected_scope, target),
        "reference_info": selected_reference,
        "reference_path": selected_reference_path,
        "reference_name": str(selected_reference.get("reference_name") or "-"),
        "split": selected_split or "-",
        "participant_count": len(participants),
        "pending_count": pending_count,
        "rows": rows,
        "generated_at": datetime.datetime.now(),
    }


def _ranking_report_bar_svg(rows: list[dict], *, title: str) -> str:
    top_rows = rows[:12]
    width = 1180
    row_height = 42
    top = 72
    left = 330
    bar_width = 720
    height = max(220, top + row_height * max(1, len(top_rows)) + 42)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#101820"/>',
        f'<text x="24" y="36" fill="#f4f7f6" font-size="24" font-family="Segoe UI, Arial" font-weight="700">{_ranking_report_escape(title)}</text>',
        '<text x="24" y="58" fill="#9fb0aa" font-size="13" font-family="Segoe UI, Arial">Ocena rankingowa w procentach. Dłuższy pasek oznacza lepszy wynik na tym samym torze testowym.</text>',
    ]
    if not top_rows:
        parts.append('<text x="24" y="110" fill="#f0b44c" font-size="18" font-family="Segoe UI, Arial">Brak wyników do wykresu.</text>')
    for index, row in enumerate(top_rows):
        y = top + index * row_height
        score = max(0.0, min(100.0, float(row.get("score", 0.0) or 0.0)))
        bar_len = (score / 100.0) * bar_width
        fill = "#2ecc71" if index == 0 else "#4aa3ff"
        label = textwrap.shorten(str(row.get("label", "-")), width=44, placeholder="...")
        parts.extend(
            [
                f'<text x="24" y="{y + 20}" fill="#f4f7f6" font-size="14" font-family="Segoe UI, Arial">#{index + 1} {_ranking_report_escape(label)}</text>',
                f'<rect x="{left}" y="{y + 5}" width="{bar_width}" height="22" rx="8" fill="#26343a"/>',
                f'<rect x="{left}" y="{y + 5}" width="{bar_len:.1f}" height="22" rx="8" fill="{fill}"/>',
                f'<text x="{left + bar_width + 18}" y="{y + 22}" fill="#f4f7f6" font-size="15" font-family="Segoe UI, Arial" font-weight="700">{score:.2f}%</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _ranking_report_metrics_svg(rows: list[dict], *, title: str) -> str:
    top_rows = rows[:8]
    width = 1180
    row_height = 58
    top = 86
    left = 330
    bar_width = 210
    gap = 38
    height = max(260, top + row_height * max(1, len(top_rows)) + 48)
    third_metric = (
        ("map50_95", "mAP50-95", "#f0b44c")
        if any(float(row.get("map50_95", 0.0) or 0.0) > 0 for row in top_rows)
        else ("f1", "F1", "#f0b44c")
    )
    metrics = (
        ("precision", "Precyzja", "#2ecc71"),
        ("recall", "Czułość", "#4aa3ff"),
        third_metric,
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#101820"/>',
        f'<text x="24" y="36" fill="#f4f7f6" font-size="24" font-family="Segoe UI, Arial" font-weight="700">{_ranking_report_escape(title)}</text>',
        '<text x="24" y="58" fill="#9fb0aa" font-size="13" font-family="Segoe UI, Arial">Metryki pomocnicze dla najlepszych kandydatów. Porównanie jest miarodajne tylko dla tego samego toru testowego.</text>',
    ]
    for metric_index, (_, label, color) in enumerate(metrics):
        x = left + metric_index * (bar_width + gap)
        parts.append(f'<text x="{x}" y="78" fill="{color}" font-size="13" font-family="Segoe UI, Arial" font-weight="700">{_ranking_report_escape(label)}</text>')
    if not top_rows:
        parts.append('<text x="24" y="120" fill="#f0b44c" font-size="18" font-family="Segoe UI, Arial">Brak wyników do wykresu.</text>')
    for index, row in enumerate(top_rows):
        y = top + index * row_height
        label = textwrap.shorten(str(row.get("label", "-")), width=42, placeholder="...")
        parts.append(f'<text x="24" y="{y + 26}" fill="#f4f7f6" font-size="14" font-family="Segoe UI, Arial">#{index + 1} {_ranking_report_escape(label)}</text>')
        for metric_index, (key, _, color) in enumerate(metrics):
            x = left + metric_index * (bar_width + gap)
            value = max(0.0, min(100.0, float(row.get(key, 0.0) or 0.0)))
            bar_len = (value / 100.0) * bar_width
            parts.extend(
                [
                    f'<rect x="{x}" y="{y + 8}" width="{bar_width}" height="16" rx="6" fill="#26343a"/>',
                    f'<rect x="{x}" y="{y + 8}" width="{bar_len:.1f}" height="16" rx="6" fill="{color}"/>',
                    f'<text x="{x}" y="{y + 42}" fill="#dfe8e4" font-size="12" font-family="Segoe UI, Arial">{value:.2f}%</text>',
                ]
            )
    parts.append("</svg>")
    return "\n".join(parts)


def _ranking_report_plate_diffs_svg(rows: list[dict], *, title: str) -> str:
    top_rows = rows[:8]
    width = 1180
    row_height = 74
    top = 92
    left = 340
    bar_width = 650
    height = max(300, top + row_height * max(1, len(top_rows)) + 58)
    segments = (
        ("plates_unchanged", "Bez zmian", "#2ecc71"),
        ("plates_minor_fix", "Małe poprawki", "#86d37a"),
        ("plates_major_fix", "Duże poprawki", "#f0b44c"),
        ("plates_added", "Brakujące", "#e05d5d"),
        ("plates_removed", "Nadmiarowe", "#9b59b6"),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#101820"/>',
        f'<text x="24" y="36" fill="#f4f7f6" font-size="24" font-family="Segoe UI, Arial" font-weight="700">{_ranking_report_escape(title)}</text>',
        '<text x="24" y="58" fill="#9fb0aa" font-size="13" font-family="Segoe UI, Arial">Rozkład zgodności detekcji tablic z anotacją odniesienia. Zielone segmenty oznaczają najmniej pracy korekcyjnej.</text>',
    ]
    legend_x = 24
    for key, label, color in segments:
        parts.extend(
            [
                f'<rect x="{legend_x}" y="72" width="12" height="12" rx="3" fill="{color}"/>',
                f'<text x="{legend_x + 17}" y="83" fill="#dfe8e4" font-size="12" font-family="Segoe UI, Arial">{_ranking_report_escape(label)}</text>',
            ]
        )
        legend_x += 128 if key != "plates_major_fix" else 132
    if not top_rows:
        parts.append('<text x="24" y="128" fill="#f0b44c" font-size="18" font-family="Segoe UI, Arial">Brak wyników do wykresu.</text>')
    for index, row in enumerate(top_rows):
        y = top + index * row_height
        label = textwrap.shorten(str(row.get("label", "-")), width=42, placeholder="...")
        corrected = int(row.get("total_corrected_plates", 0) or 0)
        detected = int(row.get("total_auto_plates", 0) or 0)
        total = max(1, sum(max(0, int(row.get(key, 0) or 0)) for key, _, _ in segments))
        x = left
        parts.append(f'<text x="24" y="{y + 22}" fill="#f4f7f6" font-size="14" font-family="Segoe UI, Arial">#{index + 1} {_ranking_report_escape(label)}</text>')
        parts.append(f'<text x="24" y="{y + 43}" fill="#9fb0aa" font-size="12" font-family="Segoe UI, Arial">wykryte: {detected} | odniesienie: {corrected}</text>')
        parts.append(f'<rect x="{left}" y="{y + 8}" width="{bar_width}" height="24" rx="8" fill="#26343a"/>')
        for key, _label, color in segments:
            value = max(0, int(row.get(key, 0) or 0))
            seg_width = (value / total) * bar_width
            if seg_width <= 0:
                continue
            parts.append(f'<rect x="{x:.1f}" y="{y + 8}" width="{seg_width:.1f}" height="24" fill="{color}"/>')
            x += seg_width
        parts.append(f'<text x="{left + bar_width + 18}" y="{y + 26}" fill="#f4f7f6" font-size="13" font-family="Segoe UI, Arial">{total} ramek</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _ranking_report_markdown(context: dict) -> str:
    generated = context.get("generated_at")
    generated_text = generated.strftime("%Y-%m-%d %H:%M:%S") if hasattr(generated, "strftime") else "-"
    rows = list(context.get("rows") or [])
    reference_info = dict(context.get("reference_info") or {})
    winner = rows[0] if rows else None
    target = str(context.get("target") or "")
    method_score = (
        "mAP50-95 z walidacji YOLO"
        if target == "char"
        else "ocena rankingowa oparta o zgodność detekcji z anotacją odniesienia"
    )
    lines = [
        "# Raport rankingu modeli",
        "",
        "## Kontekst testu",
        "",
        f"- Data raportu: {generated_text}",
        f"- Tryb modelu: {context.get('target_task') or '-'}",
        f"- Zakres uczestników: {context.get('scope_label') or context.get('scope') or '-'}",
        f"- Tor testowy: {context.get('reference_name') or '-'}",
        f"- Ścieżka toru: `{context.get('reference_path') or reference_info.get('selected_path') or '-'}`",
        f"- Split: {context.get('split') or '-'}",
        f"- Liczba obrazów w teście: {int(reference_info.get('image_count', 0) or 0)}",
        f"- Liczba uczestników w zakresie: {int(context.get('participant_count', 0) or 0)}",
        f"- Liczba modeli z wynikiem: {len(rows)}",
        f"- Liczba modeli czekających na test: {int(context.get('pending_count', 0) or 0)}",
        "",
        "## Metoda wyłaniania zwycięzcy",
        "",
        "Ranking porównuje modele wyłącznie w obrębie jednego trybu modelu, jednego zakresu uczestników i jednego toru testowego. Każdy kandydat dostaje ten sam zestaw danych odniesienia, dlatego wynik jest porównywalny tylko w tym konkretnym kontekście.",
        "",
        f"Modele są sortowane malejąco według pola `Ocena`. W tym raporcie ocena oznacza: {method_score}. Metryki `Precyzja`, `Czułość`, `F1`, `mAP50` i `mAP50-95` są metrykami pomocniczymi, które pozwalają opisać, dlaczego dany model wygrał albo przegrał.",
        "",
        "Zwycięzca rankingu jest rekomendacją eksperymentalną. Program nie ustawia modelu projektowego automatycznie, ponieważ ostateczny wybór powinien pozostać jawną decyzją użytkownika.",
        "",
        "## Definicje metryk",
        "",
        "- Precyzja opisuje, jaka część wykryć modelu była trafna.",
        "- Czułość opisuje, jaka część obiektów z toru odniesienia została wykryta.",
        "- F1 jest średnią harmoniczną precyzji i czułości: `F1 = 2 * P * C / (P + C)`.",
        "- mAP50 i mAP50-95 pochodzą z walidacji YOLO, jeśli ranking dotyczy modelu z datasetem YOLO.",
        "- Dla modeli tablic porównanie z zapisanym XML opiera się o dopasowanie ramek przez IoU; szczegółowe liczniki różnic są zapisane w CSV.",
        "",
    ]
    if target == "plate":
        lines.extend(
            [
                "## Analiza zgodności tablic",
                "",
                "Dla modelu tablic raport zapisuje dodatkowy rozkład pracy korekcyjnej: ramki bez zmian, ramki wymagające małej poprawki, ramki wymagające dużej poprawki, brakujące tablice oraz wykrycia nadmiarowe. Ten rozkład jest ważny, bo dwa modele mogą mieć podobną ocenę końcową, ale generować zupełnie inny koszt ręcznej korekty.",
                "",
            ]
        )
    lines.extend(
        [
            "## Wynik",
            "",
        ]
    )
    if winner:
        lines.extend(
            [
                f"- Zwycięzca: **{winner.get('label') or '-'}**",
                f"- Ocena: **{_ranking_report_percent_text(winner.get('score'))}**",
                f"- Precyzja / czułość: {_ranking_report_percent_text(winner.get('precision'))} / {_ranking_report_percent_text(winner.get('recall'))}",
                f"- Źródło metryk: {winner.get('metrics_source') or '-'}",
            ]
        )
    else:
        lines.append("- Brak wyników dla wybranego toru i zakresu.")
    lines.extend(
        [
            "",
            "## Tabela wyników",
            "",
            "| # | Model | Zakres | Ocena | Precyzja | Czułość | F1 | mAP50 | mAP50-95 | Próbka | Oceniono |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {rank} | {label} | {scope} | {score} | {precision} | {recall} | {f1} | {map50} | {map50_95} | {sample} | {evaluated_at} |".format(
                rank=int(row.get("rank", 0) or 0),
                label=str(row.get("label") or "-").replace("|", "\\|"),
                scope=str(row.get("scope") or "-"),
                score=_ranking_report_percent_text(row.get("score")),
                precision=_ranking_report_percent_text(row.get("precision")),
                recall=_ranking_report_percent_text(row.get("recall")),
                f1=_ranking_report_percent_text(row.get("f1")),
                map50=_ranking_report_percent_text(row.get("map50")),
                map50_95=_ranking_report_percent_text(row.get("map50_95")),
                sample=int(row.get("sample", 0) or 0),
                evaluated_at=str(row.get("evaluated_at") or "-"),
            )
        )
    lines.extend(
        [
            "",
            "## Pliki wygenerowane z raportem",
            "",
            "- `ranking_results.csv` - dane tabelaryczne do dalszej analizy.",
            "- `ranking_score.svg` - wykres oceny rankingowej.",
            "- `ranking_metrics.svg` - wykres metryk pomocniczych.",
            "",
        ]
    )
    if target == "plate":
        lines.append("- `ranking_plate_diffs.svg` - wykres zgodności i rodzaju korekt dla modeli tablic.")
    lines.extend(
        [
            "",
            "## Ograniczenia interpretacji",
            "",
            "Porównanie modeli trenowanych na różnych datasetach jest sensowne dopiero wtedy, gdy wszystkie modele zostaną sprawdzone na tym samym torze rankingowym. Zmiana toru, splitu albo zakresu uczestników tworzy nowy eksperyment i wymaga osobnego raportu.",
            "",
        ]
    )
    return "\n".join(lines)


def _draw_ranking_score_canvas(canvas: tk.Canvas, rows: list[dict], palette: dict):
    canvas.delete("all")
    width = max(int(canvas.winfo_width() or 0), 980)
    row_height = 42
    top = 76
    left = 330
    right_pad = 118
    bar_width = max(360, width - left - right_pad)
    height = max(260, top + row_height * max(1, min(len(rows), 14)) + 44)
    bg = palette.get("panel", "#101820")
    fg = palette.get("fg", "#f4f7f6")
    muted = palette.get("muted", "#9fb0aa")
    track = blend_hex_colors(palette.get("panel_border", "#3c3c3c"), bg, 0.42)
    canvas.configure(bg=bg, scrollregion=(0, 0, width, height))
    canvas.create_text(
        24,
        28,
        text="Ranking modeli - ocena",
        fill=fg,
        font=("Segoe UI", 18, "bold"),
        anchor=tk.W,
    )
    canvas.create_text(
        24,
        54,
        text="Dłuższy pasek oznacza lepszy wynik na tym samym torze testowym.",
        fill=muted,
        font=("Segoe UI", 9),
        anchor=tk.W,
    )
    if not rows:
        canvas.create_text(24, 112, text="Brak wyników do wykresu.", fill=palette.get("warning", "#f0b44c"), anchor=tk.W)
        return
    for index, row in enumerate(rows[:14]):
        y = top + index * row_height
        score = max(0.0, min(100.0, float(row.get("score", 0.0) or 0.0)))
        label = textwrap.shorten(str(row.get("label", "-")), width=42, placeholder="...")
        fill = palette.get("success", "#2ecc71") if index == 0 else palette.get("accent", "#4aa3ff")
        canvas.create_text(
            24,
            y + 17,
            text=f"#{index + 1} {label}",
            fill=fg,
            font=("Segoe UI", 10),
            anchor=tk.W,
        )
        canvas.create_rectangle(left, y + 4, left + bar_width, y + 26, fill=track, outline="", width=0)
        canvas.create_rectangle(left, y + 4, left + (score / 100.0) * bar_width, y + 26, fill=fill, outline="", width=0)
        canvas.create_text(
            left + bar_width + 16,
            y + 17,
            text=f"{score:.2f}%",
            fill=fg,
            font=("Segoe UI", 10, "bold"),
            anchor=tk.W,
        )


def _draw_ranking_metrics_canvas(canvas: tk.Canvas, rows: list[dict], palette: dict):
    canvas.delete("all")
    width = max(int(canvas.winfo_width() or 0), 980)
    row_height = 58
    top = 90
    left = 320
    gap = 32
    bar_width = max(130, int((width - left - 90 - gap * 2) / 3))
    height = max(290, top + row_height * max(1, min(len(rows), 10)) + 46)
    bg = palette.get("panel", "#101820")
    fg = palette.get("fg", "#f4f7f6")
    muted = palette.get("muted", "#9fb0aa")
    track = blend_hex_colors(palette.get("panel_border", "#3c3c3c"), bg, 0.42)
    third_metric = (
        ("map50_95", "mAP50-95", palette.get("warning", "#f0b44c"))
        if any(float(row.get("map50_95", 0.0) or 0.0) > 0 for row in rows[:10])
        else ("f1", "F1", palette.get("warning", "#f0b44c"))
    )
    metrics = (
        ("precision", "Precyzja", palette.get("success", "#2ecc71")),
        ("recall", "Czułość", palette.get("accent", "#4aa3ff")),
        third_metric,
    )
    canvas.configure(bg=bg, scrollregion=(0, 0, width, height))
    canvas.create_text(
        24,
        28,
        text="Metryki pomocnicze",
        fill=fg,
        font=("Segoe UI", 18, "bold"),
        anchor=tk.W,
    )
    canvas.create_text(
        24,
        54,
        text="Pomagają opisać przewagi modeli, ale zwycięzca wynika z pola Ocena.",
        fill=muted,
        font=("Segoe UI", 9),
        anchor=tk.W,
    )
    for metric_index, (_, label, color) in enumerate(metrics):
        x = left + metric_index * (bar_width + gap)
        canvas.create_text(x, 78, text=label, fill=color, font=("Segoe UI", 9, "bold"), anchor=tk.W)
    if not rows:
        canvas.create_text(24, 124, text="Brak wyników do wykresu.", fill=palette.get("warning", "#f0b44c"), anchor=tk.W)
        return
    for index, row in enumerate(rows[:10]):
        y = top + index * row_height
        label = textwrap.shorten(str(row.get("label", "-")), width=39, placeholder="...")
        canvas.create_text(24, y + 22, text=f"#{index + 1} {label}", fill=fg, font=("Segoe UI", 10), anchor=tk.W)
        for metric_index, (key, _label, color) in enumerate(metrics):
            x = left + metric_index * (bar_width + gap)
            value = max(0.0, min(100.0, float(row.get(key, 0.0) or 0.0)))
            canvas.create_rectangle(x, y + 8, x + bar_width, y + 24, fill=track, outline="", width=0)
            canvas.create_rectangle(x, y + 8, x + (value / 100.0) * bar_width, y + 24, fill=color, outline="", width=0)
            canvas.create_text(x, y + 44, text=f"{value:.2f}%", fill=fg, font=("Segoe UI", 8), anchor=tk.W)


def _draw_ranking_plate_diffs_canvas(canvas: tk.Canvas, rows: list[dict], palette: dict):
    canvas.delete("all")
    width = max(int(canvas.winfo_width() or 0), 980)
    row_height = 74
    top = 96
    left = 330
    bar_width = max(360, width - left - 150)
    height = max(320, top + row_height * max(1, min(len(rows), 10)) + 52)
    bg = palette.get("panel", "#101820")
    fg = palette.get("fg", "#f4f7f6")
    muted = palette.get("muted", "#9fb0aa")
    track = blend_hex_colors(palette.get("panel_border", "#3c3c3c"), bg, 0.42)
    segments = (
        ("plates_unchanged", "Bez zmian", palette.get("success", "#2ecc71")),
        ("plates_minor_fix", "Małe poprawki", "#86d37a"),
        ("plates_major_fix", "Duże poprawki", palette.get("warning", "#f0b44c")),
        ("plates_added", "Brakujące", palette.get("error", "#e05d5d")),
        ("plates_removed", "Nadmiarowe", "#9b59b6"),
    )
    canvas.configure(bg=bg, scrollregion=(0, 0, width, height))
    canvas.create_text(
        24,
        28,
        text="Analiza detekcji tablic",
        fill=fg,
        font=("Segoe UI", 18, "bold"),
        anchor=tk.W,
    )
    canvas.create_text(
        24,
        54,
        text="Rozkład zgodności z anotacją odniesienia. Im więcej zieleni, tym mniej korekt po pracy modelu.",
        fill=muted,
        font=("Segoe UI", 9),
        anchor=tk.W,
    )
    legend_x = 24
    for key, label, color in segments:
        canvas.create_rectangle(legend_x, 72, legend_x + 12, 84, fill=color, outline="")
        canvas.create_text(legend_x + 18, 78, text=label, fill=fg, font=("Segoe UI", 8), anchor=tk.W)
        legend_x += 118 if key != "plates_major_fix" else 124
    if not rows:
        canvas.create_text(24, 130, text="Brak wyników do wykresu.", fill=palette.get("warning", "#f0b44c"), anchor=tk.W)
        return
    for index, row in enumerate(rows[:10]):
        y = top + index * row_height
        label = textwrap.shorten(str(row.get("label", "-")), width=40, placeholder="...")
        corrected = int(row.get("total_corrected_plates", 0) or 0)
        detected = int(row.get("total_auto_plates", 0) or 0)
        total = max(1, sum(max(0, int(row.get(key, 0) or 0)) for key, _, _ in segments))
        x = left
        canvas.create_text(24, y + 22, text=f"#{index + 1} {label}", fill=fg, font=("Segoe UI", 10), anchor=tk.W)
        canvas.create_text(24, y + 44, text=f"wykryte: {detected} | odniesienie: {corrected}", fill=muted, font=("Segoe UI", 8), anchor=tk.W)
        canvas.create_rectangle(left, y + 8, left + bar_width, y + 32, fill=track, outline="", width=0)
        for key, _label, color in segments:
            value = max(0, int(row.get(key, 0) or 0))
            seg_width = (value / total) * bar_width
            if seg_width <= 0:
                continue
            canvas.create_rectangle(x, y + 8, x + seg_width, y + 32, fill=color, outline="", width=0)
            x += seg_width
        canvas.create_text(left + bar_width + 16, y + 23, text=f"{total} ramek", fill=fg, font=("Segoe UI", 9, "bold"), anchor=tk.W)


def _open_ranking_report_viewer(self):
    try:
        context = _collect_current_ranking_report_context(self)
    except Exception as exc:
        logger.error(f"Nie udało się przygotować przeglądarki raportu rankingu: {exc}")
        return messagebox.showerror("Przegląd raportu", f"Nie udało się przygotować raportu:\n{exc}")

    rows = list(context.get("rows") or [])
    if not rows:
        return messagebox.showinfo(
            "Przegląd raportu",
            "Brak ocenionych modeli dla aktualnego toru i zakresu. Najpierw uruchom ranking albo zmień tor testowy.",
        )

    existing = getattr(self, "_ranking_report_viewer_modal", None)
    try:
        if existing is not None and existing.winfo_exists():
            existing.destroy()
    except Exception:
        pass

    palette = getattr(self.app, "palette", {})
    dialog = tk.Toplevel(getattr(self, "frame", None))
    self._ranking_report_viewer_modal = dialog
    dialog.title("Przegląd raportu rankingu")
    dialog.configure(bg=palette.get("panel", "#252526"))
    dialog.resizable(True, True)
    try:
        dialog.transient(self.frame.winfo_toplevel())
    except Exception:
        pass

    def close_dialog():
        try:
            self._ranking_report_viewer_modal = None
        except Exception:
            pass
        try:
            dialog.destroy()
        except Exception:
            pass

    dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    shell = ttk.Frame(dialog, padding=12, style="Panel.TFrame")
    shell.pack(fill=tk.BOTH, expand=True)
    shell.grid_rowconfigure(2, weight=1)
    shell.grid_columnconfigure(0, weight=1)

    def _wheel_units(event) -> int:
        delta = int(getattr(event, "delta", 0) or 0)
        if delta:
            return -1 if delta > 0 else 1
        button = int(getattr(event, "num", 0) or 0)
        if button == 4:
            return -1
        if button == 5:
            return 1
        return 0

    def _bind_local_mousewheel(widget, scroll_target=None):
        target_widget = scroll_target or widget

        def _on_wheel(event):
            units = _wheel_units(event)
            if units:
                try:
                    target_widget.yview_scroll(units * 3, "units")
                except Exception:
                    pass
            return "break"

        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                widget.bind(sequence, _on_wheel)
            except Exception:
                pass
        return _on_wheel

    def _consume_modal_wheel(_event=None):
        return "break"

    for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        try:
            dialog.bind(sequence, _consume_modal_wheel)
        except Exception:
            pass

    winner = rows[0]
    ttk.Label(
        shell,
        text="Przegląd raportu rankingu",
        style="Panel.TLabel",
        anchor=tk.W,
    ).grid(row=0, column=0, sticky="ew")
    ttk.Label(
        shell,
        text=(
            f"Tor: {context.get('reference_name') or '-'} | "
            f"Zakres: {context.get('scope_label') or '-'} | "
            f"Wygrywa: {winner.get('label') or '-'} ({_ranking_report_percent_text(winner.get('score'))})"
        ),
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
    ).grid(row=1, column=0, sticky="ew", pady=(3, 10))

    notebook = ttk.Notebook(shell)
    notebook.grid(row=2, column=0, sticky="nsew")

    method_tab = ttk.Frame(notebook, padding=8, style="Panel.TFrame")
    results_tab = ttk.Frame(notebook, padding=8, style="Panel.TFrame")
    score_tab = ttk.Frame(notebook, padding=8, style="Panel.TFrame")
    metrics_tab = ttk.Frame(notebook, padding=8, style="Panel.TFrame")
    plate_diffs_tab = ttk.Frame(notebook, padding=8, style="Panel.TFrame")
    notebook.add(method_tab, text="Metoda")
    notebook.add(results_tab, text="Tabela wyników")
    notebook.add(score_tab, text="Wykres oceny")
    notebook.add(metrics_tab, text="Metryki")
    if str(context.get("target") or "") == "plate":
        notebook.add(plate_diffs_tab, text="Analiza tablic")

    method_tab.grid_rowconfigure(0, weight=1)
    method_tab.grid_columnconfigure(0, weight=1)
    text = tk.Text(
        method_tab,
        wrap=tk.WORD,
        bg=palette.get("input_bg", palette.get("panel_alt", "#1f1f1f")),
        fg=palette.get("fg", "#f3f3f3"),
        insertbackground=palette.get("fg", "#f3f3f3"),
        relief=tk.FLAT,
        borderwidth=0,
        padx=10,
        pady=10,
        font=("Segoe UI", 10),
    )
    method_scroll = WebSlimScrollbar(method_tab, orient=tk.VERTICAL, command=text.yview)
    text.configure(yscrollcommand=method_scroll.set)
    text.grid(row=0, column=0, sticky="nsew")
    method_scroll.grid(row=0, column=1, sticky="ns")
    text.insert("1.0", _ranking_report_markdown(context))
    text.configure(state=tk.DISABLED)
    _bind_local_mousewheel(text)
    _bind_local_mousewheel(method_tab, text)

    results_tab.grid_rowconfigure(0, weight=1)
    results_tab.grid_columnconfigure(0, weight=1)
    cols = ("rank", "model", "scope", "score", "precision", "recall", "f1", "map50_95", "sample", "evaluated")
    headings = {
        "rank": "#",
        "model": "Model",
        "scope": "Zakres",
        "score": "Ocena",
        "precision": "Precyzja",
        "recall": "Czułość",
        "f1": "F1",
        "map50_95": "mAP50-95",
        "sample": "Próbka",
        "evaluated": "Oceniono",
    }
    tree = ttk.Treeview(results_tab, columns=cols, show="headings")
    for col in cols:
        tree.heading(col, text=headings.get(col, col))
    tree.column("rank", width=58, anchor=tk.CENTER, stretch=False)
    tree.column("model", width=420, minwidth=260, anchor=tk.W, stretch=True)
    tree.column("scope", width=82, anchor=tk.CENTER, stretch=False)
    tree.column("score", width=90, anchor=tk.CENTER, stretch=False)
    tree.column("precision", width=90, anchor=tk.CENTER, stretch=False)
    tree.column("recall", width=90, anchor=tk.CENTER, stretch=False)
    tree.column("f1", width=80, anchor=tk.CENTER, stretch=False)
    tree.column("map50_95", width=92, anchor=tk.CENTER, stretch=False)
    tree.column("sample", width=82, anchor=tk.CENTER, stretch=False)
    tree.column("evaluated", width=148, anchor=tk.CENTER, stretch=False)
    try:
        tree.tag_configure("winner", background=blend_hex_colors(palette.get("success", "#2ecc71"), palette.get("panel", "#252526"), 0.84))
    except Exception:
        pass
    for row in rows:
        tree.insert(
            "",
            tk.END,
            values=(
                "WYGRANY" if int(row.get("rank", 0) or 0) == 1 else f"#{row.get('rank')}",
                row.get("label", "-"),
                row.get("scope", "-"),
                _ranking_report_percent_text(row.get("score")),
                _ranking_report_percent_text(row.get("precision")),
                _ranking_report_percent_text(row.get("recall")),
                _ranking_report_percent_text(row.get("f1")),
                _ranking_report_percent_text(row.get("map50_95")),
                row.get("sample", 0),
                row.get("evaluated_at", "-"),
            ),
            tags=("winner",) if int(row.get("rank", 0) or 0) == 1 else (),
        )
    yscroll = WebSlimScrollbar(results_tab, orient=tk.VERTICAL, command=tree.yview)
    xscroll = WebSlimScrollbar(results_tab, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    tree.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    _bind_local_mousewheel(tree)
    _bind_local_mousewheel(results_tab, tree)

    chart_specs = [
        (score_tab, _draw_ranking_score_canvas),
        (metrics_tab, _draw_ranking_metrics_canvas),
    ]
    if str(context.get("target") or "") == "plate":
        chart_specs.append((plate_diffs_tab, _draw_ranking_plate_diffs_canvas))

    for tab, drawer in chart_specs:
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        canvas = tk.Canvas(tab, highlightthickness=0)
        chart_scroll = WebSlimScrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=chart_scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        chart_scroll.grid(row=0, column=1, sticky="ns")
        _bind_local_mousewheel(canvas)
        _bind_local_mousewheel(tab, canvas)
        canvas.bind(
            "<Configure>",
            lambda _event, c=canvas, fn=drawer: fn(c, rows, palette),
            add="+",
        )
        try:
            dialog.after_idle(lambda c=canvas, fn=drawer: fn(c, rows, palette))
        except Exception:
            pass

    bottom = ttk.Frame(shell, style="Panel.TFrame")
    bottom.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    ttk.Button(
        bottom,
        text="[ RAPORT ] Eksportuj pakiet",
        command=self._export_ranking_analysis_report,
    ).pack(side=tk.LEFT)
    ttk.Button(bottom, text="Zamknij", command=close_dialog).pack(side=tk.RIGHT)

    try:
        dialog.update_idletasks()
        root = self.frame.winfo_toplevel()
        width = min(max(1120, int(root.winfo_width() * 0.9)), 1500)
        height = min(max(720, int(root.winfo_height() * 0.84)), 980)
        x = int(root.winfo_rootx() + max(0, (root.winfo_width() - width) // 2))
        y = int(root.winfo_rooty() + max(0, (root.winfo_height() - height) // 2))
        dialog.geometry(f"{width}x{height}+{x}+{y}")
    except Exception:
        dialog.geometry("1180x760")
    try:
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass


def _export_ranking_analysis_report(self):
    try:
        context = _collect_current_ranking_report_context(self)
    except Exception as exc:
        logger.error(f"Nie udało się przygotować danych raportu rankingu: {exc}")
        return messagebox.showerror("Raport rankingu", f"Nie udało się przygotować danych raportu:\n{exc}")

    rows = list(context.get("rows") or [])
    if not rows:
        return messagebox.showinfo(
            "Raport rankingu",
            "Brak ocenionych modeli dla aktualnego toru i zakresu. Najpierw uruchom ranking albo zmień tor testowy.",
        )

    try:
        base_dir = Path(getattr(getattr(self, "ranking_engine", None), "ranking_dir", CONFIG.get_ranking_dir(context.get("target"))))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = base_dir / "reports" / f"ranking_report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        csv_path = report_dir / "ranking_results.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(
                [
                    "pozycja",
                    "model",
                    "plik_modelu",
                    "zakres",
                    "ocena_pct",
                    "precyzja_pct",
                    "czulosc_pct",
                    "f1_pct",
                    "dokladnosc_pct",
                    "map50_pct",
                    "map50_95_pct",
                    "probka",
                    "wykrycia_modelu",
                    "anotacje_odniesienia",
                    "bez_zmian",
                    "male_poprawki",
                    "duze_poprawki",
                    "dodane_w_odniesieniu",
                    "usuniete_z_modelu",
                    "tor",
                    "split",
                    "zrodlo_metryk",
                    "oceniono",
                    "sciezka_modelu",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row.get("rank", ""),
                        row.get("label", ""),
                        row.get("model_file", ""),
                        row.get("scope", ""),
                        f"{float(row.get('score', 0) or 0):.4f}",
                        f"{float(row.get('precision', 0) or 0):.4f}",
                        f"{float(row.get('recall', 0) or 0):.4f}",
                        f"{float(row.get('f1', 0) or 0):.4f}",
                        f"{float(row.get('accuracy', 0) or 0):.4f}",
                        f"{float(row.get('map50', 0) or 0):.4f}",
                        f"{float(row.get('map50_95', 0) or 0):.4f}",
                        row.get("sample", 0),
                        row.get("total_auto_plates", 0),
                        row.get("total_corrected_plates", 0),
                        row.get("plates_unchanged", 0),
                        row.get("plates_minor_fix", 0),
                        row.get("plates_major_fix", 0),
                        row.get("plates_added", 0),
                        row.get("plates_removed", 0),
                        row.get("reference", ""),
                        row.get("split", ""),
                        row.get("metrics_source", ""),
                        row.get("evaluated_at", ""),
                        row.get("model_path", ""),
                    ]
                )

        (report_dir / "ranking_score.svg").write_text(
            _ranking_report_bar_svg(rows, title="Ranking modeli - ocena"),
            encoding="utf-8",
        )
        (report_dir / "ranking_metrics.svg").write_text(
            _ranking_report_metrics_svg(rows, title="Ranking modeli - metryki pomocnicze"),
            encoding="utf-8",
        )
        if str(context.get("target") or "") == "plate":
            (report_dir / "ranking_plate_diffs.svg").write_text(
                _ranking_report_plate_diffs_svg(rows, title="Ranking modeli tablic - zgodność detekcji"),
                encoding="utf-8",
            )
        (report_dir / "ranking_report.md").write_text(
            _ranking_report_markdown(context),
            encoding="utf-8",
        )
        logger.info(f"Zapisano raport rankingu modeli: {report_dir}")
        if messagebox.askyesno(
            "Raport rankingu",
            f"Zapisano raport rankingu:\n{report_dir}\n\nOtworzyć folder raportu?",
        ):
            try:
                self._open_path(report_dir)
            except Exception:
                pass
    except Exception as exc:
        logger.error(f"Nie udało się zapisać raportu rankingu: {exc}")
        return messagebox.showerror("Raport rankingu", f"Nie udało się zapisać raportu:\n{exc}")


def _count_ranking_dataset_splits(yaml_path: Path) -> dict[str, int]:
    try:
        cfg = safe_load_yaml(yaml_path) or {}
    except Exception:
        cfg = {}

    def dataset_root() -> Path:
        raw_root = str(cfg.get("path") or "").strip() if isinstance(cfg, dict) else ""
        if not raw_root:
            return yaml_path.parent
        root_path = Path(raw_root)
        return root_path if root_path.is_absolute() else yaml_path.parent / root_path

    def count_split(split_name: str) -> int:
        if not isinstance(cfg, dict):
            return 0
        split_value = cfg.get(split_name)
        if not split_value:
            return 0
        root = dataset_root()
        values = split_value if isinstance(split_value, list) else [split_value]
        total = 0
        for item in values:
            raw_item = str(item or "").strip()
            if not raw_item:
                continue
            split_path = Path(raw_item)
            if not split_path.is_absolute():
                split_path = root / split_path
            try:
                if split_path.is_file() and split_path.suffix.lower() == ".txt":
                    total += sum(
                        1
                        for line in split_path.read_text(encoding="utf-8-sig").splitlines()
                        if line.strip()
                    )
                elif split_path.is_dir():
                    total += len(get_image_files(split_path))
                elif split_path.is_file() and split_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                    total += 1
            except Exception:
                continue
        return total

    counts = {split: count_split(split) for split in ("train", "val", "test")}
    counts["total"] = sum(counts.values())
    return counts


def _collect_ranking_track_candidates(self) -> list[dict]:
    target = self._get_ranking_task_target()
    selected_split = self._get_ranking_split_name()
    candidates: list[dict] = []
    seen: set[str] = set()

    def add_candidate(path_like, *, source: str) -> None:
        raw = str(path_like or "").strip()
        if not raw:
            return
        try:
            path = Path(raw)
        except Exception:
            return
        if path.name.lower() == "data.yaml":
            path = path.parent
        key = _ranking_path_key(path)
        if not key or key in seen:
            return
        seen.add(key)

        if target == "char":
            yaml_path = path / "data.yaml"
            if not yaml_path.exists():
                return
            try:
                inferred = self._infer_dataset_target(str(yaml_path))
            except Exception:
                inferred = "char"
            if inferred and inferred != "char":
                return
            counts = _count_ranking_dataset_splits(yaml_path)
            if int(counts.get("val", 0) or 0) <= 0 and int(counts.get("test", 0) or 0) <= 0:
                return
            ref = build_dataset_display_ref(path, target_hint="char", counts=counts)
            split_count = int(counts.get(selected_split, 0) or 0)
            candidates.append(
                {
                    "path": str(path),
                    "id": ref.id,
                    "type": "znaki",
                    "split": selected_split,
                    "split_count": split_count,
                    "counts": counts,
                    "created": ref.created_label or "-",
                    "source": source,
                    "status": "gotowy" if split_count > 0 else f"brak splitu {selected_split}",
                    "ready": split_count > 0,
                    "details": ref.split_label,
                }
            )
            return

        info = self._resolve_ranking_reference_source(str(path))
        if not info.get("ok"):
            return
        try:
            created = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except Exception:
            created = "-"
        try:
            run_ref = build_run_display_ref(path, kind_hint="annotation")
            track_id = run_ref.id
        except Exception:
            track_id = path.name
        image_count = int(info.get("image_count", 0) or 0)
        candidates.append(
            {
                "path": str(path),
                "id": track_id,
                "type": "tablice",
                "split": "run",
                "split_count": image_count,
                "counts": {"total": image_count},
                "created": created,
                "source": source,
                "status": "gotowy" if image_count > 0 else "brak obrazów",
                "ready": image_count > 0,
                "details": f"obrazy {image_count}",
            }
        )

    current_value = str(getattr(getattr(self, "rank_data_dir", None), "get", lambda: "")() or "").strip()
    add_candidate(current_value, source="Aktualny")

    roots: list[tuple[Path, str]] = []
    try:
        project_root = CAMPAIGN.get_active_project_root_dir()
        if project_root is not None:
            project_root = Path(project_root)
            if target == "char":
                roots.append((project_root / "4_training_datasets" / "chars", "Projekt"))
                roots.append((project_root / "4_training_datasets", "Projekt"))
            else:
                auto_dir = CAMPAIGN.get_dir("auto_ann")
                if auto_dir is not None:
                    roots.append((Path(auto_dir), "Projekt"))
    except Exception:
        pass

    try:
        if target == "char":
            roots.append((Path(CONFIG.get_datasets_dir("char")), "Globalne"))
        else:
            roots.append((Path(CONFIG.get_auto_annotations_dir("plate")), "Globalne"))
    except Exception:
        pass

    if target == "char":
        try:
            dataset_yaml = self._resolve_training_dataset_yaml_path()
            if dataset_yaml is not None:
                add_candidate(Path(dataset_yaml).parent, source="Aktualny wariant")
        except Exception:
            pass
        for root, source in roots:
            try:
                if root.exists():
                    for yaml_path in sorted(root.rglob("data.yaml")):
                        add_candidate(yaml_path.parent, source=source)
            except Exception:
                continue
    else:
        try:
            stored = CAMPAIGN.get_last_plate_training_source()
            source_run = str((stored or {}).get("source_run_path") or "").strip()
            if source_run:
                add_candidate(source_run, source="Aktualny wariant")
        except Exception:
            pass
        for root, source in roots:
            try:
                if root.exists():
                    for xml_path in sorted(root.rglob("annotations.xml")):
                        add_candidate(xml_path.parent, source=source)
            except Exception:
                continue

    candidates.sort(
        key=lambda row: (
            0 if row.get("source") in {"Aktualny", "Aktualny wariant"} else 1,
            0 if row.get("ready") else 1,
            str(row.get("created") or ""),
            str(row.get("id") or ""),
        ),
        reverse=False,
    )
    return candidates


def _open_ranking_track_modal(self):
    existing = getattr(self, "_ranking_track_modal", None)
    try:
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
    except Exception:
        pass

    palette = getattr(self.app, "palette", {})
    dialog = tk.Toplevel(getattr(self, "frame", None))
    self._ranking_track_modal = dialog
    dialog.title("Wybór toru testowego rankingu")
    dialog.configure(bg=palette.get("panel", "#252526"))
    dialog.resizable(True, True)
    try:
        dialog.transient(self.frame.winfo_toplevel())
    except Exception:
        pass

    def close_dialog():
        try:
            self._ranking_track_modal = None
        except Exception:
            pass
        try:
            dialog.destroy()
        except Exception:
            pass

    dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    shell = ttk.Frame(dialog, padding=14, style="Panel.TFrame")
    shell.pack(fill=tk.BOTH, expand=True)
    shell.grid_columnconfigure(0, weight=1)
    shell.grid_rowconfigure(3, weight=1)

    ttk.Label(
        shell,
        text="Tor testowy rankingu",
        style="Panel.TLabel",
        anchor=tk.W,
    ).grid(row=0, column=0, sticky="ew")
    ttk.Label(
        shell,
        text=(
            "Wybierz jeden wspólny materiał testowy. Wszystkie modele pobiegną po tym samym torze, "
            "więc wynik będzie porównywalny."
        ),
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=760,
    ).grid(row=1, column=0, sticky="ew", pady=(2, 10))

    toolbar = ttk.Frame(shell, style="Panel.TFrame")
    toolbar.grid(row=2, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(
        toolbar,
        text=f"Konie: {self._get_ranking_task_label()}",
        style="PanelMuted.TLabel",
    ).pack(side=tk.LEFT, padx=(0, 12))
    split_combo = None
    if self._get_ranking_task_target() == "char":
        ttk.Label(toolbar, text="Split toru:", style="PanelMuted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        split_combo = ttk.Combobox(
            toolbar,
            textvariable=self.rank_split_var,
            values=("test", "val"),
            state="readonly",
            width=8,
        )
        split_combo.pack(side=tk.LEFT)
    ttk.Button(toolbar, text="Odśwież", command=lambda: load_rows()).pack(side=tk.RIGHT)

    table_frame = ttk.Frame(shell, style="Panel.TFrame")
    table_frame.grid(row=3, column=0, sticky="nsew")
    table_frame.rowconfigure(0, weight=1)
    table_frame.columnconfigure(0, weight=1)

    cols = ("status", "track", "type", "split", "count", "created", "source")
    tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
    headings = {
        "status": "Status",
        "track": "Tor",
        "type": "Typ",
        "split": "Split",
        "count": "Materiał",
        "created": "Utworzono",
        "source": "Źródło",
    }
    for col, text in headings.items():
        tree.heading(col, text=text)
    tree.column("status", width=82, minwidth=70, anchor=tk.CENTER, stretch=False)
    tree.column("track", width=240, minwidth=170, anchor=tk.W, stretch=True)
    tree.column("type", width=78, minwidth=68, anchor=tk.CENTER, stretch=False)
    tree.column("split", width=72, minwidth=62, anchor=tk.CENTER, stretch=False)
    tree.column("count", width=260, minwidth=180, anchor=tk.W, stretch=True)
    tree.column("created", width=120, minwidth=104, anchor=tk.W, stretch=False)
    tree.column("source", width=92, minwidth=80, anchor=tk.CENTER, stretch=False)
    try:
        tree.tag_configure("ready", foreground=palette.get("success", "#2ecc71"))
        tree.tag_configure("blocked", foreground=palette.get("error", "#e05d5d"))
        tree.tag_configure("current", background=blend_hex_colors(
            palette.get("accent", "#0e639c"),
            palette.get("panel", "#252526"),
            0.82,
        ))
    except Exception:
        pass

    yscroll = WebSlimScrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=yscroll.set)
    tree.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")

    status_lbl = ttk.Label(
        shell,
        text="Zaznacz wiersz i zastosuj go jako tor testowy rankingu.",
        style="PanelMuted.TLabel",
        anchor=tk.W,
    )
    status_lbl.grid(row=4, column=0, sticky="ew", pady=(8, 0))

    rows_by_id: dict[str, dict] = {}

    def load_rows():
        selected_path = str(getattr(getattr(self, "rank_data_dir", None), "get", lambda: "")() or "").strip()
        selected_key = _ranking_path_key(selected_path)
        rows_by_id.clear()
        try:
            tree.delete(*tree.get_children())
        except Exception:
            pass
        participant_paths_by_item.clear()
        rows = _collect_ranking_track_candidates(self)
        for index, row in enumerate(rows):
            path_key = _ranking_path_key(row.get("path"))
            tags = []
            tags.append("ready" if row.get("ready") else "blocked")
            if selected_key and path_key == selected_key:
                tags.append("current")
            item_id = tree.insert("", tk.END, values=(
                "jest" if row.get("ready") else "-",
                row.get("id") or "-",
                row.get("type") or "-",
                str(row.get("split") or "-"),
                row.get("details") or "-",
                row.get("created") or "-",
                row.get("source") or "-",
            ), tags=tuple(tags))
            rows_by_id[item_id] = row
            if index == 0 and not selected_key:
                try:
                    tree.selection_set(item_id)
                    tree.focus(item_id)
                except Exception:
                    pass
            elif selected_key and path_key == selected_key:
                try:
                    tree.selection_set(item_id)
                    tree.focus(item_id)
                    tree.see(item_id)
                except Exception:
                    pass
        if rows:
            status_lbl.configure(text=f"Dostępne tory: {len(rows)}. Modele będą porównane tylko na zaznaczonym torze.")
        else:
            status_lbl.configure(text="Nie znaleziono gotowych torów testowych. Użyj Zaawansowane, jeśli musisz wskazać ścieżkę ręcznie.")

    def accept_selection():
        selected = tree.selection()
        if not selected:
            return messagebox.showwarning("Brak wyboru", "Zaznacz tor testowy w tabeli.")
        row = rows_by_id.get(selected[0]) or {}
        if not row.get("ready"):
            return messagebox.showwarning(
                "Tor nie jest gotowy",
                "Ten tor nie ma materiału dla wybranego splitu. Wybierz inny tor albo zmień split.",
            )
        path = str(row.get("path") or "").strip()
        if not path:
            return
        try:
            self.rank_data_dir.set(path)
        except Exception:
            pass
        try:
            self._refresh_ranking_reference_ui()
        except Exception:
            pass
        close_dialog()

    tree.bind("<Double-1>", lambda _event: accept_selection(), add="+")
    if split_combo is not None:
        try:
            split_combo.bind("<<ComboboxSelected>>", lambda _event: load_rows(), add="+")
        except Exception:
            pass

    bottom = ttk.Frame(shell, style="Panel.TFrame")
    bottom.grid(row=5, column=0, sticky="ew", pady=(12, 0))
    ttk.Button(
        bottom,
        text="[ OPCJE ] Zaawansowane",
        command=self._open_ranking_advanced_modal,
    ).pack(side=tk.LEFT)
    ttk.Button(bottom, text="[ X ] Zamknij", command=close_dialog).pack(side=tk.RIGHT)
    ttk.Button(
        bottom,
        text="[ TOR ] Zastosuj zaznaczony tor",
        style="Accent.TButton",
        command=accept_selection,
    ).pack(side=tk.RIGHT, padx=(0, 8))

    load_rows()
    try:
        dialog.update_idletasks()
        root = self.frame.winfo_toplevel()
        width = min(max(980, int(root.winfo_width() * 0.78)), 1280)
        height = min(max(560, int(root.winfo_height() * 0.68)), 820)
        x = int(root.winfo_rootx() + max(0, (root.winfo_width() - width) // 2))
        y = int(root.winfo_rooty() + max(0, (root.winfo_height() - height) // 2))
        dialog.geometry(f"{width}x{height}+{x}+{y}")
    except Exception:
        dialog.geometry("1040x620")
    try:
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass


def _open_ranking_participants_modal(self):
    existing = getattr(self, "_ranking_participants_modal", None)
    try:
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
    except Exception:
        pass

    palette = getattr(self.app, "palette", {})
    if not hasattr(self, "rank_scope_var"):
        default_scope = "Projekt" if CAMPAIGN.get_active_project_name() else "Wszystkie"
        self.rank_scope_var = tk.StringVar(value=default_scope)

    dialog = tk.Toplevel(getattr(self, "frame", None))
    self._ranking_participants_modal = dialog
    dialog.title("Uczestnicy rankingu modeli")
    dialog.configure(bg=palette.get("panel", "#252526"))
    dialog.resizable(True, True)
    try:
        dialog.transient(self.frame.winfo_toplevel())
    except Exception:
        pass

    def close_dialog():
        try:
            self._ranking_participants_modal = None
        except Exception:
            pass
        try:
            dialog.destroy()
        except Exception:
            pass

    dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    shell = ttk.Frame(dialog, padding=14, style="Panel.TFrame")
    shell.pack(fill=tk.BOTH, expand=True)
    shell.grid_columnconfigure(0, weight=1)
    shell.grid_rowconfigure(3, weight=1)

    ttk.Label(
        shell,
        text="Konie rankingu",
        style="Panel.TLabel",
        anchor=tk.W,
    ).grid(row=0, column=0, sticky="ew")
    intro = ttk.Label(
        shell,
        text=(
            "Tu widać dokładnie, które modele wystartują w wyścigu. Zakres zmienia listę uczestników "
            "i tę samą listę dostaje potem ranking."
        ),
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=820,
    )
    intro.grid(row=1, column=0, sticky="ew", pady=(2, 10))

    toolbar = ttk.Frame(shell, style="Panel.TFrame")
    toolbar.grid(row=2, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(toolbar, text="Zakres:", style="PanelMuted.TLabel").pack(side=tk.LEFT, padx=(0, 8))

    def refresh_after_scope_change():
        try:
            self._refresh_ranking_reference_ui()
        except Exception:
            pass
        try:
            self._load_ranking()
        except Exception:
            pass
        load_rows()

    for label, value in (
        ("Projektowe", "Projekt"),
        ("Globalne", "Globalne"),
        ("Wszystkie", "Wszystkie"),
    ):
        ttk.Radiobutton(
            toolbar,
            text=label,
            value=value,
            variable=self.rank_scope_var,
            command=refresh_after_scope_change,
        ).pack(side=tk.LEFT, padx=(0, 12))

    ttk.Button(toolbar, text="Odśwież", command=lambda: load_rows()).pack(side=tk.RIGHT)

    def choose_global_models_dir():
        initial = str(getattr(getattr(self, "rank_models_dir", None), "get", lambda: "")() or "")
        dialog_kwargs = {
            "title": "Wskaż katalog modeli globalnych",
            "parent": dialog,
        }
        if initial and Path(initial).exists():
            dialog_kwargs["initialdir"] = initial
        selected = filedialog.askdirectory(**dialog_kwargs)
        if not selected:
            return
        try:
            self.rank_models_dir.set(selected)
            self.rank_scope_var.set("Wszystkie")
        except Exception:
            pass
        refresh_after_scope_change()

    table_frame = ttk.Frame(shell, style="Panel.TFrame")
    table_frame.grid(row=3, column=0, sticky="nsew")
    table_frame.rowconfigure(0, weight=1)
    table_frame.columnconfigure(0, weight=1)

    participants_tree_style = "RankingParticipants.Treeview"
    try:
        style = ttk.Style()
        style.configure(participants_tree_style, font=("Segoe UI", 9, "bold"), rowheight=24)
        style.configure(f"{participants_tree_style}.Heading", font=("Segoe UI", 9, "bold"))
    except Exception:
        pass

    def make_start_icon(enabled: bool) -> tk.PhotoImage:
        icon = tk.PhotoImage(width=18, height=18)
        color = palette.get("success", "#2ecc71") if enabled else palette.get("danger", "#e74c3c")
        if enabled:
            icon.put(color, to=(8, 3, 11, 15))
            icon.put(color, to=(3, 8, 16, 11))
        else:
            icon.put(color, to=(3, 8, 16, 11))
        return icon

    start_icons = {
        "on": make_start_icon(True),
        "off": make_start_icon(False),
    }
    self._ranking_participants_start_icons = start_icons

    cols = ("scope", "participant", "family", "size", "created", "epochs", "map", "train", "file", "source")
    tree = ttk.Treeview(
        table_frame,
        columns=cols,
        show="tree headings",
        selectmode="browse",
        style=participants_tree_style,
    )
    participant_sort_values: dict[str, dict] = {}
    sort_state = {"column": "", "descending": False}

    def sort_value(item_id: str, column: str):
        data = participant_sort_values.get(str(item_id), {})
        value = data.get(column)
        if value is None:
            return (2, "")
        if isinstance(value, bool):
            return (0, 0 if value else 1)
        if isinstance(value, (int, float)):
            return (0, float(value))
        text = str(value or "").strip().lower()
        if not text or text == "-":
            return (2, "")
        return (1, text)

    def sort_by_column(column: str) -> None:
        descending = False
        if sort_state.get("column") == column:
            descending = not bool(sort_state.get("descending"))
        sort_state["column"] = column
        sort_state["descending"] = descending
        try:
            items = list(tree.get_children(""))
            items.sort(key=lambda item_id: sort_value(str(item_id), column), reverse=descending)
            for index, item_id in enumerate(items):
                tree.move(item_id, "", index)
            refresh_sort_headings()
        except Exception:
            pass

    headings = {
        "scope": "Zakres",
        "participant": "Uczestnik",
        "family": "Rodzina",
        "size": "MB",
        "created": "Utworzono",
        "epochs": "Epoki",
        "map": "mAP50-95",
        "train": "Obrazy train",
        "file": "Wagi",
        "source": "Źródło modelu",
    }

    def refresh_sort_headings() -> None:
        active = str(sort_state.get("column") or "")
        arrow = " ↓" if bool(sort_state.get("descending")) else " ↑"
        tree.heading("#0", text=f"Start{arrow if active == 'start' else ''}", command=lambda: sort_by_column("start"))
        for column, text in headings.items():
            tree.heading(
                column,
                text=f"{text}{arrow if active == column else ''}",
                command=lambda c=column: sort_by_column(c),
            )

    refresh_sort_headings()
    tree.column("#0", width=88, minwidth=76, anchor=tk.CENTER, stretch=False)
    tree.column("scope", width=132, minwidth=96, anchor=tk.W, stretch=False)
    tree.column("participant", width=170, minwidth=140, anchor=tk.W, stretch=False)
    tree.column("family", width=92, minwidth=74, anchor=tk.CENTER, stretch=False)
    tree.column("size", width=64, minwidth=54, anchor=tk.CENTER, stretch=False)
    tree.column("created", width=124, minwidth=108, anchor=tk.CENTER, stretch=False)
    tree.column("epochs", width=76, minwidth=64, anchor=tk.CENTER, stretch=False)
    tree.column("map", width=88, minwidth=76, anchor=tk.CENTER, stretch=False)
    tree.column("train", width=94, minwidth=80, anchor=tk.CENTER, stretch=False)
    tree.column("file", width=136, minwidth=106, anchor=tk.W, stretch=False)
    tree.column("source", width=330, minwidth=240, anchor=tk.W, stretch=True)
    try:
        tree.tag_configure("project", foreground=palette.get("success", "#2ecc71"))
        tree.tag_configure("global", foreground=palette.get("fg", "#f3f3f3"))
        tree.tag_configure("disabled", foreground=palette.get("muted", "#8f8f8f"))
    except Exception:
        pass

    yscroll = WebSlimScrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    xscroll = WebSlimScrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    tree.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")

    status_lbl = ttk.Label(
        shell,
        text="Ładuję uczestników...",
        style="PanelMuted.TLabel",
        anchor=tk.W,
    )
    status_lbl.grid(row=4, column=0, sticky="ew", pady=(8, 0))
    participant_paths_by_item: dict[str, Path] = {}

    def scope_cell_label(candidate_scope: str) -> str:
        if candidate_scope == "Projekt":
            try:
                return str(CAMPAIGN.get_active_project_name() or "").strip() or "Projekt"
            except Exception:
                return "Projekt"
        return "Globalne"

    def model_family_label(path: Path, run) -> str:
        raw = ""
        if run is not None:
            raw = str(getattr(run, "base_model", "") or "").strip()
        if not raw:
            raw = str(path.name or "").strip()
        stem = Path(raw).stem if raw else ""
        lower = stem.lower()
        for suffix in ("-pose", "_pose", "-detect", "_detect", "-seg", "_seg", "-cls", "_cls"):
            if lower.endswith(suffix):
                stem = stem[: -len(suffix)]
                lower = lower[: -len(suffix)]
                break
        match = re.search(r"(yolo(?:v)?\d+[nslmx]?)", lower)
        if match:
            return match.group(1)
        return stem[:18] if stem else "-"

    def model_size_label(path: Path) -> str:
        try:
            size_mb = float(Path(path).stat().st_size) / (1024.0 * 1024.0)
            return f"{size_mb:.1f}"
        except Exception:
            return "-"

    def format_timestamp_label(timestamp: float | None) -> str:
        if not timestamp:
            return "-"
        try:
            return datetime.datetime.fromtimestamp(float(timestamp)).strftime("%d.%m.%y %H:%M")
        except Exception:
            return "-"

    def model_created_info(path: Path, run) -> tuple[str, float]:
        raw = ""
        if run is not None:
            raw = str(getattr(run, "created_at", "") or getattr(run, "started_at", "") or "").strip()
        if raw:
            try:
                dt = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return dt.strftime("%d.%m.%y %H:%M"), float(dt.timestamp())
            except Exception:
                pass
        try:
            timestamp = float(Path(path).stat().st_ctime)
            return format_timestamp_label(timestamp), timestamp
        except Exception:
            return "-", 0.0

    def model_epochs_info(run) -> tuple[str, int]:
        if run is None:
            return "-", -1
        current = int(getattr(run, "current_epoch", 0) or 0)
        target = int(getattr(run, "epochs", 0) or 0)
        if target <= 0 and current <= 0:
            return "-", -1
        if current > 0 and target > 0:
            return f"{current}/{target}", current
        return str(target or current), target or current

    def model_map_info(run) -> tuple[str, float]:
        if run is None:
            return "-", -1.0
        value = float(getattr(run, "best_map50_95", 0.0) or 0.0)
        if value <= 0:
            return "-", -1.0
        percent = value * 100.0 if 0.0 < value <= 1.0 else value
        return f"{percent:.1f}%", percent

    dataset_train_count_cache: dict[str, tuple[str, int]] = {}

    def dataset_train_images_info(run) -> tuple[str, int]:
        raw = str(getattr(run, "dataset_path", "") or "").strip() if run is not None else ""
        if not raw:
            return "-", -1
        try:
            dataset_path = Path(raw)
        except Exception:
            return "-", -1
        yaml_path = dataset_path if dataset_path.name.lower() == "data.yaml" else dataset_path / "data.yaml"
        key = _ranking_path_key(yaml_path)
        if key in dataset_train_count_cache:
            return dataset_train_count_cache[key]
        if not yaml_path.exists():
            result = ("-", -1)
        else:
            try:
                counts = _count_ranking_dataset_splits(yaml_path)
                train_count = int((counts or {}).get("train", 0) or 0)
                result = (str(train_count) if train_count > 0 else "-", train_count if train_count > 0 else -1)
            except Exception:
                result = ("-", -1)
        dataset_train_count_cache[key] = result
        return result

    def participant_tags(candidate_scope: str, enabled: bool) -> tuple[str, ...]:
        if not enabled:
            return ("disabled",)
        return ("project",) if candidate_scope == "Projekt" else ("global",)

    def update_start_cell(item_id: str, enabled: bool) -> None:
        try:
            tree.item(
                item_id,
                text="TAK" if enabled else "NIE",
                image=start_icons["on" if enabled else "off"],
            )
        except Exception:
            pass

    def update_status_label(total: int) -> None:
        selected = sum(
            1 for path in participant_paths_by_item.values()
            if _is_ranking_participant_enabled(self, path)
        )
        scope = _get_ranking_scope(self)
        target = self._get_ranking_task_target()
        scope_label = _format_ranking_scope_label(self, scope, target)
        status_lbl.configure(
            text=(
                f"Zakres: {scope_label}. Startuje: {selected}/{total}. "
                "Tylko zaznaczone modele trafią do rankingu."
            )
        )

    def toggle_participant_item(item_id: str) -> None:
        path = participant_paths_by_item.get(str(item_id))
        if path is None:
            return
        enabled = not _is_ranking_participant_enabled(self, path)
        _set_ranking_participant_enabled(self, path, enabled)
        update_start_cell(item_id, enabled)
        try:
            sort_data = participant_sort_values.get(str(item_id), {})
            sort_data["start"] = bool(enabled)
            tree.item(item_id, tags=participant_tags(str(sort_data.get("scope_key") or ""), enabled))
        except Exception:
            pass
        update_status_label(len(participant_paths_by_item))
        try:
            self._refresh_ranking_reference_ui()
        except Exception:
            pass
        try:
            self._load_ranking()
        except Exception:
            pass

    def on_tree_click(event):
        try:
            if tree.identify_column(event.x) != "#0":
                return None
            item_id = str(tree.identify_row(event.y) or "")
        except Exception:
            item_id = ""
        if item_id:
            toggle_participant_item(item_id)
            return "break"
        return None

    def on_tree_space(_event=None):
        try:
            selection = list(tree.selection() or [])
        except Exception:
            selection = []
        if selection:
            toggle_participant_item(str(selection[0]))
            return "break"
        return None

    def on_tree_motion(event):
        try:
            cursor = "hand2" if tree.identify_column(event.x) == "#0" and tree.identify_row(event.y) else ""
            if str(tree.cget("cursor") or "") != cursor:
                tree.configure(cursor=cursor)
        except Exception:
            pass

    def on_tree_leave(_event=None):
        try:
            tree.configure(cursor="")
        except Exception:
            pass

    tree.bind("<Button-1>", on_tree_click, add="+")
    tree.bind("<space>", on_tree_space, add="+")
    tree.bind("<Motion>", on_tree_motion, add="+")
    tree.bind("<Leave>", on_tree_leave, add="+")

    def model_run_for_path(path: Path):
        resolver = getattr(self, "_resolve_training_run_from_model_path", None)
        if callable(resolver):
            try:
                return resolver(Path(path))
            except Exception:
                return None
        return None

    def model_labels(path: Path, scope: str) -> tuple[str, str]:
        run = model_run_for_path(path)
        target = self._get_ranking_task_target()
        if run is not None:
            try:
                run_ref = build_run_display_ref(run, kind_hint="training")
                model_ref = build_model_display_ref(
                    path,
                    run=run,
                    target_hint=target,
                    source_run_label=run_ref.id,
                )
                return model_ref.id, f"Run {run_ref.id}"
            except Exception:
                run_id = str(getattr(run, "id", "") or "").strip()
                try:
                    target_task = self._get_ranking_task_label()
                    return format_ranking_model_label(path.name, str(path), target_task), f"Run {run_id or '-'}"
                except Exception:
                    pass
        try:
            label = build_model_display_ref(path, target_hint=target).id
        except Exception:
            target_task = self._get_ranking_task_label()
            label = format_ranking_model_label(path.name, str(path), target_task)
        return label, "Projekt" if scope == "Projekt" else "Katalog modeli"

    def load_rows():
        try:
            tree.delete(*tree.get_children())
        except Exception:
            pass
        participant_paths_by_item.clear()
        participant_sort_values.clear()
        target = self._get_ranking_task_target()
        scope = _get_ranking_scope(self)
        models_dir_raw = str(getattr(getattr(self, "rank_models_dir", None), "get", lambda: "")() or "").strip()
        try:
            models_dir = Path(models_dir_raw) if models_dir_raw else None
        except Exception:
            models_dir = None
        try:
            participants = _collect_ranking_participant_candidates(self, models_dir, target, scope)
        except Exception:
            participants = []
        for path in participants:
            model_path = Path(path)
            candidate_scope = _ranking_model_candidate_scope(self, model_path, target)
            label, source = model_labels(model_path, candidate_scope)
            run = model_run_for_path(model_path)
            enabled = _is_ranking_participant_enabled(self, model_path)
            created_label, created_sort = model_created_info(model_path, run)
            epochs_label, epochs_sort = model_epochs_info(run)
            map_label, map_sort = model_map_info(run)
            train_label, train_sort = dataset_train_images_info(run)
            size_label = model_size_label(model_path)
            try:
                size_sort = float(size_label)
            except Exception:
                size_sort = -1.0
            scope_label = scope_cell_label(candidate_scope)
            family_label = model_family_label(model_path, run)
            item_id = tree.insert("", tk.END, values=(
                scope_cell_label(candidate_scope),
                label,
                family_label,
                size_label,
                created_label,
                epochs_label,
                map_label,
                train_label,
                model_path.name,
                source,
            ), tags=participant_tags(candidate_scope, enabled))
            update_start_cell(str(item_id), enabled)
            participant_paths_by_item[str(item_id)] = model_path
            participant_sort_values[str(item_id)] = {
                "start": bool(enabled),
                "scope": scope_label,
                "scope_key": candidate_scope,
                "participant": label,
                "family": family_label,
                "size": size_sort,
                "created": created_sort,
                "epochs": epochs_sort,
                "map": map_sort,
                "train": train_sort,
                "file": model_path.name,
                "source": source,
            }
        update_status_label(len(participants))

    bottom = ttk.Frame(shell, style="Panel.TFrame")
    bottom.grid(row=5, column=0, sticky="ew", pady=(12, 0))
    ttk.Button(
        bottom,
        text="[ DODAJ ] Wskaż katalog modeli",
        command=choose_global_models_dir,
    ).pack(side=tk.LEFT)
    ttk.Button(bottom, text="[ X ] Zamknij", command=close_dialog).pack(side=tk.RIGHT)

    load_rows()
    try:
        dialog.update_idletasks()
        root = self.frame.winfo_toplevel()
        width = min(max(1320, int(root.winfo_width() * 0.9)), 1600)
        height = min(max(580, int(root.winfo_height() * 0.7)), 860)
        x = int(root.winfo_rootx() + max(0, (root.winfo_width() - width) // 2))
        y = int(root.winfo_rooty() + max(0, (root.winfo_height() - height) // 2))
        dialog.geometry(f"{width}x{height}+{x}+{y}")
    except Exception:
        dialog.geometry("1360x680")
    try:
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass

def _on_analysis_plot_selected(self, event=None):
    listbox = getattr(self, "_analysis_plots_list", None)
    if listbox is None or not self._analysis_plot_paths:
        return
    selection = listbox.curselection()
    if not selection:
        return
    index = int(selection[0])
    if 0 <= index < len(self._analysis_plot_paths):
        self._show_analysis_plot(self._analysis_plot_paths[index])

def _show_analysis_plot(self, path):
    if not PIL_AVAILABLE:
        return
    try:
        title, description = self._analysis_plot_info(path)
        title_lbl = getattr(self, "_analysis_plot_title_lbl", None)
        hint_lbl = getattr(self, "_analysis_plot_hint_lbl", None)
        if title_lbl is not None:
            title_lbl.configure(text=title)
        if hint_lbl is not None:
            hint_lbl.configure(text=description)
        img = Image.open(path)
        self._analysis_plot_canvas.set_image(img)
        self._analysis_plot_canvas.fit_to_view()
    except Exception as e:
        logger.error(f"Nie udało się wyswietlic wykresu: {e}")

def _build_ranking_panel_v2(self, parent):
    palette = getattr(self.app, "palette", {})
    shell = ttk.Frame(parent, padding=10, style="Panel.TFrame")
    shell.pack(fill=tk.BOTH, expand=True)

    ranking_intro_lbl = ttk.Label(
        shell,
        text=(
            "Ranking działa jak wyścig: konie to modele, tor to jeden wspólny dataset testowy, "
            "a wynik powstaje dopiero po sprawdzeniu wszystkich modeli na tym samym materiale."
        ),
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=760,
    )
    ranking_intro_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))

    config_box = ttk.LabelFrame(shell, text=" Ranking modeli ", padding=10)
    config_box.pack(fill=tk.X, pady=(0, 10))

    self.rank_models_dir = tk.StringVar(value=str(self._get_ranking_models_default_dir()))
    self.rank_data_dir = tk.StringVar()
    self.rank_split_var = tk.StringVar(value="test")
    if not hasattr(self, "rank_scope_var"):
        default_scope = "Projekt" if CAMPAIGN.get_active_project_name() else "Wszystkie"
        self.rank_scope_var = tk.StringVar(value=default_scope)
    self.rank_progress_var = tk.DoubleVar(value=0.0)
    self.btn_run_rank = None
    self.btn_cancel_rank = None
    self.rank_progress = None
    self.rank_reference_hint_lbl = None
    self._rank_advanced_modal = None
    self._ranking_track_modal = None
    self._ranking_participants_modal = None

    rank_config_hint_lbl = ttk.Label(
        config_box,
        text=(
            "Najpierw wybierz tor testowy, potem uruchom wyścig. Ranking niczego nie zatwierdza automatycznie."
        ),
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=760,
    )
    rank_config_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))

    target_text = (
        f"Konie: {self._get_ranking_task_label()} | czekam na wybór toru testowego."
    )
    self.rank_target_lbl = ttk.Label(
        config_box,
        text=target_text,
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
    )
    self.rank_target_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))

    self.rank_track_lbl = ttk.Label(
        config_box,
        text="Tor testowy: nie wybrano",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=760,
    )
    self.rank_track_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))

    dynamic_row = ttk.Frame(config_box, style="Panel.TFrame")
    dynamic_row.pack(fill=tk.X, pady=(0, 8))

    def _dynamic_value(parent, title: str, value: str, color: str):
        ttk.Label(parent, text=title, style="PanelMuted.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        label = tk.Label(
            parent,
            text=value,
            bg=palette.get("panel", "#252526"),
            fg=color,
            font=("Segoe UI", 9, "bold"),
            padx=6,
            pady=1,
            anchor=tk.W,
        )
        label.pack(side=tk.LEFT, padx=(0, 12))
        return label

    self.rank_count_value_lbl = _dynamic_value(
        dynamic_row,
        "Konie:",
        "0",
        palette.get("success", "#2ecc71"),
    )
    self.rank_track_count_value_lbl = _dynamic_value(
        dynamic_row,
        "Tor:",
        "brak",
        palette.get("warning", "#f0b44c"),
    )

    rank_primary = ttk.Frame(config_box, style="Panel.TFrame")
    rank_primary.pack(fill=tk.X, pady=(2, 8))
    rank_primary.columnconfigure(0, weight=1)

    self.btn_run_rank = ttk.Button(
        rank_primary,
        text="[ TOR ] Wybierz tor testowy",
        style="Accent.TButton",
        command=self._run_ranking_v2,
    )
    self.btn_run_rank.grid(row=0, column=0, sticky="ew", ipady=8)

    rank_secondary = ttk.Frame(config_box, style="Panel.TFrame")
    rank_secondary.pack(fill=tk.X, pady=(0, 10))
    rank_secondary.columnconfigure(0, weight=1)
    rank_secondary.columnconfigure(1, weight=1)
    rank_secondary.columnconfigure(2, weight=1)

    self.btn_cancel_rank = ttk.Button(
        rank_secondary,
        text="[ STOP ] Anuluj",
        command=self._cancel_ranking_v2,
        state=tk.DISABLED,
    )
    self.btn_open_rank_participants = ttk.Button(
        rank_secondary,
        text="[ KONIE ] Uczestnicy",
        command=self._open_ranking_participants_modal,
    )
    self.btn_open_rank_participants.grid(row=0, column=0, sticky="ew", padx=(0, 5), ipady=3)
    self.btn_open_rank_track = ttk.Button(
        rank_secondary,
        text="[ TOR ] Zmień tor testowy",
        command=self._open_ranking_track_modal,
    )
    self.btn_open_rank_track.grid(row=0, column=1, sticky="ew", padx=(5, 5), ipady=3)
    self.btn_open_rank_results = ttk.Button(
        rank_secondary,
        text="[ WYNIKI ] Pokaż wyniki",
        command=lambda: _open_ranking_results_modal(self),
    )
    self.btn_open_rank_results.grid(row=0, column=2, sticky="ew", padx=(5, 5), ipady=3)
    self.btn_open_rank_advanced = ttk.Button(
        rank_secondary,
        text="[ OPCJE ] Zaawansowane",
        command=self._open_ranking_advanced_modal,
    )
    self.btn_open_rank_advanced.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 5), pady=(6, 0), ipady=3)
    self.btn_cancel_rank.grid(row=1, column=2, sticky="ew", padx=(5, 0), pady=(6, 0), ipady=3)

    self.rank_progress = TrainProgressBar(
        config_box,
        variable=self.rank_progress_var,
        mode="determinate",
        thickness=6,
        trough_color=palette.get("panel_alt", palette.get("panel", "#252526")),
        fill_color=palette.get("accent_hover", palette.get("accent", "#0e639c")),
        bg=palette.get("panel", "#252526"),
        height=10,
    )
    self.rank_progress.pack(fill=tk.X, pady=(0, 4))

    self.rank_status = ttk.Label(config_box, text="Gotowy", style="PanelMuted.TLabel")
    self.rank_status.pack(anchor=tk.W, fill=tk.X)

    results_scroll_host = ttk.Frame(shell, style="Panel.TFrame")
    results_scroll_host.grid_rowconfigure(0, weight=1)
    results_scroll_host.grid_columnconfigure(0, weight=1)

    results_canvas = tk.Canvas(
        results_scroll_host,
        bg=palette.get("panel", "#252526"),
        bd=0,
        highlightthickness=0,
    )
    results_canvas.grid(row=0, column=0, sticky="nsew")
    results_scrollbar = WebSlimScrollbar(
        results_scroll_host,
        orient=tk.VERTICAL,
        command=results_canvas.yview,
        auto_hide=False,
    )
    results_scrollbar.grid(row=0, column=1, sticky="ns")
    results_canvas.configure(yscrollcommand=results_scrollbar.set)

    results_content = ttk.Frame(results_canvas, style="Panel.TFrame")
    results_window = results_canvas.create_window((0, 0), window=results_content, anchor="nw")
    self.rank_results_canvas = results_canvas
    self.rank_results_content = results_content

    def _sync_results_scrollregion(_event=None):
        try:
            results_canvas.configure(scrollregion=results_canvas.bbox("all"))
        except Exception:
            pass

    def _sync_results_canvas_width(event=None):
        try:
            width = max(360, int(results_canvas.winfo_width() or 0) - 2)
            results_canvas.itemconfigure(results_window, width=width)
        except Exception:
            pass
        _sync_results_scrollregion()

    def _on_results_mousewheel(event):
        try:
            direction = -1 if (getattr(event, "num", None) == 4 or int(getattr(event, "delta", 0)) > 0) else 1
            results_canvas.yview_scroll(direction * 3, "units")
            return "break"
        except Exception:
            return None

    def _bind_results_scroll_children(widget):
        if widget is not None:
            try:
                widget.bind("<MouseWheel>", _on_results_mousewheel, add="+")
                widget.bind("<Button-4>", _on_results_mousewheel, add="+")
                widget.bind("<Button-5>", _on_results_mousewheel, add="+")
            except Exception:
                pass
        try:
            children = widget.winfo_children()
        except Exception:
            return
        for child in children:
            _bind_results_scroll_children(child)

    results_content.bind("<Configure>", _sync_results_scrollregion, add="+")
    results_canvas.bind("<Configure>", _sync_results_canvas_width, add="+")
    results_canvas.bind("<MouseWheel>", _on_results_mousewheel, add="+")
    results_canvas.bind("<Button-4>", _on_results_mousewheel, add="+")
    results_canvas.bind("<Button-5>", _on_results_mousewheel, add="+")

    ttk.Label(
        results_content,
        text="Wyniki i kandydaci",
        style="Panel.TLabel",
        anchor=tk.W,
    ).pack(anchor=tk.W, fill=tk.X, pady=(0, 6))

    ranking_decision_hint_lbl = ttk.Label(
        results_content,
        text="Porównaj metryki i dopiero potem jawnie ustaw model projektowy.",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=520,
    )
    ranking_decision_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))

    ranking_metrics_hint_lbl = ttk.Label(
        results_content,
        text=(
            "Ocena to główny wynik sortowania modeli. P/C oznacza precyzję i czułość: "
            "ile wykryć było poprawnych oraz ile prawdziwych obiektów model odnalazł."
        ),
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=760,
    )
    self.rank_metrics_hint_lbl = ranking_metrics_hint_lbl
    ranking_metrics_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))

    def _sync_ranking_copy_wraps(_event=None):
        try:
            shell_width = max(420, int(shell.winfo_width() or 0) - 30)
        except Exception:
            shell_width = 760
        try:
            config_width = max(420, int(config_box.winfo_width() or shell_width) - 28)
        except Exception:
            config_width = shell_width
        for widget, width in (
            (ranking_intro_lbl, shell_width),
            (rank_config_hint_lbl, config_width),
            (getattr(self, "rank_target_lbl", None), config_width),
            (getattr(self, "rank_track_lbl", None), config_width),
            (ranking_decision_hint_lbl, shell_width),
            (ranking_metrics_hint_lbl, shell_width),
        ):
            try:
                if widget is not None:
                    widget.configure(wraplength=width)
            except Exception:
                pass

    shell.bind("<Configure>", _sync_ranking_copy_wraps, add="+")
    config_box.bind("<Configure>", _sync_ranking_copy_wraps, add="+")

    leader_bg = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel", "#252526"),
        0.88,
    )
    leader_border = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        0.48,
    )
    self.rank_leader_card = tk.Frame(
        results_content,
        bg=leader_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=leader_border,
        highlightcolor=leader_border,
    )
    self.rank_leader_card.pack(fill=tk.X, pady=(0, 10))
    self.rank_leader_title = tk.Label(
        self.rank_leader_card,
        text="Brak wyników dla wybranego zakresu",
        bg=leader_bg,
        fg=palette.get("fg", "#f3f3f3"),
        font=("Segoe UI", 10, "bold"),
        anchor=tk.W,
        padx=10,
        pady=4,
    )
    self.rank_leader_title.pack(fill=tk.X)
    self.rank_leader_hint = tk.Label(
        self.rank_leader_card,
        text="Ranking podpowiada kandydata. Model projektowy wybieramy jawnie.",
        bg=leader_bg,
        fg=palette.get("muted", "#c7c7c7"),
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        padx=10,
        pady=4,
    )
    self.rank_leader_hint.pack(fill=tk.X)

    table_frame = ttk.Frame(results_content, style="Panel.TFrame")
    table_frame.pack(fill=tk.BOTH, expand=True)
    table_frame.rowconfigure(0, weight=1)
    table_frame.columnconfigure(0, weight=1)

    cols = ("Pozycja", "Model", "Zakres", "Data", "Ocena", "P/C", "Status")
    self.rank_tree = ttk.Treeview(table_frame, columns=cols, show="headings")
    for c in cols:
        self.rank_tree.heading(c, text=c)
    self.rank_tree.heading("Data", text="Data rankingu")
    self.rank_tree.column("Pozycja", width=84, anchor=tk.CENTER, stretch=False)
    self.rank_tree.column("Model", width=420, minwidth=280, anchor=tk.W, stretch=True)
    self.rank_tree.column("Zakres", width=76, minwidth=64, anchor=tk.CENTER, stretch=False)
    self.rank_tree.column("Data", width=124, minwidth=108, anchor=tk.CENTER, stretch=False)
    self.rank_tree.column("Ocena", width=82, minwidth=68, anchor=tk.CENTER, stretch=False)
    self.rank_tree.column("P/C", width=112, minwidth=92, anchor=tk.CENTER, stretch=False)
    self.rank_tree.column("Status", width=148, minwidth=118, anchor=tk.W, stretch=False)

    try:
        self.rank_tree.tag_configure(
            "leader",
            background=blend_hex_colors(
                palette.get("success", "#2ecc71"),
                palette.get("panel", "#252526"),
                0.86,
            ),
            foreground=palette.get("fg", "#f3f3f3"),
        )
        self.rank_tree.tag_configure(
            "project",
            foreground=palette.get("fg", "#f3f3f3"),
        )
        self.rank_tree.tag_configure(
            "global",
            foreground=palette.get("muted", "#c7c7c7"),
        )
        self.rank_tree.tag_configure(
            "pending_project",
            foreground=palette.get("warning", "#f0b44c"),
        )
        self.rank_tree.tag_configure(
            "pending_global",
            foreground=palette.get("muted", "#c7c7c7"),
        )
    except Exception:
        pass

    yscroll = WebSlimScrollbar(table_frame, orient=tk.VERTICAL, command=self.rank_tree.yview)
    xscroll = WebSlimScrollbar(table_frame, orient=tk.HORIZONTAL, command=self.rank_tree.xview)
    self.rank_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    self.rank_tree.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    self.rank_tree.bind("<Button-3>", self._show_ranking_context_menu, add="+")
    self.rank_tree.bind("<Button-2>", self._show_ranking_context_menu, add="+")
    _bind_results_scroll_children(results_content)
    try:
        self.frame.after_idle(_sync_results_canvas_width)
    except Exception:
        pass

    HELP.bind_help(self.btn_open_rank_advanced, "tr_rank_conf")
    HELP.bind_help(self.btn_open_rank_track, "tr_rank_reference")
    HELP.bind_help(self.btn_run_rank, "tr_rank_btn")
    HELP.bind_help(self.btn_cancel_rank, "tr_rank_btn")
    HELP.bind_help(config_box, "tr_rank_conf")
    HELP.bind_help(self.rank_tree, "tr_rank_table")
    self.rank_models_dir.trace_add("write", self._refresh_ranking_reference_ui)
    self.rank_data_dir.trace_add("write", self._refresh_ranking_reference_ui)
    self.rank_split_var.trace_add("write", self._refresh_ranking_reference_ui)
    self._prefill_ranking_reference_if_empty()
    self._refresh_ranking_reference_ui()

def _open_ranking_results_modal(self):
    existing = getattr(self, "_ranking_results_modal", None)
    try:
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            self._load_ranking()
            return
    except Exception:
        pass

    palette = getattr(self.app, "palette", {})
    dialog = tk.Toplevel(getattr(self, "frame", None))
    self._ranking_results_modal = dialog
    dialog.title("Porównanie modeli - uczestnicy i wyniki")
    dialog.configure(bg=palette.get("panel", "#252526"))
    dialog.resizable(True, True)
    try:
        dialog.transient(self.frame.winfo_toplevel())
    except Exception:
        pass

    previous_refs = {
        "tree": getattr(self, "rank_tree", None),
        "leader_title": getattr(self, "rank_leader_title", None),
        "leader_hint": getattr(self, "rank_leader_hint", None),
        "metrics_hint": getattr(self, "rank_metrics_hint_lbl", None),
    }

    def close_dialog():
        try:
            self.rank_tree = previous_refs.get("tree")
            self.rank_leader_title = previous_refs.get("leader_title")
            self.rank_leader_hint = previous_refs.get("leader_hint")
            self.rank_metrics_hint_lbl = previous_refs.get("metrics_hint")
            self._ranking_results_modal = None
        except Exception:
            pass
        try:
            dialog.destroy()
        except Exception:
            pass

    dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    shell = ttk.Frame(dialog, padding=12, style="Panel.TFrame")
    shell.pack(fill=tk.BOTH, expand=True)
    shell.grid_rowconfigure(3, weight=1)
    shell.grid_columnconfigure(0, weight=1)

    header = ttk.Frame(shell, style="Panel.TFrame")
    header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    header.columnconfigure(0, weight=1)
    ttk.Label(
        header,
        text="Porównanie modeli",
        style="Panel.TLabel",
        anchor=tk.W,
    ).grid(row=0, column=0, sticky="ew")
    ttk.Label(
        header,
        text=(
            "Tabela pokazuje uczestników wyścigu także przed testem. Modele bez wyniku mają status `czeka na test`."
        ),
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
    ).grid(row=1, column=0, sticky="ew", pady=(2, 0))

    scope_row = ttk.Frame(shell, style="Panel.TFrame")
    scope_row.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(
        scope_row,
        text=f"Zakres i startujące modele ustawisz w modalu uczestników. Aktywnie: {_format_ranking_scope_label(self)}.",
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(
        scope_row,
        text="[ KONIE ] Uczestnicy",
        command=self._open_ranking_participants_modal,
    ).pack(side=tk.RIGHT, padx=(8, 0))
    ttk.Button(
        scope_row,
        text="Zaawansowane",
        command=self._open_ranking_advanced_modal,
    ).pack(side=tk.RIGHT, padx=(8, 0))

    leader_bg = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel", "#252526"),
        0.88,
    )
    leader_border = blend_hex_colors(
        palette.get("success", "#2ecc71"),
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        0.48,
    )
    leader_card = tk.Frame(
        shell,
        bg=leader_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=leader_border,
        highlightcolor=leader_border,
    )
    leader_card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
    self.rank_leader_title = tk.Label(
        leader_card,
        text="Brak wyników dla wybranego zakresu",
        bg=leader_bg,
        fg=palette.get("fg", "#f3f3f3"),
        font=("Segoe UI", 10, "bold"),
        anchor=tk.W,
        padx=10,
        pady=4,
    )
    self.rank_leader_title.pack(fill=tk.X)
    self.rank_leader_hint = tk.Label(
        leader_card,
        text="Ranking podpowiada kandydata. Model projektowy wybieramy jawnie.",
        bg=leader_bg,
        fg=palette.get("muted", "#c7c7c7"),
        font=("Segoe UI", 8),
        anchor=tk.W,
        justify=tk.LEFT,
        padx=10,
        pady=4,
    )
    self.rank_leader_hint.pack(fill=tk.X)

    table_frame = ttk.Frame(shell, style="Panel.TFrame")
    table_frame.grid(row=3, column=0, sticky="nsew")
    table_frame.rowconfigure(0, weight=1)
    table_frame.columnconfigure(0, weight=1)

    cols = ("Pozycja", "Model", "Zakres", "Data", "Ocena", "P/C", "Status")
    self.rank_tree = ttk.Treeview(table_frame, columns=cols, show="headings")
    for col in cols:
        self.rank_tree.heading(col, text=col)
    self.rank_tree.heading("Data", text="Data rankingu")
    self.rank_tree.column("Pozycja", width=92, anchor=tk.CENTER, stretch=False)
    self.rank_tree.column("Model", width=520, minwidth=340, anchor=tk.W, stretch=True)
    self.rank_tree.column("Zakres", width=78, minwidth=64, anchor=tk.CENTER, stretch=False)
    self.rank_tree.column("Data", width=132, minwidth=112, anchor=tk.CENTER, stretch=False)
    self.rank_tree.column("Ocena", width=86, minwidth=70, anchor=tk.CENTER, stretch=False)
    self.rank_tree.column("P/C", width=118, minwidth=96, anchor=tk.CENTER, stretch=False)
    self.rank_tree.column("Status", width=160, minwidth=126, anchor=tk.W, stretch=False)

    try:
        self.rank_tree.tag_configure(
            "leader",
            background=blend_hex_colors(
                palette.get("success", "#2ecc71"),
                palette.get("panel", "#252526"),
                0.86,
            ),
            foreground=palette.get("fg", "#f3f3f3"),
        )
        self.rank_tree.tag_configure("project", foreground=palette.get("fg", "#f3f3f3"))
        self.rank_tree.tag_configure("global", foreground=palette.get("muted", "#c7c7c7"))
        self.rank_tree.tag_configure(
            "pending_project",
            foreground=palette.get("warning", "#f0b44c"),
        )
        self.rank_tree.tag_configure(
            "pending_global",
            foreground=palette.get("muted", "#c7c7c7"),
        )
    except Exception:
        pass

    yscroll = WebSlimScrollbar(table_frame, orient=tk.VERTICAL, command=self.rank_tree.yview)
    xscroll = WebSlimScrollbar(table_frame, orient=tk.HORIZONTAL, command=self.rank_tree.xview)
    self.rank_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    self.rank_tree.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    self.rank_tree.bind("<Button-3>", self._show_ranking_context_menu, add="+")
    self.rank_tree.bind("<Button-2>", self._show_ranking_context_menu, add="+")

    bottom = ttk.Frame(shell, style="Panel.TFrame")
    bottom.grid(row=4, column=0, sticky="ew", pady=(10, 0))
    ttk.Label(
        bottom,
        text="PPM na modelu projektu: wybór jako wynik bramki, dotrenowanie, szczegóły runu.",
        style="PanelMuted.TLabel",
        anchor=tk.W,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(bottom, text="Zamknij", command=close_dialog).pack(side=tk.RIGHT)
    ttk.Button(
        bottom,
        text="[ PODGLĄD ] Przegląd raportu",
        command=self._open_ranking_report_viewer,
    ).pack(side=tk.RIGHT, padx=(0, 8))
    ttk.Button(
        bottom,
        text="[ RAPORT ] Udokumentuj ranking",
        command=self._export_ranking_analysis_report,
    ).pack(side=tk.RIGHT, padx=(0, 8))

    try:
        HELP.bind_help(self.rank_tree, "tr_rank_table")
    except Exception:
        pass

    try:
        dialog.update_idletasks()
        root = self.frame.winfo_toplevel()
        width = min(max(980, int(root.winfo_width() * 0.86)), 1380)
        height = min(max(620, int(root.winfo_height() * 0.78)), 900)
        x = int(root.winfo_rootx() + max(0, (root.winfo_width() - width) // 2))
        y = int(root.winfo_rooty() + max(0, (root.winfo_height() - height) // 2))
        dialog.geometry(f"{width}x{height}+{x}+{y}")
    except Exception:
        dialog.geometry("1120x700")

    self._load_ranking()
    try:
        dialog.lift()
        dialog.focus_force()
    except Exception:
        pass

def _open_ranking_advanced_modal(self):
    existing = getattr(self, "_rank_advanced_modal", None)
    try:
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
    except Exception:
        pass

    palette = getattr(self.app, "palette", {})
    dialog = tk.Toplevel(getattr(self, "frame", None))
    self._rank_advanced_modal = dialog
    dialog.title("Zaawansowane źródła rankingu")
    dialog.configure(bg=palette.get("panel", "#252526"))
    try:
        dialog.transient(self.frame.winfo_toplevel())
    except Exception:
        pass

    def close_dialog():
        try:
            self.rank_reference_hint_lbl = None
            self.rank_advanced_status = None
            self._rank_advanced_modal = None
        except Exception:
            pass
        try:
            dialog.destroy()
        except Exception:
            pass

    dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    draft_data_dir = tk.StringVar(value=str(getattr(self, "rank_data_dir", tk.StringVar()).get() or ""))
    draft_split_var = tk.StringVar(value=str(getattr(self, "rank_split_var", tk.StringVar(value="test")).get() or "test"))

    shell = ttk.Frame(dialog, padding=14, style="Panel.TFrame")
    shell.pack(fill=tk.BOTH, expand=True)

    ttk.Label(
        shell,
        text="Ręczny wybór toru testowego",
        style="Panel.TLabel",
        anchor=tk.W,
    ).pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
    ttk.Label(
        shell,
        text=(
            f"Ręcznie wskaż tor testowy dla trybu: {self._get_ranking_task_label()}. "
            "Zmiany są robocze, dopóki nie użyjesz przycisku zastosowania na dole okna."
        ),
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=520,
    ).pack(anchor=tk.W, fill=tk.X, pady=(0, 12))

    form = ttk.LabelFrame(shell, text=" Ręczny wybór toru ", padding=10)
    form.pack(fill=tk.X, pady=(0, 10))

    reference_label = (
        "Tor testowy znaków (data.yaml):"
        if self._get_ranking_task_target() == "char"
        else "Tor testowy tablic (run z annotations.xml):"
    )
    ttk.Label(form, text=reference_label, style="Panel.TLabel").pack(anchor=tk.W)
    row2 = ttk.Frame(form, style="Panel.TFrame")
    row2.pack(fill=tk.X, pady=(4, 8))
    ttk.Entry(row2, textvariable=draft_data_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
    if self._get_ranking_task_target() == "char":
        def choose_data_yaml():
            initial = self._get_ranking_reference_picker_dir()
            dialog_kwargs = {
                "title": "Wskaż data.yaml toru testowego",
                "filetypes": (("YOLO data.yaml", "data.yaml"), ("YAML", "*.yaml *.yml"), ("Wszystkie pliki", "*.*")),
                "parent": dialog,
            }
            if initial and Path(initial).exists():
                dialog_kwargs["initialdir"] = initial
            selected = filedialog.askopenfilename(**dialog_kwargs)
            if selected:
                draft_data_dir.set(selected)

        ttk.Button(
            row2,
            text="Wybierz data.yaml",
            command=choose_data_yaml,
        ).pack(side=tk.RIGHT, padx=(8, 0))
    else:
        ttk.Button(
            row2,
            text="Wybierz run",
            command=lambda: self._pick_dir(
                draft_data_dir,
                initialdir=self._get_ranking_reference_picker_dir(),
            ),
        ).pack(side=tk.RIGHT, padx=(8, 0))

    self.rank_reference_hint_lbl = ttk.Label(
        form,
        text=(
            "Tor znaków to dataset z data.yaml. Po wskazaniu ścieżki zatwierdź ją przyciskiem na dole."
            if self._get_ranking_task_target() == "char"
            else "Tor tablic to zapisany run z obrazami i annotations.xml. Po wskazaniu ścieżki zatwierdź ją przyciskiem na dole."
        ),
        style="PanelMuted.TLabel",
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=520,
    )
    self.rank_reference_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))

    if self._get_ranking_task_target() == "char":
        split_row = ttk.Frame(form, style="Panel.TFrame")
        split_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(split_row, text="Split toru:", style="PanelMuted.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Combobox(
            split_row,
            textvariable=draft_split_var,
            values=("test", "val"),
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT)
        ttk.Label(
            split_row,
            text="Najlepiej używać splitu test, którego nie używano do wyboru epok.",
            style="PanelMuted.TLabel",
        ).pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
    else:
        ttk.Label(
            form,
            text=f"Próg wykrycia dla nowego porównania: {float(CONFIG.DEFAULT_CONFIDENCE):.2f}.",
            style="PanelMuted.TLabel",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=520,
        ).pack(anchor=tk.W, fill=tk.X)

    self.rank_advanced_status = ttk.Label(
        shell,
        text="Zmiany nie są jeszcze zastosowane. Wskaż tor i użyj przycisku zastosowania.",
        style="PanelMuted.TLabel",
    )
    self.rank_advanced_status.pack(anchor=tk.W, fill=tk.X)

    def apply_manual_sources():
        reference_raw = str(draft_data_dir.get() or "").strip()
        if not reference_raw:
            try:
                self.rank_advanced_status.configure(text="Nie wskazano toru testowego.")
            except Exception:
                pass
            return messagebox.showwarning("Brak toru", "Wskaż dataset/run, który ma być torem testowym rankingu.")

        old_reference = str(getattr(self, "rank_data_dir", tk.StringVar()).get() or "")
        old_split = str(getattr(self, "rank_split_var", tk.StringVar(value="test")).get() or "test")
        try:
            self.rank_split_var.set(str(draft_split_var.get() or "test").strip() or "test")
            self.rank_data_dir.set(reference_raw)
            info = self._resolve_ranking_reference_source(reference_raw)
            if not info.get("ok"):
                self.rank_split_var.set(old_split)
                self.rank_data_dir.set(old_reference)
                message = str(info.get("message") or "Wskazany tor testowy nie jest gotowy.")
                try:
                    self.rank_advanced_status.configure(text=message)
                except Exception:
                    pass
                return messagebox.showwarning("Tor nie jest gotowy", message)
            try:
                self._refresh_ranking_reference_ui()
                self._refresh_ranking_start_state()
                self._load_ranking()
            except Exception:
                pass
            try:
                self.rank_advanced_status.configure(
                    text=f"Zastosowano tor: {info.get('reference_name') or Path(reference_raw).name}"
                )
            except Exception:
                pass
            close_dialog()
        except Exception as exc:
            try:
                self.rank_split_var.set(old_split)
                self.rank_data_dir.set(old_reference)
            except Exception:
                pass
            logger.error(f"Nie udało się zastosować ręcznego toru rankingu: {exc}")
            return messagebox.showerror("Błąd", f"Nie udało się zastosować toru testowego:\n{exc}")

    bottom = ttk.Frame(shell, style="Panel.TFrame")
    bottom.pack(fill=tk.X, pady=(12, 0))
    ttk.Button(bottom, text="Zamknij", command=close_dialog).pack(side=tk.RIGHT)
    ttk.Button(
        bottom,
        text="[ TOR ] Zastosuj wybrany tor",
        style="Accent.TButton",
        command=apply_manual_sources,
    ).pack(side=tk.RIGHT, padx=(0, 8))

    HELP.bind_help(row2, "tr_rank_reference")
    HELP.bind_help(self.rank_reference_hint_lbl, "tr_rank_reference")
    HELP.bind_help(form, "tr_rank_conf")

    if not str(draft_data_dir.get() or "").strip():
        try:
            draft_data_dir.set(self._get_default_ranking_reference_dir())
        except Exception:
            pass

    try:
        dialog.update_idletasks()
        root = self.frame.winfo_toplevel()
        width = max(540, int(dialog.winfo_reqwidth() or 540))
        height = max(360, int(dialog.winfo_reqheight() or 360))
        x = int(root.winfo_rootx() + max(0, (root.winfo_width() - width) // 2))
        y = int(root.winfo_rooty() + max(0, (root.winfo_height() - height) // 2))
        dialog.geometry(f"{width}x{height}+{x}+{y}")
    except Exception:
        dialog.geometry("560x390")

def _run_ranking_v2(self):
    if self.rank_is_running:
        return

    self._ensure_plate_ranking_engine()
    target = self._get_ranking_task_target()
    target_task = self._get_ranking_task_label(target)
    selected_scope = _get_ranking_scope(self)
    models_dir_raw = str(getattr(self, "rank_models_dir", tk.StringVar()).get() or "").strip()
    reference_raw = str(getattr(self, "rank_data_dir", tk.StringVar()).get() or "").strip()
    models_dir = Path(models_dir_raw) if models_dir_raw else Path(".")

    if selected_scope == "Globalne" and (not models_dir.exists() or not models_dir.is_dir()):
        return messagebox.showerror("Błąd", "Wskaż poprawny folder z modelami .pt.")

    reference_info = self._resolve_ranking_reference_source(reference_raw)
    if not reference_info.get("ok"):
        return messagebox.showerror(
            "[ TOR ] Wybierz tor testowy",
            str(reference_info.get("message") or "Wybierz gotowy tor testowy przed uruchomieniem rankingu."),
        )

    if not self._begin_step4_operation("z4.ranking.run", "Z4: ranking modeli"):
        return
    self.rank_is_running = True
    self.rank_cancel_requested = False
    try:
        self.btn_run_rank.config(state=tk.DISABLED)
    except Exception:
        pass
    try:
        self.rank_progress_var.set(0)
    except Exception:
        pass
    self._set_ranking_ui_state(
        status="Przygotowuję ranking...",
        status_color="gray",
        button_text="[ START ] Przygotowanie...",
        cancel_enabled=True,
        preparing=True,
        progress_value=0,
    )
    self._append_ranking_log(f"Start przygotowania rankingu: {target_task}.")
    self._append_ranking_log(f"Zakres koni: {_format_ranking_scope_label(self, selected_scope, target)}.")
    self._append_ranking_log(f"Folder modeli: {models_dir}")
    if reference_raw:
        self._append_ranking_log(f"Wybrany tor testowy: {reference_raw}")
    else:
        self._append_ranking_log("Nie wskazano jeszcze toru testowego.")
    self._start_ranking_watchdog("start przygotowania rankingu")

    def worker():
        try:
            ranking_started_at = time.perf_counter()
            cancelled = False
            self._touch_ranking_watchdog("sprawdzanie toru testowego")
            self._set_ranking_ui_state(
                status="Sprawdzam tor testowy...",
                status_color="gray",
                button_text="[ START ] Przygotowanie...",
                cancel_enabled=True,
                preparing=True,
            )
            reference_info = self._resolve_ranking_reference_source(reference_raw)
            if not reference_info.get("ok"):
                self._append_ranking_log(str(reference_info.get("message") or "Nie udało się przygotować toru testowego."))
                self._ui(
                    lambda: messagebox.showerror(
                        "Błąd",
                        str(reference_info.get("message") or "Wybierz gotowy tor testowy."),
                    )
                )
                return
            if not self.rank_is_running:
                cancelled = True
                self._append_ranking_log("Przerwano ranking po przygotowaniu toru testowego.")
                return

            self._append_ranking_log(
                f"Zestaw odniesienia: {reference_info.get('reference_name') or '-'} | "
                f"obrazy: {int(reference_info.get('image_count', 0) or 0)}"
            )
            self._append_ranking_log(
                f"Przygotowanie materiału odniesienia zajelo {time.perf_counter() - ranking_started_at:.1f}s."
            )
            self._touch_ranking_watchdog(f"szukanie modeli: {target_task}")

            self._set_ranking_ui_state(
                status=f"Szukam modeli: {target_task}...",
                status_color="gray",
                button_text="[ START ] Przygotowanie...",
                cancel_enabled=True,
                preparing=True,
            )
            all_models_to_test = self._collect_ranking_participant_candidates(models_dir, target, selected_scope)
            models_to_test = _filter_enabled_ranking_participants(self, all_models_to_test)
            if not self.rank_is_running:
                cancelled = True
                self._append_ranking_log("Przerwano ranking po odczytaniu listy modeli.")
                return

            if all_models_to_test and not models_to_test:
                self._append_ranking_log("Nie zaznaczono żadnego uczestnika rankingu.")
                self._ui(
                    lambda: messagebox.showinfo(
                        "Info",
                        "Nie zaznaczono żadnego modelu do rankingu. Otwórz Uczestników rankingu modeli i zostaw co najmniej jeden model ze statusem Startuje.",
                    )
                )
                return

            if not models_to_test:
                self._append_ranking_log(
                    f"Nie znaleziono uczestników dla trybu {target_task} i zakresu {_format_ranking_scope_label(self, selected_scope, target)}."
                )
                self._ui(
                    lambda: messagebox.showinfo(
                        "Info",
                        f"Brak modeli dla trybu {target_task} w wybranym zakresie rankingu.",
                    )
                )
                return

            selected_device_display = self._normalize_training_device_choice(self.device_var.get())
            effective_device_raw, effective_device_profile = self._get_effective_training_device_profile(selected_device_display)
            device = self._device_to_ultralytics(self.device_var.get())
            if effective_device_profile is not None:
                effective_device_desc = (
                    f"{effective_device_profile.get('name', effective_device_raw)} "
                    f"({float(effective_device_profile.get('memory_gb', 0.0) or 0.0):.1f} GB VRAM)"
                )
            else:
                effective_device_desc = "CPU"

            if target == "char":
                if not YOLO_AVAILABLE:
                    self._append_ranking_log("Brak modułu YOLO - nie można uruchomić rankingu znaków.")
                    self._ui(lambda: messagebox.showerror("Błąd", "Brak modułu YOLO."))
                    return
                YoloClass = get_yolo_class()
                if YoloClass is None:
                    self._append_ranking_log("Nie udało się załadować klasy YOLO.")
                    self._ui(lambda: messagebox.showerror("Błąd", "Nie udało się załadować modułu YOLO."))
                    return

                data_yaml = Path(str(reference_info.get("data_yaml_path") or reference_info.get("yaml_path") or "").strip())
                split_name = str(reference_info.get("split_name") or self._get_ranking_split_name()).strip() or "test"
                sample_count = int(reference_info.get("image_count", 0) or 0)
                if not data_yaml.exists():
                    self._append_ranking_log("Tor testowy znaków nie ma pliku data.yaml.")
                    self._ui(lambda: messagebox.showerror("Błąd", "Tor testowy znaków nie ma pliku data.yaml."))
                    return

                total_models = len(models_to_test)
                reference_dataset = build_dataset_display_ref(
                    data_yaml.parent,
                    target_hint="char",
                    counts={split_name: sample_count, "total": sample_count},
                )
                self._append_ranking_log(
                    f"Przygotowanie zakończone. Modele znaków: {total_models} | "
                    f"dataset odniesienia: {reference_dataset.id} | split: {split_name} | obrazy: {sample_count}"
                )
                self._append_ranking_log(
                    f"Urządzenie rankingu: {selected_device_display} -> {effective_device_desc} | backend Ultralytics: {device}"
                )
                self._set_ranking_ui_state(
                    status=f"Waliduję modele znaków... 0/{total_models}",
                    status_color="gray",
                    button_text="[ RANKING ] Porównywanie...",
                    cancel_enabled=True,
                    preparing=False,
                    progress_value=0,
                )

                for idx, model_path in enumerate(models_to_test):
                    if not self.rank_is_running:
                        cancelled = True
                        break

                    model_started_at = time.perf_counter()
                    model_display = Path(model_path).name
                    self._touch_ranking_watchdog(f"walidacja modelu znaków {model_display}")
                    self._append_ranking_log(f"{idx + 1}/{total_models} | Start walidacji: {model_display}")
                    self._set_ranking_ui_state(
                        status=f"Walidacja {model_display} ({idx + 1}/{total_models})",
                        status_color="gray",
                    )

                    model = None
                    try:
                        model = YoloClass(str(model_path))
                        val_kwargs = {
                            "data": str(data_yaml),
                            "split": split_name,
                            "verbose": False,
                        }
                        if device is not None:
                            val_kwargs["device"] = device
                        metrics = model.val(**val_kwargs)
                        stats = _extract_yolo_ranking_metrics(
                            metrics,
                            split_name=split_name,
                            total_images=sample_count,
                        )
                    except Exception as model_error:
                        self._append_ranking_log(f"Pominięto {model_display}: walidacja nie powiodła się: {model_error}")
                        continue
                    finally:
                        try:
                            del model
                        except Exception:
                            pass
                        try:
                            cleanup_gpu_memory()
                        except Exception:
                            pass

                    self.ranking_engine.add_entry(
                        model_name=model_path.name,
                        model_path=str(model_path),
                        comparison_stats=stats,
                        task_type=target_task,
                        reference_name=str(reference_info.get("reference_name") or ""),
                        reference_path=str(reference_info.get("reference_dir") or ""),
                        save=False,
                    )
                    self._ui(lambda p=((idx + 1) / max(1, total_models)) * 100: self.rank_progress_var.set(p))
                    self._append_ranking_log(
                        f"Zakończono {model_display} | mAP50-95={float(stats.get('map50_95', 0) or 0):.1f}% | "
                        f"mAP50={float(stats.get('map50', 0) or 0):.1f}% | "
                        f"Precision={float(stats.get('precision', 0) or 0):.1f}% | "
                        f"Recall={float(stats.get('recall', 0) or 0):.1f}% | "
                        f"czas: {time.perf_counter() - model_started_at:.1f}s"
                    )

                try:
                    self._touch_ranking_watchdog("zapisywanie wyników rankingu")
                    self.ranking_engine.flush()
                except Exception as save_error:
                    self._append_ranking_log(f"Ostrzeżenie: nie udało się zapisać rankingu: {save_error}")
                if cancelled or self.rank_cancel_requested:
                    self._append_ranking_log("Ranking anulowany przez użytkownika.")
                    self._ui(lambda: self._load_ranking())
                    self._set_ranking_ui_state(status="Ranking anulowany.", status_color="#d35400")
                else:
                    self._append_ranking_log("Ranking zakończony.")
                    self._ui(lambda: self.rank_progress_var.set(100))
                    self._ui(lambda: self._load_ranking())
                    self._set_ranking_ui_state(status="Ranking zakończony.", status_color="green")
                return

            from ..ranking.annotation_comparator import AnnotationComparator
            from ..annotators.runtime_factory import create_plate_annotator
            from ..exporters.cvat_exporter import CVATExporter

            gt_xml = Path(str(reference_info.get("xml_path") or "").strip())
            images = list(reference_info.get("image_paths") or [])
            if not gt_xml.exists() or not images:
                self._append_ranking_log("Wybrany tor testowy nie zawiera kompletu obrazów i zapisanych zmian tablic.")
                self._ui(
                    lambda: messagebox.showerror(
                        "Błąd",
                        "Wybrany folder nie zawiera kompletu obrazów i zapisanych zmian tablic.",
                    )
                )
                return

            comparator = AnnotationComparator()
            conf_thresh = float(CONFIG.DEFAULT_CONFIDENCE)
            total_models = len(models_to_test)
            temp_xml_path = Path(str(reference_info.get("reference_dir") or models_dir)) / "temp_ranking_auto.xml"
            self._append_ranking_log(
                f"Przygotowanie zakończone. Modele pose: {total_models} | obrazy do porównania: {len(images)}"
            )
            self._append_ranking_log(
                f"Urządzenie rankingu: {selected_device_display} -> {effective_device_desc} | backend Ultralytics: {device}"
            )
            self._set_ranking_ui_state(
                status=f"Porównywanie modeli... 0/{total_models}",
                status_color="gray",
                button_text="[ RANKING ] Porównywanie...",
                cancel_enabled=True,
                preparing=False,
                progress_value=0,
            )

            for idx, model_path in enumerate(models_to_test):
                if not self.rank_is_running:
                    cancelled = True
                    break

                model_started_at = time.perf_counter()
                model_display = format_ranking_model_label(model_path.name, str(model_path), target_task)
                self._touch_ranking_watchdog(f"ladowanie modelu {model_display}")
                self._append_ranking_log(f"{idx + 1}/{total_models} | Start modelu: {model_display}")
                self._set_ranking_ui_state(
                    status=f"Ładowanie modelu {model_display} ({idx + 1}/{total_models})",
                    status_color="gray",
                )

                annotator = create_plate_annotator(model_path, conf_thresh, device)
                success, _msg = annotator.load_models()
                if not self.rank_is_running:
                    cancelled = True
                    try:
                        annotator.unload_models()
                    except Exception:
                        pass
                    break
                if not success:
                    self._append_ranking_log(f"Pominieto model {model_path.name}: nie udało się go załadować.")
                    continue
                self._append_ranking_log(
                    f"{idx + 1}/{total_models} | Model załadowany po {time.perf_counter() - model_started_at:.1f}s. "
                    f"Start analizy {len(images)} obrazów."
                )
                self._touch_ranking_watchdog(f"{model_display}: start analizy obrazów")

                auto_annotations = []
                for img_idx, img_path in enumerate(images):
                    if not self.rank_is_running:
                        cancelled = True
                        break
                    self._touch_ranking_watchdog(
                        f"{model_display}: analiza obrazu {img_idx + 1}/{len(images)}"
                    )
                    auto_annotations.append(annotator.process_image(img_path))
                    sub_pct = ((idx + ((img_idx + 1) / max(1, len(images)))) / max(1, total_models)) * 100
                    self._ui(lambda p=sub_pct: self.rank_progress_var.set(p))
                    if (img_idx == 0) or ((img_idx + 1) % 10 == 0) or (img_idx + 1 == len(images)):
                        self._set_ranking_ui_state(
                            status=(
                                f"Model {model_display} | obraz {img_idx + 1}/{len(images)} "
                                f"({idx + 1}/{total_models})"
                            ),
                            status_color="gray",
                        )
                    if ((img_idx + 1) % 25 == 0) or (img_idx + 1 == len(images)):
                        self._append_ranking_log(
                            f"{idx + 1}/{total_models} | {model_display} | obrazy: {img_idx + 1}/{len(images)}"
                        )

                annotator.unload_models()
                if cancelled:
                    break
                self._touch_ranking_watchdog(f"{model_display}: eksport i porównanie wyników")
                self._set_ranking_ui_state(
                    status=f"Analiza wyników {model_display}...",
                    status_color="gray",
                )

                exporter = CVATExporter()
                exporter.export(auto_annotations, temp_xml_path, include_confidence=True)

                stats = comparator.compare(auto_xml_path=temp_xml_path, corrected_xml_path=gt_xml)
                self.ranking_engine.add_entry(
                    model_name=model_path.name,
                    model_path=str(model_path),
                    comparison_stats=stats,
                    task_type=target_task,
                    reference_name=str(reference_info.get("reference_name") or ""),
                    reference_path=str(reference_info.get("reference_dir") or ""),
                    save=False,
                )
                precision = float(stats.get("precision", 0) or 0)
                recall = float(stats.get("recall", 0) or 0)
                f1_score = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                self._append_ranking_log(
                    f"Zakończono {model_display} | Precision={precision:.1f}% | "
                    f"Recall={recall:.1f}% | F1={f1_score:.1f}% | czas: {time.perf_counter() - model_started_at:.1f}s"
                )

                if temp_xml_path.exists():
                    temp_xml_path.unlink()

            try:
                self._touch_ranking_watchdog("zapisywanie wyników rankingu")
                self.ranking_engine.flush()
            except Exception as save_error:
                self._append_ranking_log(f"Ostrzeżenie: nie udało się zapisać rankingu: {save_error}")
            if cancelled or self.rank_cancel_requested:
                self._append_ranking_log("Ranking anulowany przez użytkownika.")
                self._ui(lambda: self._load_ranking())
                self._set_ranking_ui_state(status="Ranking anulowany.", status_color="#d35400")
            else:
                self._append_ranking_log("Ranking zakończony.")
                self._ui(lambda: self.rank_progress_var.set(100))
                self._ui(lambda: self._load_ranking())
                self._set_ranking_ui_state(status="Ranking zakończony.", status_color="green")

        except Exception as e:
            self._append_ranking_log(f"Błąd rankingu: {e}")
            self._ui(lambda err=e: messagebox.showerror("Błąd", f"Błąd w trakcie rankingu:\n{err}"))
            self._set_ranking_ui_state(status="Błąd rankingu", status_color="red")
        finally:
            self._stop_ranking_watchdog()
            self.rank_is_running = False
            self.rank_cancel_requested = False
            self._end_step4_operation("z4.ranking.run")
            self._set_ranking_ui_state(button_text="[ START ] Uruchom wyścig", cancel_enabled=False, preparing=False)
            self._ui(lambda: self._refresh_ranking_start_state())
            self._ui(self._refresh_training_start_state)

    threading.Thread(target=worker, daemon=True).start()

def _cancel_ranking_v2(self):
    if not getattr(self, "rank_is_running", False):
        return

    self.rank_cancel_requested = True
    self.rank_is_running = False
    self._touch_ranking_watchdog("przerywanie rankingu")
    self._append_ranking_log("Użytkownik zazadal przerwania rankingu. Czekam na bezpieczne zatrzymanie...")
    self._set_ranking_ui_state(
        status="Przerywanie rankingu...",
        status_color="#d35400",
        button_text="[ STOP ] Przerywanie...",
        cancel_enabled=False,
        preparing=False,
    )
