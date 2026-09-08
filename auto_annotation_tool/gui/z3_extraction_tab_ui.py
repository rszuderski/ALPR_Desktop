#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z3/PZ1 extraction tab UI builder extracted from CharacterAnnotationTab."""

import tkinter as tk
import datetime as _datetime
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

from ..campaign_manager import CAMPAIGN
from ..config import CONFIG, logger
from ..character_recognition import PlateGenerator
from ..data_models import Detection, ImageAnnotation
from ..project_cache import PROJECT_CACHE
from .help_manager import HELP
from .lazy_notebook_tab import _LazyNotebookTab
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors


def _is_live_widget(widget) -> bool:
    if widget is None:
        return False
    try:
        return bool(widget.winfo_exists())
    except Exception:
        return False


def _ensure_campaign_detect_splash_widgets(host):
    force_root_surface = bool(getattr(host, "_campaign_detect_splash_force_root_surface", False))
    surface = getattr(host, "frame", None) if force_root_surface else getattr(host, "detect_content_frame", None)
    if not _is_live_widget(surface):
        surface = getattr(host, "frame", None)
    if not _is_live_widget(surface):
        return None

    overlay = getattr(host, "campaign_detect_splash_overlay", None)
    existing_surface = getattr(host, "_campaign_detect_splash_surface", None)
    has_widgets = (
        _is_live_widget(overlay)
        and _is_live_widget(getattr(host, "campaign_detect_splash_card", None))
        and _is_live_widget(getattr(host, "campaign_detect_splash_title_lbl", None))
        and _is_live_widget(getattr(host, "campaign_detect_splash_body_lbl", None))
        and _is_live_widget(getattr(host, "campaign_detect_splash_progress", None))
    )
    if (
        has_widgets
        and bool(getattr(host, "_campaign_detect_splash_visible", False))
        and _is_live_widget(existing_surface)
    ):
        return existing_surface
    if has_widgets and existing_surface is surface:
        return surface

    if _is_live_widget(overlay):
        try:
            overlay.destroy()
        except Exception:
            pass

    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    accent = palette.get("accent", "#4f8de3")
    overlay_bg = blend_hex_colors(panel_bg, "#000000", 0.24)
    card_bg = blend_hex_colors(panel_bg, accent, 0.10)
    border_color = blend_hex_colors(accent, palette.get("panel_border", palette.get("border", "#3c3c3c")), 0.48)

    overlay = tk.Frame(surface, bd=0, highlightthickness=0, bg=overlay_bg)
    card = tk.Frame(
        overlay,
        bd=0,
        highlightthickness=1,
        highlightbackground=border_color,
        highlightcolor=border_color,
        padx=22,
        pady=20,
        bg=card_bg,
    )
    card.place(relx=0.5, rely=0.34, anchor="n")
    card.grid_columnconfigure(0, weight=1, minsize=560)

    title_lbl = tk.Label(
        card,
        text="Przygotowuję wyodrębnione tablice dla Z3",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI Semibold", 13),
        bd=0,
        highlightthickness=0,
        bg=card_bg,
        fg=accent,
    )
    title_lbl.grid(row=0, column=0, sticky="ew")

    body_lbl = tk.Label(
        card,
        text="To automatyczny krok pośredni przed pracą nad znakami.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=560,
        bd=0,
        highlightthickness=0,
        bg=card_bg,
        fg=palette.get("fg", "#f3f3f3"),
    )
    body_lbl.grid(row=1, column=0, sticky="ew", pady=(10, 0))

    progressbar = ttk.Progressbar(card, mode="determinate", maximum=100.0)
    progressbar.grid(row=2, column=0, sticky="ew", pady=(14, 0))

    return_btn = ttk.Button(
        card,
        text="Zatwierdź PZ1 i wróć do pracy T05",
        command=getattr(host, "_return_to_t05_work_after_step3_pz1", lambda: None),
    )
    return_btn.grid(row=3, column=0, sticky="w", pady=(14, 0))
    return_btn.grid_remove()
    overlay.place_forget()

    host.campaign_detect_splash_overlay = overlay
    host.campaign_detect_splash_card = card
    host.campaign_detect_splash_title_lbl = title_lbl
    host.campaign_detect_splash_body_lbl = body_lbl
    host.campaign_detect_splash_progress = progressbar
    host.campaign_detect_splash_return_btn = return_btn
    host._campaign_detect_splash_surface = surface
    host._campaign_detect_splash_visible = False
    host._campaign_detect_splash_key = None
    host._campaign_detect_splash_progress_indeterminate = False
    return surface


def show_campaign_detect_splash(
    host,
    *,
    title: str,
    body: str = "",
    tone: str = "info",
    progress: float | None = None,
    show_progress: bool = True,
    show_return: bool = False,
) -> None:
    title_text = str(title or "").strip()
    body_text = str(body or "").strip()

    surface = _ensure_campaign_detect_splash_widgets(host)
    overlay = getattr(host, "campaign_detect_splash_overlay", None)
    title_lbl = getattr(host, "campaign_detect_splash_title_lbl", None)
    body_lbl = getattr(host, "campaign_detect_splash_body_lbl", None)
    progressbar = getattr(host, "campaign_detect_splash_progress", None)
    return_btn = getattr(host, "campaign_detect_splash_return_btn", None)
    card = getattr(host, "campaign_detect_splash_card", None)
    if overlay is None or title_lbl is None or body_lbl is None or progressbar is None or card is None or surface is None:
        return

    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    fg = palette.get("fg", "#f3f3f3")
    tone_key = str(tone or "info").strip().lower()
    splash_key = (
        title_text,
        body_text,
        tone_key,
        bool(show_progress),
        bool(show_return),
    )
    same_splash = bool(
        getattr(host, "_campaign_detect_splash_visible", False)
        and getattr(host, "_campaign_detect_splash_key", None) == splash_key
    )
    splash_already_placed = bool(
        getattr(host, "_campaign_detect_splash_visible", False)
        and _is_live_widget(overlay)
        and _is_live_widget(surface)
        and str(overlay.winfo_manager())
    )
    if tone_key == "error":
        accent = palette.get("error", "#e74c3c")
    elif tone_key == "success":
        accent = palette.get("success", "#2ecc71")
    else:
        accent = palette.get("accent", "#4f8de3")

    overlay_bg = blend_hex_colors(panel_bg, "#000000", 0.24)
    card_bg = blend_hex_colors(panel_bg, accent, 0.10)
    border_color = blend_hex_colors(accent, palette.get("panel_border", palette.get("border", "#3c3c3c")), 0.48)

    if not same_splash:
        try:
            overlay.configure(bg=overlay_bg, highlightbackground=overlay_bg, highlightcolor=overlay_bg)
        except Exception:
            pass
        try:
            card.configure(bg=card_bg, highlightbackground=border_color, highlightcolor=border_color)
        except Exception:
            pass
        try:
            title_lbl.configure(text=title_text, bg=card_bg, fg=accent)
        except Exception:
            pass
        try:
            body_lbl.configure(text=body_text, bg=card_bg, fg=fg)
        except Exception:
            pass

    try:
        if show_progress:
            if progress is not None:
                try:
                    if bool(getattr(host, "_campaign_detect_splash_progress_indeterminate", False)):
                        progressbar.stop()
                except Exception:
                    pass
                host._campaign_detect_splash_progress_indeterminate = False
                value = max(0.0, min(100.0, float(progress)))
                progressbar.configure(mode="determinate", maximum=100.0, value=value)
                host._campaign_detect_splash_progress_value = value
            else:
                progressbar.configure(mode="indeterminate", maximum=100.0)
                if not bool(getattr(host, "_campaign_detect_splash_progress_indeterminate", False)):
                    try:
                        progressbar.start(12)
                    except Exception:
                        pass
                host._campaign_detect_splash_progress_indeterminate = True
            if not str(progressbar.winfo_manager()):
                progressbar.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        else:
            try:
                if bool(getattr(host, "_campaign_detect_splash_progress_indeterminate", False)):
                    progressbar.stop()
            except Exception:
                pass
            host._campaign_detect_splash_progress_indeterminate = False
            if str(progressbar.winfo_manager()):
                progressbar.grid_remove()
    except Exception:
        pass

    if not same_splash:
        try:
            if return_btn is not None:
                if show_return:
                    if not str(return_btn.winfo_manager()):
                        return_btn.grid(row=3, column=0, sticky="w", pady=(14, 0))
                elif str(return_btn.winfo_manager()):
                    return_btn.grid_remove()
        except Exception:
            pass

        if not splash_already_placed:
            try:
                overlay.place(in_=surface, relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
                overlay.lift()
            except Exception:
                pass
        else:
            try:
                overlay.lift()
                overlay.after_idle(overlay.lift)
            except Exception:
                pass

    host._campaign_detect_splash_visible = True
    host._campaign_detect_splash_title_text = title_text
    host._campaign_detect_splash_key = splash_key
    if not same_splash:
        try:
            host._refresh_campaign_step3_navigation_visibility()
        except Exception:
            pass


def hide_campaign_detect_splash(host) -> None:
    overlay = getattr(host, "campaign_detect_splash_overlay", None)
    progressbar = getattr(host, "campaign_detect_splash_progress", None)
    if progressbar is not None:
        try:
            if bool(getattr(host, "_campaign_detect_splash_progress_indeterminate", False)):
                progressbar.stop()
        except Exception:
            pass
    host._campaign_detect_splash_progress_indeterminate = False
    if overlay is None:
        host._campaign_detect_splash_visible = False
        host._campaign_detect_splash_title_text = ""
        host._campaign_detect_splash_key = None
        host._campaign_detect_splash_force_root_surface = False
        return
    try:
        overlay.place_forget()
    except Exception:
        pass
    host._campaign_detect_splash_visible = False
    host._campaign_detect_splash_title_text = ""
    host._campaign_detect_splash_key = None
    host._campaign_detect_splash_force_root_surface = False
    try:
        host._refresh_campaign_step3_navigation_visibility()
    except Exception:
        pass


def set_extraction_status(host, text: str, tone: str = "neutral") -> None:
    label = getattr(host, "ext_status", None)
    if not host._set_inline_status_label_state(label, text=text, tone=tone, emphasis=True):
        host._set_themed_label_state(label, text=text, tone=tone, emphasis=True)
    try:
        if bool(getattr(host, "_campaign_detect_splash_visible", False)):
            title = str(getattr(host, "_campaign_detect_splash_title_text", "") or "").strip()
            tone_key = str(tone or "neutral").strip().lower()
            if tone_key == "error":
                title = "Nie udało się przygotować tablic dla Z3"
            elif tone_key == "success":
                title = "Wyodrębnione tablice do pracy nad znakami"
            elif not title:
                title = "Przygotowuję wyodrębnione tablice dla Z3"

            progress_value = None
            if tone_key != "error":
                try:
                    progress_value = float(getattr(host, "ext_progress", None).cget("value") or 0.0)
                except Exception:
                    progress_value = getattr(host, "_campaign_detect_splash_progress_value", None)

            host._show_campaign_detect_splash(
                title=title,
                body=str(text or "").strip(),
                tone=tone_key,
                progress=progress_value,
                show_progress=(tone_key != "error"),
                show_return=(tone_key == "error"),
            )
    except Exception:
        pass


def set_source_binding_status(host, text: str, tone: str = "warning") -> None:
    frame = getattr(host, "source_binding_status_frame", None)
    title_label = getattr(host, "source_binding_status_title_lbl", None)
    label = getattr(host, "source_binding_status_lbl", None)
    if frame is None or label is None:
        return

    has_text = bool(str(text or "").strip())
    tone_key = str(tone or "").strip().lower()
    palette = getattr(host.app, "palette", {})
    base_bg = palette.get("panel_alt", palette.get("panel", "#252526"))
    default_fg = palette.get("fg", "#f3f3f3")
    host._source_binding_status_tone = tone_key

    if tone_key == "error":
        border_color = palette.get("error", "#e74c3c")
        frame_bg = blend_hex_colors(border_color, base_bg, 0.84)
        title_fg = border_color
        text_fg = default_fg
    else:
        border_color = palette.get("success", "#2ecc71")
        frame_bg = palette.get("surface_success", blend_hex_colors(border_color, base_bg, 0.84))
        title_fg = border_color
        text_fg = default_fg

    try:
        if has_text:
            if not str(frame.winfo_manager()):
                frame.pack(fill=tk.X, pady=(0, 10))
        else:
            if str(frame.winfo_manager()):
                frame.pack_forget()
    except Exception:
        pass

    try:
        frame.configure(bg=frame_bg, highlightbackground=border_color, highlightcolor=border_color)
    except Exception:
        pass
    try:
        if title_label is not None:
            title_label.configure(bg=frame_bg, fg=title_fg)
    except Exception:
        pass
    try:
        label.configure(
            text=text,
            bg=frame_bg,
            fg=text_fg,
            font=("Segoe UI", 9),
        )
    except Exception:
        if not host._set_inline_status_label_state(label, text=text, tone=tone, emphasis=False):
            host._set_themed_label_state(label, text=text, tone=tone, emphasis=False)


def refresh_continue_source_summary(host) -> None:
    if not hasattr(host, "extract_continue_xml_value_lbl"):
        return

    xml_raw = str(host.xml_path_var.get() if hasattr(host, "xml_path_var") else "").strip()
    images_raw = str(host.images_dir_var.get() if hasattr(host, "images_dir_var") else "").strip()
    has_source = bool(xml_raw or images_raw)
    xml_text = xml_raw or "Po użyciu runu Z2 pojawi się ścieżka annotations.xml."
    images_text = images_raw or "Po użyciu runu Z2 pojawi się katalog obrazów źródłowych."
    tone = "success" if bool(getattr(host, "_extract_last_source_binding_result", {}).get("ok")) else ("muted" if has_source else "warning")

    for label, text in (
        (getattr(host, "extract_continue_xml_value_lbl", None), xml_text),
        (getattr(host, "extract_continue_images_value_lbl", None), images_text),
    ):
        if label is None:
            continue
        try:
            host._set_inline_status_label_state(label, text=text, tone=tone, emphasis=False)
        except Exception:
            pass


def refresh_extract_action_state(host, *, lightweight: bool = False) -> None:
    btn = getattr(host, "btn_extract", None)
    if btn is None:
        return

    source_ready = bool(getattr(host, "_extract_last_source_binding_result", {}).get("ok"))
    if lightweight:
        manifest_state = host._get_extract_preview_manifest_state()
        technical_preview_ready = bool(
            int(manifest_state.get("plate_count", 0) or 0) > 0
            and bool(manifest_state.get("source_matches", False))
        )
        preview_ready = bool(manifest_state.get("ready", False))
    else:
        technical_preview_ready = bool(
            host._is_usable_step3_preview_dir(
                host.preview_dir_var.get() if hasattr(host, "preview_dir_var") else "",
                require_plates=True,
            )
            and host._preview_matches_current_extract_source(
                host.preview_dir_var.get() if hasattr(host, "preview_dir_var") else ""
            )
        )
        preview_ready = host._is_extract_preview_ready()
    if bool(getattr(host, "is_processing", False)):
        text = "Wyodrębnianie w toku"
        state = tk.DISABLED
    elif technical_preview_ready and host._campaign_preview_is_below_min_extracted_plate_count():
        text = "Za mało tablic"
        state = tk.DISABLED
        try:
            host._set_subtab_state(host.tab_detect, "disabled")
            host._set_button_state("btn_to_detect", False)
        except Exception:
            pass
    elif preview_ready:
        text = "Tablice wyodrębnione"
        state = tk.DISABLED
        try:
            host._set_subtab_state(host.tab_detect, "normal")
            host._set_button_state("btn_to_detect", True)
        except Exception:
            pass
    else:
        text = "Wyodrębnij tablice do PZ2"
        state = tk.NORMAL if source_ready else tk.DISABLED
        try:
            host._set_subtab_state(host.tab_detect, "disabled")
            host._set_button_state("btn_to_detect", False)
        except Exception:
            pass

    try:
        btn.config(text=text, state=state)
    except Exception:
        pass
    try:
        host._refresh_extract_continue_action_visibility()
    except Exception:
        pass


def refresh_extract_continue_action_visibility(host) -> None:
    route = host._get_extract_entry_mode() if hasattr(host, "_get_extract_entry_mode") else ""
    continue_mode = bool(route == "continue")

    cancel_btn = getattr(host, "btn_cancel_continue_extract", None)
    if cancel_btn is not None:
        try:
            if continue_mode:
                if not str(cancel_btn.winfo_manager()):
                    cancel_btn.pack(side=tk.LEFT, padx=(8, 0))
            elif str(cancel_btn.winfo_manager()):
                cancel_btn.pack_forget()
        except Exception:
            pass

    stop_btn = getattr(host, "btn_ext_stop", None)
    if stop_btn is not None:
        try:
            if continue_mode:
                if str(stop_btn.winfo_manager()):
                    stop_btn.pack_forget()
            elif not str(stop_btn.winfo_manager()):
                stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        except Exception:
            pass


def set_widget_state(host, widget, state: str) -> None:
    if widget is None:
        return
    try:
        widget.config(state=state)
    except Exception as e:
        logger.debug(f"Nie udało się ustawić stanu widgetu: {e}")


def update_step3_source_path_lock(host) -> None:
    for attr_name in ("xml_path_entry", "images_dir_entry"):
        widget = getattr(host, attr_name, None)
        if widget is None:
            continue
        try:
            widget.config(state="normal")
        except Exception as e:
            logger.debug(f"Nie udało się ustawić stanu {attr_name}: {e}")

    for attr_name in (
        "xml_path_browse_btn",
        "images_dir_browse_btn",
        "annotation_run_browse_btn",
        "extract_use_z2_source_btn",
    ):
        widget = getattr(host, attr_name, None)
        if widget is None:
            continue
        try:
            widget.config(state="normal")
        except Exception as e:
            logger.debug(f"Nie udało się ustawić stanu {attr_name}: {e}")

    annotation_run_entry = getattr(host, "annotation_run_dir_entry", None)
    if annotation_run_entry is not None:
        try:
            annotation_run_entry.config(state="readonly")
        except Exception as e:
            logger.debug(f"Nie udało się ustawić stanu annotation_run_dir_entry: {e}")


def bind_source_path_watchers(host) -> None:
    if getattr(host, "_source_binding_watchers_bound", False):
        return

    host._source_binding_watchers_bound = True
    for var in (host.xml_path_var, host.images_dir_var):
        try:
            var.trace_add("write", host._on_source_path_var_changed)
        except Exception as e:
            logger.debug(f"Nie udało się podpiąć watcherów źródeł Z3/PZ1: {e}")


def on_source_path_var_changed(host, *_args) -> None:
    if getattr(host, "_source_binding_sync_in_progress", False):
        return
    host._schedule_source_binding_refresh()


def schedule_source_binding_refresh(host, delay_ms: int = 150) -> None:
    if not hasattr(host, "frame") or host.frame is None:
        return

    if not bool(host.is_startup_ui_ready()):
        delay_ms = max(int(delay_ms or 0), 900)

    pending = getattr(host, "_source_binding_after_id", None)
    if pending:
        try:
            host.frame.after_cancel(pending)
        except Exception:
            pass

    host._source_binding_after_id = host.frame.after(delay_ms, host._refresh_source_binding_status)


def format_extract_cut_plan(host, payload: dict | None = None) -> str:
    data = dict(payload or getattr(host, "_extract_last_source_binding_result", {}) or {})
    xml_images_total = int(data.get("xml_images_total") or data.get("total") or 0)
    if xml_images_total <= 0 and not data.get("plate_count"):
        return ""

    plate_count = int(data.get("plate_count") or 0)
    images_with_plates = int(data.get("images_with_plates") or 0)
    if plate_count <= 0:
        return (
            f"Plan wyodrębniania: 0 tablic. XML obejmuje {xml_images_total} obrazów, "
            "ale nie zawiera poprawnych polygonów plate do wyodrębnienia."
        )
    return (
        f"Plan wyodrębniania: {plate_count} tablic z {images_with_plates} obrazów. "
        f"XML obejmuje łącznie {xml_images_total} obrazów."
    )


def append_extract_cut_plan(host, message: str, payload: dict | None = None) -> str:
    plan = host._format_extract_cut_plan(payload)
    if not plan:
        return str(message or "")
    base = str(message or "").strip()
    if plan in base:
        return base
    return f"{base}\n\n{plan}" if base else plan


def format_extract_source_datetime(path_like) -> str:
    raw = str(path_like or "").strip()
    if not raw:
        return "Brak przejętego runu"
    try:
        path = Path(raw)
        if not path.exists():
            return "Nie znaleziono katalogu runu"
        stamp = path.stat().st_ctime
        return _datetime.datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "Nie udało się odczytać daty"


def format_extract_source_folder_name(path_like, fallback: str = "Brak danych") -> str:
    raw = str(path_like or "").strip()
    if not raw:
        return fallback
    try:
        path = Path(raw)
        name = path.name or raw
        parent = path.parent.name
        return f"{parent}/{name}" if parent else name
    except Exception:
        return raw


def build_continue_extract_summary_rows(host, output_text: str) -> list[tuple[str, str, str]]:
    self = host
    binding = dict(getattr(self, "_extract_last_source_binding_result", {}) or {})
    run_raw = str(self.annotation_run_dir_var.get() if hasattr(self, "annotation_run_dir_var") else "").strip()
    images_raw = str(self.images_dir_var.get() if hasattr(self, "images_dir_var") else "").strip()

    try:
        run_name = Path(run_raw).name if run_raw else ""
    except Exception:
        run_name = run_raw
    run_text = run_name or "Nie przejęto runu z Z2"

    matched = int(binding.get("matched", 0) or 0)
    total = int(binding.get("total", 0) or binding.get("xml_images_total", 0) or 0)
    if total > 0 and matched > 0:
        images_count = f"zgodne zdjęcia: {matched}; obrazy w XML: {total}"
    elif total > 0:
        images_count = f"{total} zdjęć w XML"
    else:
        images_count = "Brak potwierdzonej liczby zdjęć"

    plate_count = int(binding.get("plate_count", 0) or 0)
    plates_text = f"{plate_count} tablic do wyodrębnienia" if plate_count > 0 else "Brak potwierdzonych tablic do wyodrębnienia"

    return [
        ("Run anotacji", run_text, "Źródło przejęte z Z2"),
        ("Data utworzenia", format_extract_source_datetime(run_raw), "Data katalogu runu"),
        (
            "Katalog zdjęć",
            format_extract_source_folder_name(images_raw, "Brak skorelowanego katalogu zdjęć"),
            "Obrazy zgodne z XML",
        ),
        ("Liczba zdjęć", images_count, "Pula sprawdzona względem XML"),
        ("Liczba anotacji", plates_text, "Tablice do przygotowania w PZ2"),
        ("Wynik", output_text, "Docelowy zestaw wyodrębnionych tablic PZ2"),
    ]


def set_extract_start_summary_rows(host, rows: list[tuple[str, ...]]) -> None:
    self = host
    row_widgets = list(getattr(self, "_extract_start_summary_rows", []) or [])
    if not row_widgets:
        return

    summary_frame = getattr(self, "extract_start_summary_frame", None)
    if summary_frame is not None:
        try:
            if not str(summary_frame.winfo_manager()):
                summary_frame.pack(fill=tk.X, pady=(0, 12))
        except Exception:
            pass

    header_widgets = list(getattr(self, "_extract_start_summary_header_widgets", []) or [])
    for header_widget, header_text in zip(header_widgets, ("Pozycja", "Wartość", "Znaczenie")):
        try:
            header_widget.grid()
            header_widget.configure(text=header_text)
        except Exception:
            pass

    for index, widgets in enumerate(row_widgets):
        title_lbl = widgets[0] if len(widgets) > 0 else None
        value_lbl = widgets[1] if len(widgets) > 1 else None
        note_lbl = widgets[2] if len(widgets) > 2 else None
        visible = index < len(rows)
        row_values = tuple(rows[index]) if visible else ("", "", "")
        title_text = row_values[0] if len(row_values) > 0 else ""
        value_text = row_values[1] if len(row_values) > 1 else ""
        note_text = row_values[2] if len(row_values) > 2 else ""
        for widget in (title_lbl, value_lbl, note_lbl):
            if widget is None:
                continue
            try:
                if visible:
                    widget.grid()
                else:
                    widget.grid_remove()
            except Exception:
                pass
        try:
            title_lbl.configure(text=str(title_text or ""))
        except Exception:
            pass
        try:
            value_lbl.configure(text=str(value_text or ""))
        except Exception:
            pass
        try:
            note_lbl.configure(text=str(note_text or ""))
        except Exception:
            pass


def bind_extract_entry_card(host, card_widget, mode: str) -> None:
    def _select(_event=None, selected_mode=mode):
        host._handle_extract_entry_selection(selected_mode)
        return "break"

    def _hover(enabled: bool):
        host._extract_entry_hover_mode = mode if enabled else None
        host._refresh_extract_entry_cards()

    for widget in card_widget.winfo_children():
        host._bind_extract_entry_card(widget, mode)

    try:
        card_widget.bind("<Button-1>", _select, add="+")
        card_widget.bind("<Enter>", lambda _event: _hover(True), add="+")
        card_widget.bind("<Leave>", lambda _event: _hover(False), add="+")
    except Exception:
        pass


def refresh_extract_entry_cards(host) -> None:
    cards = getattr(host, "_extract_entry_cards", {}) or {}
    if not cards:
        return

    palette = getattr(host.app, "palette", {})
    selected_mode = host._get_extract_entry_mode()
    hover_mode = getattr(host, "_extract_entry_hover_mode", None)
    panel_alt = palette.get("panel_alt", palette.get("panel", "#252526"))
    hover_bg = palette.get("button_hover", panel_alt)
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    success = palette.get("success", fg)
    selected_bg = blend_hex_colors(panel_alt, hover_bg, 0.42)

    try:
        preview_ready = bool(host._is_extract_preview_ready_fast())
    except Exception:
        preview_ready = False
    source_ready = bool(getattr(host, "_extract_last_source_binding_result", {}).get("ok"))

    descriptions = {
        "manual": (
            "Podaj annotations.xml oraz folder oryginalnych obrazów, z których mam "
            "wyodrębnić tablice. To dobry tor dla importu z zewnątrz."
        ),
        "continue": (
            "Tablice są już wyodrębnione. Otwórz kafel, aby zobaczyć tabelę "
            "przejętego runu i wynik wyodrębniania."
            if preview_ready
            else (
                "Przejmij potwierdzony run Z2. PZ1 pokaże tabelę źródła i pozwoli "
                "wyodrębnić tablice do PZ2."
                if source_ready
                else (
                    "Przejmij aktywny albo ostatni run anotacji Z2. "
                    "W tym torze nie wskazujesz ręcznie folderu runu, XML ani obrazów."
                )
            )
        ),
    }

    for mode, widgets in cards.items():
        is_selected = mode == selected_mode
        is_hovered = mode == hover_mode
        card_bg = selected_bg if is_selected else (hover_bg if is_hovered else panel_alt)
        desc_fg = fg if is_selected else muted

        for widget_name in ("frame", "badge", "title", "desc"):
            widget = widgets.get(widget_name)
            if widget is None:
                continue
            try:
                widget.configure(bg=card_bg, highlightbackground=border, highlightcolor=border)
            except Exception:
                pass

        try:
            widgets["title"].configure(fg=fg)
        except Exception:
            pass
        try:
            widgets["desc"].configure(fg=desc_fg, text=descriptions.get(mode, str(widgets["desc"].cget("text") or "")))
        except Exception:
            pass
        try:
            badge = widgets.get("badge")
            if badge is None:
                continue
            if mode == "continue" and preview_ready:
                badge.configure(text="TABLICE WYODRĘBNIONE", fg=success)
            elif mode == "continue" and source_ready:
                badge.configure(text="ŹRÓDŁO OK", fg=success)
            elif mode == "continue":
                badge.configure(text="WEJŚCIE Z Z2", fg=muted)
            elif mode == "manual":
                badge.configure(text="XML + OBRAZY", fg=muted)
            else:
                badge.configure(text="WYBIERZ", fg=muted)
        except Exception:
            pass


def normalize_extract_entry_mode(host, value: str | None = None) -> str:
    normalized = str(
        value if value is not None else (host.extract_entry_mode_var.get() if hasattr(host, "extract_entry_mode_var") else "")
        or ""
    ).strip().lower()
    if not normalized:
        return ""
    return normalized if normalized in {"continue", "manual"} else ""


def normalize_extract_workflow_step(step: str | None = None) -> str:
    normalized = str(step or "").strip().lower()
    return normalized if normalized in {"entry", "source", "start"} else "entry"


def coerce_extract_workflow_step(host, step: str | None = None, *, lightweight: bool = False) -> str:
    route = host._get_extract_entry_mode()
    current = host._normalize_extract_workflow_step(
        step if step is not None else getattr(host, "_extract_workflow_step", "entry")
    )
    try:
        has_preview = bool(
            host._is_extract_preview_ready_fast()
            if lightweight
            else host._is_extract_preview_ready()
        )
    except Exception:
        has_preview = bool(str(host.preview_dir_var.get() or "").strip())
    source_ready = bool(getattr(host, "_extract_last_source_binding_result", {}).get("ok"))

    if not route:
        return "entry"
    if current == "entry":
        return "entry"
    if route == "continue" and current == "start":
        return "start"
    if current == "start" and not (source_ready or has_preview):
        return "source"
    return current if current in {"source", "start"} else "source"


def get_extract_workflow_step(host, *, lightweight: bool = False) -> str:
    current = host._coerce_extract_workflow_step(lightweight=lightweight)
    if not lightweight:
        host._extract_workflow_step = current
    return current


def set_extract_workflow_step(host, step: str, *, refresh: bool = True) -> None:
    host._extract_workflow_step = host._coerce_extract_workflow_step(step)
    try:
        host._persist_step3_extract_state()
    except Exception:
        pass
    if refresh:
        host._refresh_extract_workflow_ui()


def persist_step3_extract_state(host) -> None:
    if not CAMPAIGN.get_active_project_name():
        return

    try:
        entry_mode = host._get_extract_entry_mode()
    except Exception:
        entry_mode = ""

    try:
        workflow_step = host._get_extract_workflow_step()
    except Exception:
        workflow_step = "entry"

    try:
        annotation_run_dir = str(host.annotation_run_dir_var.get() or "").strip()
    except Exception:
        annotation_run_dir = ""

    try:
        xml_path = str(host.xml_path_var.get() or "").strip()
    except Exception:
        xml_path = ""

    try:
        images_dir = str(host.images_dir_var.get() or "").strip()
    except Exception:
        images_dir = ""

    try:
        CAMPAIGN.set_step3_extract_state(
            entry_mode=entry_mode,
            workflow_step=workflow_step,
            annotation_run_dir=annotation_run_dir,
            xml_path=xml_path,
            images_dir=images_dir,
        )
    except Exception:
        pass

    try:
        host._sync_campaign_step3_artifact_registry(
            entry_mode=entry_mode,
            workflow_step=workflow_step,
            annotation_run_dir=annotation_run_dir,
            xml_path=xml_path,
            images_dir=images_dir,
        )
    except Exception:
        pass


def sync_campaign_step3_artifact_registry(
    host,
    *,
    entry_mode: str = "",
    workflow_step: str = "entry",
    annotation_run_dir: str = "",
    xml_path: str = "",
    images_dir: str = "",
) -> None:
    if not CAMPAIGN.get_active_project_name():
        return

    try:
        iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        iteration_num = 1

    package_images_dir = CAMPAIGN.get_iteration_image_source_dir(iteration_num) or CAMPAIGN.get_master_pool_dir() or CAMPAIGN.get_iteration_raw_dir(iteration_num)
    if package_images_dir is None:
        return

    updates = {
        "step3_extract_source": {
            "entry_mode": str(entry_mode or "").strip(),
            "workflow_step": str(workflow_step or "entry").strip() or "entry",
            "annotation_run_dir": str(annotation_run_dir or "").strip(),
            "annotation_run_token": CAMPAIGN._build_registry_path_token(annotation_run_dir),
            "xml_path": str(xml_path or "").strip(),
            "xml_token": CAMPAIGN._build_registry_path_token(xml_path),
            "images_dir": str(images_dir or "").strip(),
            "images_token": CAMPAIGN._build_registry_path_token(images_dir),
        }
    }
    try:
        CAMPAIGN.upsert_iteration_artifact_bundle(
            images_dir=package_images_dir,
            iteration_num=iteration_num,
            updates=updates,
        )
    except Exception:
        pass


def sync_campaign_step3_preview_artifact_registry(host, preview_dir) -> None:
    if not CAMPAIGN.get_active_project_name():
        return

    try:
        iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        iteration_num = 1

    package_images_dir = CAMPAIGN.get_iteration_image_source_dir(iteration_num) or CAMPAIGN.get_master_pool_dir() or CAMPAIGN.get_iteration_raw_dir(iteration_num)
    if package_images_dir is None:
        return

    preview_dir_raw = str(preview_dir or "").strip()
    preview_path = None
    metadata_path = None
    images_dir = None
    plate_count = 0
    if preview_dir_raw:
        try:
            candidate = Path(preview_dir_raw)
        except Exception:
            candidate = None
        if (
            candidate is not None
            and candidate.exists()
            and candidate.is_dir()
            and (candidate / "metadata.json").exists()
            and (candidate / "images").exists()
        ):
            preview_path = candidate
            metadata_path = candidate / "metadata.json"
            images_dir = candidate / "images"
            try:
                plate_count = int(host._get_preview_dir_plate_count(candidate) or 0)
            except Exception:
                plate_count = 0

    updates = {
        "step3_preview_source": {
            "preview_dir": str(preview_path.resolve()) if preview_path is not None else "",
            "preview_token": CAMPAIGN._build_registry_path_token(preview_path),
            "metadata_path": str(metadata_path.resolve()) if metadata_path is not None else "",
            "metadata_token": CAMPAIGN._build_registry_path_token(metadata_path),
            "images_dir": str(images_dir.resolve()) if images_dir is not None else "",
            "images_token": CAMPAIGN._build_registry_path_token(images_dir),
            "plate_count": int(plate_count or 0),
        }
    }

    try:
        CAMPAIGN.upsert_iteration_artifact_bundle(
            images_dir=package_images_dir,
            iteration_num=iteration_num,
            updates=updates,
        )
    except Exception:
        pass


def restore_step3_extract_state_from_project(host) -> None:
    if not CAMPAIGN.get_active_project_name():
        return

    try:
        saved_state = CAMPAIGN.get_step3_extract_state()
    except Exception:
        saved_state = {}

    saved_mode = host._normalize_extract_entry_mode(saved_state.get("entry_mode"))
    saved_step = host._normalize_extract_workflow_step(saved_state.get("workflow_step"))
    saved_run_dir = str(saved_state.get("annotation_run_dir", "") or "").strip()
    saved_xml = str(saved_state.get("xml_path", "") or "").strip()
    saved_images = str(saved_state.get("images_dir", "") or "").strip()

    if not saved_mode:
        try:
            host.extract_entry_mode_var.set("")
        except Exception:
            pass
        host._extract_workflow_step = "entry"
        return

    if saved_mode == "continue":
        restored = False
        try:
            restored = host._use_z2_source_on_demand(
                notify_on_failure=False,
                advance_to_start=(saved_step == "start"),
            )
        except Exception:
            restored = False

        if restored:
            if saved_step != "start":
                host._extract_workflow_step = host._coerce_extract_workflow_step(saved_step)
            try:
                host._refresh_extract_workflow_ui()
            except Exception:
                pass
            return

    host._source_binding_sync_in_progress = True
    try:
        host.annotation_run_dir_var.set(saved_run_dir)
        host.xml_path_var.set(saved_xml)
        host.images_dir_var.set(saved_images)
    finally:
        host._source_binding_sync_in_progress = False

    try:
        host._set_extract_entry_mode(saved_mode, persist=False)
    except Exception:
        host._extract_workflow_step = "source" if saved_mode else "entry"

    try:
        validation = host._refresh_source_binding_status(allow_autofind=True)
    except Exception:
        validation = {"ok": False}

    target_step = saved_step
    if target_step == "start" and not bool(validation.get("ok")) and not bool(str(host.preview_dir_var.get() or "").strip()):
        target_step = "source"

    host._extract_workflow_step = host._coerce_extract_workflow_step(target_step)
    try:
        host._refresh_extract_workflow_ui()
    except Exception:
        pass


def paths_equivalent(left, right) -> bool:
    if not left or not right:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except Exception:
        try:
            return str(Path(left)) == str(Path(right))
        except Exception:
            return False


def resolve_existing_annotation_run_dir(candidate) -> Path | None:
    raw = str(candidate or "").strip()
    if not raw:
        return None

    try:
        path = Path(raw)
    except Exception:
        return None

    if path.suffix.lower() == ".xml":
        path = path.parent

    try:
        if not path.exists() or not path.is_dir():
            return None
    except Exception:
        return None

    return path if (path / "annotations.xml").exists() else None


def annotation_run_manifest_path(run_dir: Path) -> Path:
    return Path(run_dir) / "run_manifest.json"


def load_annotation_run_manifest(host, run_dir: Path) -> dict:
    manifest_path = host._annotation_run_manifest_path(run_dir)
    payload = PROJECT_CACHE.load_json(manifest_path, default={})
    return payload if isinstance(payload, dict) else {}


def get_campaign_iteration_artifact_bundle() -> dict:
    if not CAMPAIGN.get_active_project_name():
        return {}

    try:
        iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        iteration_num = 1

    package_images_dir = CAMPAIGN.get_iteration_image_source_dir(iteration_num) or CAMPAIGN.get_master_pool_dir() or CAMPAIGN.get_iteration_raw_dir(iteration_num)
    if package_images_dir is None:
        return {}

    try:
        return dict(
            CAMPAIGN.get_iteration_artifact_bundle(
                images_dir=package_images_dir,
                iteration_num=iteration_num,
            ) or {}
        )
    except Exception:
        return {}


def get_registry_preferred_step3_source_candidate(host) -> dict | None:
    bundle = host._get_campaign_iteration_artifact_bundle()
    if not bundle:
        return None

    candidate_specs = [
        (
            "char_effective_source",
            dict(bundle.get("char_effective_source") or {}),
            "gotowe źródło znaków z rejestru",
            "run_dir",
        ),
        (
            "step3_extract_source",
            dict(bundle.get("step3_extract_source") or {}),
            "zapisane źródło etapu Z3",
            "annotation_run_dir",
        ),
        (
            "plate_source",
            dict(bundle.get("plate_source") or {}),
            "źródło tablic z rejestru projektu",
            "run_dir",
        ),
    ]

    for _scope_name, raw_entry, label, run_key in candidate_specs:
        run_dir = str(raw_entry.get(run_key) or "").strip()
        xml_path = str(raw_entry.get("xml_path") or "").strip()
        images_dir = str(raw_entry.get("images_dir") or "").strip()
        if not any((run_dir, xml_path, images_dir)):
            continue
        try:
            run_name = Path(run_dir).name if run_dir else ""
        except Exception:
            run_name = ""
        return {
            "run_dir": run_dir,
            "xml_path": xml_path,
            "images_dir": images_dir,
            "label": f"{label}: {run_name}" if run_name else label,
        }

    return None


def get_annotation_tab(host):
    try:
        tab = getattr(host.app, "tabs", {}).get("annotation")
        # Nie dotykamy lazy-placeholdera Z2. Każdy getattr na nim uruchamia
        # pełne ładowanie Z2, co blokowało pierwszy render Z3/PZ1.
        if isinstance(tab, _LazyNotebookTab):
            return None
        return tab
    except Exception:
        return None


def get_annotation_run_input_dir(host, run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None

    manifest = host._load_annotation_run_manifest(run_dir)
    input_dir_raw = str(manifest.get("input_dir") or "").strip()
    if input_dir_raw:
        try:
            input_dir = Path(input_dir_raw)
            if input_dir.exists() and input_dir.is_dir():
                return input_dir
        except Exception:
            pass

    annotation_tab = host._get_annotation_tab()
    if annotation_tab is not None:
        try:
            current_run = getattr(annotation_tab, "current_annotation_run_dir", None)
            if host._paths_equivalent(run_dir, current_run):
                current_input = getattr(annotation_tab, "current_input_dir", None)
                if current_input is not None and Path(current_input).exists():
                    return Path(current_input)
        except Exception:
            pass

    pending_images = str(getattr(host, "_pending_z2_source", {}).get("images_dir") or "").strip()
    pending_run = str(getattr(host, "_pending_z2_source", {}).get("run_dir") or "").strip()
    if pending_images and pending_run and host._paths_equivalent(run_dir, pending_run):
        try:
            pending_input = Path(pending_images)
            if pending_input.exists() and pending_input.is_dir():
                return pending_input
        except Exception:
            pass

    return None


def apply_annotation_run_source(host, run_dir: Path | None, *, refresh_status: bool = True) -> bool:
    safe_run_dir = host._resolve_existing_annotation_run_dir(run_dir)
    if safe_run_dir is None:
        return False

    input_dir = host._get_annotation_run_input_dir(safe_run_dir)
    host._source_binding_sync_in_progress = True
    try:
        host.annotation_run_dir_var.set(str(safe_run_dir))
        host.xml_path_var.set(str(safe_run_dir / "annotations.xml"))
        if input_dir is not None:
            host.images_dir_var.set(str(input_dir))
    finally:
        host._source_binding_sync_in_progress = False

    try:
        host._force_save_all()
    except Exception:
        pass

    if refresh_status:
        host._schedule_source_binding_refresh(delay_ms=0)
    host._refresh_extract_workflow_ui()
    return True


def set_pending_z2_annotation_source(host, *, xml_path: str = "", images_dir: str = "", run_dir: str = "") -> None:
    candidate_run = str(run_dir or "").strip()
    candidate_xml = str(xml_path or "").strip()
    candidate_images = str(images_dir or "").strip()

    if not candidate_run and candidate_xml:
        try:
            candidate_run = str(Path(candidate_xml).parent)
        except Exception:
            candidate_run = ""

    host._pending_z2_source = {
        "run_dir": candidate_run,
        "xml_path": candidate_xml,
        "images_dir": candidate_images,
    }
    host._refresh_extract_workflow_ui()


def pick_annotation_run_dir(host) -> None:
    initial_dir = ""
    try:
        initial_dir = str(CAMPAIGN.get_dir("auto_ann") or "")
    except Exception:
        initial_dir = ""
    if not initial_dir:
        initial_dir = str(Path(CONFIG.get_auto_annotations_dir("plate")).absolute())

    selected = filedialog.askdirectory(
        initialdir=initial_dir,
        title="Wskaż folder runu anotacji z annotations.xml",
    )
    if not selected:
        return

    if not host._apply_annotation_run_source(Path(selected)):
        messagebox.showwarning(
            "Błędny run anotacji",
            "Wybrany folder nie zawiera pliku annotations.xml, więc nie może być źródłem dla PZ1.",
        )


def pick_xml_file(host) -> None:
    initial_dir = ""
    try:
        initial_dir = str(CAMPAIGN.get_dir("auto_ann") or "")
    except Exception:
        initial_dir = ""
    if not initial_dir:
        initial_dir = str(Path(CONFIG.get_auto_annotations_dir("plate")).absolute())
    path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Wybierz annotations.xml dla tego samego katalogu obrazów",
        filetypes=[("XML", "*.xml")]
    )
    if path:
        host.xml_path_var.set(path)


def pick_images_dir(host) -> None:
    initial_dir = ""
    try:
        initial_dir = str(CAMPAIGN.get_dir("raw") or "")
    except Exception:
        initial_dir = ""
    if not initial_dir:
        initial_dir = str(Path(CONFIG.DIR_1_RAW).absolute())
    path = filedialog.askdirectory(
        initialdir=initial_dir,
        title="Wybierz katalog obrazów powiązany z annotations.xml"
    )
    if path:
        host.images_dir_var.set(path)


def get_preferred_z2_source_candidate(host) -> dict | None:
    pending = dict(getattr(host, "_pending_z2_source", {}) or {})
    if any(str(pending.get(key) or "").strip() for key in ("run_dir", "xml_path", "images_dir")):
        run_dir = str(pending.get("run_dir") or "").strip()
        label = (
            f"ostatni run anotacji Z2: {Path(run_dir).name}"
            if run_dir else "ostatnie źródło z runu anotacji Z2"
        )
        pending["label"] = label
        return pending

    annotation_tab = host._get_annotation_tab()
    if annotation_tab is not None:
        try:
            getter = getattr(annotation_tab, "_get_preferred_annotation_run_dir", None)
            run_dir = getter(require_xml=True) if callable(getter) else None
        except Exception:
            run_dir = None

        try:
            xml_path = getattr(annotation_tab, "current_annotation_xml_path", None)
            if not xml_path and run_dir is not None:
                xml_path = run_dir / "annotations.xml"
        except Exception:
            xml_path = run_dir / "annotations.xml" if run_dir is not None else None

        try:
            images_dir = getattr(annotation_tab, "current_input_dir", None)
        except Exception:
            images_dir = None

        if run_dir is not None or xml_path is not None or images_dir is not None:
            return {
                "run_dir": str(run_dir or ""),
                "xml_path": str(xml_path or ""),
                "images_dir": str(images_dir or ""),
                "label": (
                    f"aktywny run anotacji Z2: {Path(run_dir).name}"
                    if run_dir is not None else "bieżące źródło z runu anotacji Z2"
                ),
            }

    registry_candidate = host._get_registry_preferred_step3_source_candidate()
    if registry_candidate:
        return registry_candidate

    try:
        stored_source = CAMPAIGN.get_last_plate_manual_source()
    except Exception:
        stored_source = {}

    if stored_source:
        run_dir = str(stored_source.get("source_run_path") or "").strip()
        xml_path = str(stored_source.get("source_xml_path") or "").strip()
        images_dir = str(stored_source.get("source_input_path") or "").strip()
        if run_dir or xml_path or images_dir:
            return {
                "run_dir": run_dir,
                "xml_path": xml_path,
                "images_dir": images_dir,
                "label": (
                    f"ostatnio zapisany run anotacji Z2: {Path(run_dir).name}"
                    if run_dir else "ostatnio zapisane źródło z runu anotacji Z2"
                ),
            }

    return None


def use_z2_source_on_demand(host, *, notify_on_failure: bool = True, advance_to_start: bool = False) -> bool:
    candidate = host._get_preferred_z2_source_candidate()
    if not candidate:
        if notify_on_failure:
            messagebox.showinfo(
                "Brak źródła z Z2",
                "Nie znalazłem gotowego źródła z Z2. Otwórz lub zakończ run anotacji w Z2, "
                "albo wróć do wyboru trybu i wskaż ręcznie annotations.xml oraz zgodny katalog obrazów.",
            )
        return False

    run_dir = host._resolve_existing_annotation_run_dir(candidate.get("run_dir") or candidate.get("xml_path"))
    host._set_extract_entry_mode("continue")
    if run_dir is not None:
        host._apply_annotation_run_source(run_dir, refresh_status=False)

    host._source_binding_sync_in_progress = True
    try:
        if candidate.get("xml_path"):
            host.xml_path_var.set(str(candidate.get("xml_path")))
        if candidate.get("images_dir"):
            host.images_dir_var.set(str(candidate.get("images_dir")))
        if run_dir is None and candidate.get("run_dir"):
            host.annotation_run_dir_var.set(str(candidate.get("run_dir")))
    finally:
        host._source_binding_sync_in_progress = False

    try:
        host._force_save_all()
    except Exception:
        pass
    validation = host._refresh_source_binding_status(allow_autofind=True)
    success = bool(validation.get("ok"))

    if advance_to_start and success:
        host._set_extract_workflow_step("start")
    elif advance_to_start and not success:
        if notify_on_failure:
            messagebox.showwarning(
                "Nie mogę kontynuować z Z2",
                validation.get("message", "Nie udało się automatycznie potwierdzić źródeł z ostatniego runu anotacji Z2."),
            )
        return False
    else:
        host._refresh_extract_workflow_ui()

    return success


def schedule_extract_workflow_refresh(host, delay_ms: int = 0) -> None:
    after_id = getattr(host, "_extract_workflow_refresh_after_id", None)
    if after_id is not None:
        try:
            host.frame.after_cancel(after_id)
        except Exception:
            pass
        finally:
            host._extract_workflow_refresh_after_id = None

    def _run():
        host._extract_workflow_refresh_after_id = None
        try:
            host._refresh_extract_workflow_ui()
        except Exception as exc:
            logger.debug(f"Nie udało się odświeżyć workflow PZ1: {exc}")

    try:
        if int(delay_ms or 0) > 0:
            host._extract_workflow_refresh_after_id = host.frame.after(int(delay_ms), _run)
        else:
            host._extract_workflow_refresh_after_id = host.frame.after_idle(_run)
    except Exception:
        host._extract_workflow_refresh_after_id = None


def show_extract_completed_modal(host, plate_count: int, run_dir: Path) -> None:
    try:
        parent = host.frame.winfo_toplevel()
    except Exception:
        parent = getattr(host.app, "root", None)

    try:
        in_campaign = bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
        if in_campaign:
            tail = (
                "Zatwierdź powyższą pracę przyciskiem w lewym dolnym rogu. "
                "Wrócisz do pracy bramki T05, gdzie wybierzesz kolejny krok: PZ2."
            )
        else:
            tail = (
                "Przycisk „Wyodrębnij tablice do PZ2” został wyłączony dla tego zestawu. "
                "Po potwierdzeniu otworzę PZ2, gdzie odbywa się dalsza praca na wyodrębnionych tablicach."
            )
        messagebox.showinfo(
            "Wyodrębnianie zakończone",
            (
                "Tablice zostały wyodrębnione poprawnie i zestaw PZ2 jest gotowy.\n\n"
                f"Liczba wyodrębnionych tablic: {int(plate_count or 0)}.\n"
                f"Run PZ2: {Path(run_dir).name}\n\n"
                f"{tail}"
            ),
            parent=parent,
        )
        if not in_campaign:
            host.go_to_substep_2()
    except Exception:
        pass


def _campaign_step3_context_active(host) -> bool:
    try:
        return bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
    except Exception:
        return False


def _get_campaign_tab(host):
    try:
        return getattr(host.app, "tabs", {}).get("campaign")
    except Exception:
        return None


def _return_campaign_step3_to_graph(host) -> None:
    try:
        host._return_to_wizard_for_step3_rework()
        return
    except Exception as exc:
        logger.debug(f"Nie udało się wrócić do grafu po problemie Z3: {exc}")
    campaign_tab = _get_campaign_tab(host)
    if campaign_tab is not None:
        try:
            campaign_tab._return_to_graph_after_tool()
            return
        except Exception as exc:
            logger.debug(f"Nie udało się użyć fallbacku powrotu do grafu: {exc}")
    try:
        if hasattr(host.app, "select_tab"):
            host.app.select_tab("campaign")
    except Exception:
        pass


def show_campaign_extract_failure_modal(
    host,
    *,
    title: str,
    message: str,
    tone: str = "warning",
    default_action: str = "repair",
    allow_retry: bool = False,
) -> None:
    try:
        parent = host.frame.winfo_toplevel()
    except Exception:
        parent = getattr(host.app, "root", None)

    try:
        host._hide_campaign_detect_splash()
    except Exception:
        pass

    try:
        host._set_extraction_status(str(message or title or "Nie udało się przygotować tablic."), tone)
    except Exception:
        pass

    buttons = ["Uzupełnij tablice w Z2"]
    if allow_retry:
        buttons.append("Ponów wyodrębnianie")
    buttons.append("Wróć do grafu")

    default_button = "Uzupełnij tablice w Z2" if default_action == "repair" else "Wróć do grafu"
    if allow_retry and default_action == "retry":
        default_button = "Ponów wyodrębnianie"

    try:
        choice = host.app.themed_message_dialog(
            str(title or "Nie udało się przygotować tablic"),
            str(message or "Proces przygotowania tablic dla pracy nad znakami wymaga decyzji."),
            parent=parent,
            buttons=buttons,
            default_button=default_button,
            tone=tone,
            wraplength=600,
        )
    except Exception:
        try:
            messagebox.showwarning(str(title or "Nie udało się przygotować tablic"), str(message or ""), parent=parent)
        except Exception:
            pass
        choice = default_button

    campaign_tab = _get_campaign_tab(host)
    if choice == "Uzupełnij tablice w Z2" and campaign_tab is not None:
        try:
            campaign_tab._step_open_z2_repair_from_later_stage()
            return
        except Exception as exc:
            logger.debug(f"Nie udało się otworzyć Z2 po problemie wyodrębniania: {exc}")

    if choice == "Ponów wyodrębnianie" and allow_retry:
        try:
            host.frame.after(120, host._run_extraction)
            return
        except Exception as exc:
            logger.debug(f"Nie udało się ponowić wyodrębniania Z3: {exc}")

    _return_campaign_step3_to_graph(host)


def show_campaign_extract_below_minimum_modal(host, plate_count: int, run_dir: Path) -> None:
    min_count = host._get_campaign_step3_min_extracted_plate_count()
    plate_count = max(0, int(plate_count or 0))
    missing = max(0, int(min_count) - plate_count)
    show_campaign_extract_failure_modal(
        host,
        title="Za mało wyodrębnionych tablic",
        message=(
            "Wyodrębnianie zakończyło się, ale wynik jest za mały dla pracy nad znakami.\n\n"
            f"Wyodrębnione tablice: {plate_count}\n"
            f"Minimum do dalszej pracy: {min_count}\n"
            f"Brakuje: {missing}\n"
            f"Zestaw techniczny: {Path(run_dir).name}\n\n"
            "Najbezpieczniej wrócić do Z2 i oznaczyć więcej tablic. Możesz też ponowić wyodrębnianie, "
            "jeżeli źródła zostały właśnie poprawione."
        ),
        tone="warning",
        default_action="repair",
        allow_retry=True,
    )
    return


def format_extract_start_path(value: str, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    try:
        path = Path(raw)
        name = path.name or raw
        parent = path.parent.name
        return f"{parent}/{name}" if parent else name
    except Exception:
        return raw


def refresh_extract_start_summary(host, *, lightweight: bool = False):
    self = host
    if not hasattr(self, "extract_start_xml_value_lbl"):
        return

    xml_path = format_extract_start_path(
        self.xml_path_var.get() if hasattr(self, "xml_path_var") else "",
        "Najpierw potwierdź annotations.xml",
    )
    images_dir = format_extract_start_path(
        self.images_dir_var.get() if hasattr(self, "images_dir_var") else "",
        "Najpierw potwierdź folder obrazów",
    )
    preview_ready = (
        self._is_extract_preview_ready_fast()
        if lightweight
        else self._is_extract_preview_ready()
    )
    try:
        output_root = self._get_step3_chars_root_dir(ensure_exists=False)
        preview_value = format_extract_start_path(str(self.preview_dir_var.get() or ""), "")
        if preview_value and preview_ready:
            output_text = f"Preview: {preview_value}"
        elif preview_value:
            output_text = f"Nowy run w {format_extract_start_path(str(output_root), 'katalogu PZ1')}"
        else:
            output_text = f"Nowy run w {format_extract_start_path(str(output_root), 'katalogu PZ1')}"
    except Exception:
        output_text = "Nowy run wyodrębnionych tablic"

    cut_plan = self._format_extract_cut_plan()
    plates_text = cut_plan.replace("Plan wyodrębniania: ", "", 1).replace("Plan wycinania: ", "", 1) if cut_plan else "Najpierw potwierdź źródła"
    source_ready = bool(getattr(self, "_extract_last_source_binding_result", {}).get("ok"))
    ready_count = self._get_extract_preview_ready_count()
    if preview_ready:
        badge_text = "PZ2 GOTOWE"
        action_note = (
            f"Wyodrębnianie zostało już wykonane poprawnie. Aktywny zestaw PZ2 zawiera {ready_count} tablic. "
            "Przejdź do PZ2 albo zmień źródła, jeśli chcesz świadomie przygotować inny zestaw."
        )
    else:
        badge_text = "ŹRÓDŁA OK" if source_ready else "SPRAWDŹ ŹRÓDŁA"
        action_note = (
            "PZ1 wyodrębni tablice z potwierdzonych ramek/polygonów i przygotuje zestaw do PZ2."
            if source_ready
            else "Wróć do źródeł i potwierdź zgodność XML z katalogiem obrazów, zanim uruchomisz wyodrębnianie."
        )

    route = self._get_extract_entry_mode() if hasattr(self, "_get_extract_entry_mode") else ""
    if route == "continue":
        if preview_ready:
            action_note = (
                "PZ2 jest gotowe do pracy nad znakami. Wybierz „Dalej do PZ2”, aby otworzyć listę tablic "
                "i podgląd bez ponownego wyodrębniania."
            )
        elif source_ready:
            action_note = "Jeśli tabela pokazuje właściwy run, uruchom akcję. Jeśli nie, wybierz „Zrezygnuj i wróć”."
        else:
            action_note = (
                "Nie udało się przejąć kompletnego źródła z Z2. Wróć do wyboru i użyj trybu ręcznego, "
                "jeśli chcesz wskazać annotations.xml oraz katalog obrazów samodzielnie."
            )
        rows = build_continue_extract_summary_rows(self, output_text)
        set_extract_start_summary_rows(self, rows)
        if preview_ready:
            title_text = "PZ2 gotowe. Tablice są już wyodrębnione"
            desc_text = (
                "Poniżej widzisz tabelę kontrolną przejętego runu Z2 i gotowego zestawu wyodrębnionych tablic. "
                "Nie uruchamiaj ponownego wyodrębniania; przejdź do PZ2, jeśli chcesz pracować nad znakami."
            )
        else:
            title_text = "Wyodrębnij tablice z przejętego runu Z2"
            desc_text = (
                "Poniżej widzisz źródło przejęte z Z2. To ono zostanie użyte w tej operacji."
            )
    else:
        set_extract_start_summary_rows(
            self,
            [
                ("XML", xml_path, "Plik anotacji tablic"),
                ("Obrazy", images_dir, "Katalog zgodny z XML"),
                ("Tablice", plates_text, "Plan wyodrębniania"),
                ("Wynik", output_text, "Docelowy zestaw PZ2"),
            ],
        )
        title_text = "Gotowe do przygotowania zestawu wyodrębnionych tablic"
        desc_text = (
            "Ten krok tworzy preview run: tablice wyodrębnione z obrazów na podstawie potwierdzonego XML. "
            "To wejście dla PZ2."
        )

    for name, text in (
        ("extract_start_ready_badge_lbl", badge_text),
        ("extract_start_action_note_lbl", action_note),
        ("extract_start_action_title_lbl", title_text),
        ("extract_start_action_desc_lbl", desc_text),
    ):
        label = getattr(self, name, None)
        if label is None:
            continue
        try:
            label.configure(text=text)
        except Exception:
            pass

    self._refresh_extract_continue_action_visibility()
    self._refresh_extract_start_card_style(lightweight=lightweight)
    self._refresh_extract_action_state(lightweight=lightweight)


def refresh_extract_workflow_ui(host):
    self = host
    workflow_vm = self._get_step3_extract_workflow_view_model()
    route = workflow_vm.route

    try:
        if hasattr(self, "extract_source_title_lbl"):
            self.extract_source_title_lbl.configure(text=workflow_vm.source_title)
        if hasattr(self, "extract_start_title_lbl"):
            self.extract_start_title_lbl.configure(text=workflow_vm.start_title)
    except Exception:
        pass

    def _set_section_visibility(widget, visible: bool, **pack_kwargs):
        if widget is None:
            return
        try:
            if visible:
                if not str(widget.winfo_manager()):
                    widget.pack(**pack_kwargs)
            elif str(widget.winfo_manager()):
                widget.pack_forget()
        except Exception:
            pass

    _set_section_visibility(
        getattr(self, "extract_entry_section", None),
        workflow_vm.show_entry,
        fill=tk.X,
    )
    _set_section_visibility(
        getattr(self, "extract_source_section", None),
        workflow_vm.show_source,
        fill=tk.X,
        pady=(18, 0),
    )
    _set_section_visibility(
        getattr(self, "extract_start_section", None),
        workflow_vm.show_start,
        fill=tk.X,
        pady=(18, 0),
    )

    if hasattr(self, "extract_continue_source_frame"):
        try:
            if route == "continue" and workflow_vm.show_source:
                if not str(self.extract_continue_source_frame.winfo_manager()):
                    self.extract_continue_source_frame.pack(
                        fill=tk.X,
                        pady=(0, 12),
                        after=getattr(self, "extract_source_intro_lbl", None),
                    )
            elif str(self.extract_continue_source_frame.winfo_manager()):
                self.extract_continue_source_frame.pack_forget()
        except Exception:
            pass
    if hasattr(self, "annotation_run_browse_btn"):
        try:
            self.annotation_run_browse_btn.pack_forget()
        except Exception:
            pass
    if hasattr(self, "extract_use_z2_source_btn"):
        try:
            self.extract_use_z2_source_btn.pack_forget()
        except Exception:
            pass

    if hasattr(self, "extract_source_fields_frame"):
        try:
            fields_visible = bool(workflow_vm.show_source and route == "manual")
            if fields_visible:
                if not str(self.extract_source_fields_frame.winfo_manager()):
                    self.extract_source_fields_frame.pack(
                        fill=tk.X,
                        after=getattr(self, "extract_source_intro_lbl", None),
                    )
            elif str(self.extract_source_fields_frame.winfo_manager()):
                self.extract_source_fields_frame.pack_forget()
        except Exception:
            pass

    if hasattr(self, "extract_source_intro_lbl"):
        self._set_inline_status_label_state(
            self.extract_source_intro_lbl,
            text=workflow_vm.source_intro,
            tone="muted",
            emphasis=False,
        )
    if hasattr(self, "extract_source_fields_hint_lbl"):
        self._set_inline_status_label_state(
            self.extract_source_fields_hint_lbl,
            text=workflow_vm.source_hint,
            tone="muted",
            emphasis=False,
        )
    if hasattr(self, "extract_run_hint_lbl"):
        self._set_inline_status_label_state(
            self.extract_run_hint_lbl,
            text=workflow_vm.run_hint,
            tone=workflow_vm.run_hint_tone,
            emphasis=False,
        )
    self._refresh_continue_source_summary()
    if hasattr(self, "extract_start_hint_lbl"):
        self._set_inline_status_label_state(
            self.extract_start_hint_lbl,
            text=workflow_vm.start_hint,
            tone=workflow_vm.start_hint_tone,
            emphasis=False,
        )
    self._refresh_extract_start_summary(lightweight=True)

    self._refresh_extract_entry_cards()
    self._refresh_extract_step_cards(workflow_vm)
    self._refresh_extract_step_nav_buttons(workflow_vm)


def refresh_extract_start_card_style(host, progress_bar_cls, *, lightweight: bool = False):
    self = host
    card = getattr(self, "extract_start_action_card", None)
    inner = getattr(self, "extract_start_action_inner", None)
    if card is None or inner is None:
        return

    palette = getattr(self.app, "palette", {})
    panel = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    success = palette.get("success", "#2ecc71")
    source_ready = bool(getattr(self, "_extract_last_source_binding_result", {}).get("ok"))
    try:
        preview_ready = bool(
            self._is_extract_preview_ready_fast()
            if lightweight
            else self._is_extract_preview_ready()
        )
    except Exception:
        preview_ready = False
    accent = success if (source_ready or preview_ready) else palette.get("warning", "#f39c12")
    border = blend_hex_colors(accent, palette.get("panel_border", palette.get("border", "#3c3c3c")), 0.34)
    card_bg = blend_hex_colors(palette.get("surface_success", panel_alt), panel, 0.72)
    action_bg = blend_hex_colors(palette.get("surface_info", panel_alt), card_bg, 0.62)
    fg = palette.get("fg", "#f3f3f3")
    muted = self._get_readable_text_color(card_bg, preferred=palette.get("muted", "#9a9a9a"))
    badge_fg = self._get_readable_text_color(accent, preferred="#ffffff")

    for widget, bg in (
        (card, border),
        (inner, card_bg),
        (getattr(self, "extract_start_header_row", None), card_bg),
        (getattr(self, "extract_start_actions_panel", None), action_bg),
        (getattr(self, "extract_start_actions_row", None), action_bg),
    ):
        if widget is None:
            continue
        try:
            widget.configure(bg=bg)
        except Exception:
            pass

    summary_frame = getattr(self, "extract_start_summary_frame", None)
    if summary_frame is not None:
        try:
            summary_frame.configure(
                bg=border,
                highlightbackground=border,
                highlightcolor=border,
            )
        except Exception:
            pass

    for row_index, row in enumerate(getattr(self, "_extract_start_summary_rows", []) or []):
        row_bg = card_bg if row_index % 2 == 0 else blend_hex_colors(panel_alt, card_bg, 0.24)
        for widget in row:
            if widget is None:
                continue
            try:
                widget.configure(bg=row_bg)
            except Exception:
                pass
        try:
            row[0].configure(fg=muted, font=("Segoe UI", 8, "bold"))
            row[1].configure(fg=fg, font=("Segoe UI", 9, "bold"))
            if len(row) > 2 and row[2] is not None:
                row[2].configure(fg=muted, font=("Segoe UI", 8))
        except Exception:
            pass

    for header in list(getattr(self, "_extract_start_summary_header_widgets", []) or []):
        if header is None:
            continue
        try:
            header.configure(
                bg=blend_hex_colors(panel_alt, card_bg, 0.36),
                fg=muted,
                font=("Segoe UI", 8, "bold"),
            )
        except Exception:
            pass

    for label_name, color, font in (
        ("extract_start_ready_badge_lbl", badge_fg, ("Segoe UI", 8, "bold")),
        ("extract_start_action_title_lbl", fg, ("Segoe UI Semibold", 11)),
        ("extract_start_action_desc_lbl", muted, ("Segoe UI", 9)),
        ("extract_start_xml_title_lbl", muted, ("Segoe UI", 8, "bold")),
        ("extract_start_images_title_lbl", muted, ("Segoe UI", 8, "bold")),
        ("extract_start_plates_title_lbl", muted, ("Segoe UI", 8, "bold")),
        ("extract_start_output_title_lbl", muted, ("Segoe UI", 8, "bold")),
        ("extract_start_xml_value_lbl", fg, ("Segoe UI", 9, "bold")),
        ("extract_start_images_value_lbl", fg, ("Segoe UI", 9, "bold")),
        ("extract_start_plates_value_lbl", fg, ("Segoe UI", 9, "bold")),
        ("extract_start_output_value_lbl", fg, ("Segoe UI", 9, "bold")),
        ("extract_start_action_note_lbl", muted, ("Segoe UI", 8)),
    ):
        label = getattr(self, label_name, None)
        if label is None:
            continue
        try:
            label.configure(
                bg=accent if label_name == "extract_start_ready_badge_lbl" else card_bg,
                fg=color,
                font=font,
            )
        except Exception:
            pass

    note = getattr(self, "extract_start_action_note_lbl", None)
    if note is not None:
        try:
            note.configure(bg=action_bg)
        except Exception:
            pass

    status = getattr(self, "ext_status", None)
    if status is not None:
        try:
            status._inline_status_bg = action_bg
            self._set_inline_status_label_state(
                status,
                text=status.cget("text"),
                tone=getattr(status, "_inline_status_tone", "neutral"),
                emphasis=True,
            )
        except Exception:
            pass

    progress = getattr(self, "ext_progress", None)
    if isinstance(progress, progress_bar_cls):
        try:
            progress.configure(
                bg=action_bg,
                trough_color=blend_hex_colors(panel_alt, action_bg, 0.34),
                fill_color=success,
            )
        except Exception:
            pass


def run_extraction(host) -> None:
    self = host
    campaign_step3_active = _campaign_step3_context_active(self)
    if self._is_extract_preview_ready():
        plate_count = self._get_extract_preview_ready_count()
        self._refresh_extract_action_state()
        if campaign_step3_active:
            try:
                self._set_extraction_status(
                    "Tablice są już wyodrębnione. Wracam do pracy bramki T05, gdzie wybierzesz kolejny krok.",
                    "success",
                )
            except Exception:
                pass
            try:
                self._return_to_t05_work_after_step3_pz1()
            except Exception as exc:
                logger.debug(f"Nie udało się wrócić do pracy T05 po gotowym PZ1 kampanii: {exc}")
            return
        if bool(getattr(self, "_campaign_detect_splash_visible", False)):
            try:
                if getattr(self, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name():
                    self.go_to_substep_2(force=True)
                    return
            except Exception:
                pass
            try:
                self._hide_campaign_detect_splash()
            except Exception:
                pass
            return
        try:
            messagebox.showinfo(
                "Tablice są już wyodrębnione",
                (
                    f"Zestaw wyodrębnionych tablic dla PZ2 został już przygotowany poprawnie.\n\n"
                    f"Liczba tablic w aktywnym zestawie: {plate_count}.\n\n"
                    "Przycisk wyodrębniania pozostaje nieaktywny, aby nie uruchamiać tej samej operacji wielokrotnie. "
                    "Przejdź do PZ2 albo zmień źródło w PZ1, jeśli chcesz świadomie przygotować inny zestaw."
                ),
                parent=self.frame.winfo_toplevel(),
            )
        except Exception:
            pass
        return

    preview_dir_raw = str(self.preview_dir_var.get() if hasattr(self, "preview_dir_var") else "").strip()
    if (
        preview_dir_raw
        and self._is_usable_step3_preview_dir(preview_dir_raw, require_plates=True)
        and self._preview_matches_current_extract_source(preview_dir_raw)
        and self._campaign_preview_is_below_min_extracted_plate_count(preview_dir_raw)
    ):
        plate_count = self._get_extract_preview_ready_count(preview_dir_raw)
        self._refresh_extract_action_state()
        try:
            self._set_extraction_status(
                (
                    f"Wyodrębniono {plate_count} tablic. Minimum toru znaków to "
                    f"{self._get_campaign_step3_min_extracted_plate_count()} tablic."
                ),
                "warning",
            )
        except Exception:
            pass
        try:
            self._show_campaign_extract_below_minimum_modal(plate_count, Path(preview_dir_raw))
        except Exception:
            pass
        return

    self._force_save_all()
    validation = self._refresh_source_binding_status(allow_autofind=True)
    if not validation.get("ok"):
        message = str(validation.get("message") or "Źródła wejściowe są niepoprawne.")
        if campaign_step3_active:
            self._show_campaign_extract_failure_modal(
                title="Nie mogę przygotować tablic dla Z3",
                message=(
                    "Źródła do wyodrębniania tablic nie są spójne.\n\n"
                    f"{message}\n\n"
                    "W kampanii nie zatrzymuję Cię w technicznym PZ1. Wróć do Z2 i popraw pulę tablic "
                    "albo wróć do grafu i wybierz inną decyzję bramki."
                ),
                tone="error",
                default_action="repair",
                allow_retry=False,
            )
            return
        if bool(getattr(self, "_campaign_detect_splash_visible", False)):
            self._set_extraction_status(
                str(validation.get("message") or "Źródła wejściowe są niepoprawne."),
                "error",
            )
            return
        return messagebox.showerror("Niezgodne źródła Z3/PZ1", validation.get("message", "Źródła wejściowe są niepoprawne."))

    xml_path = Path(self.xml_path_var.get().strip())
    images_dir = Path(self.images_dir_var.get().strip())
    session_token = self._project_reset_token

    if not xml_path.exists() or not images_dir.exists():
        message = "Brak plików wejściowych. annotations.xml i katalog obrazów muszą pochodzić z tego samego zestawu."
        if campaign_step3_active:
            self._show_campaign_extract_failure_modal(
                title="Brakuje źródeł do wyodrębniania",
                message=(
                    f"{message}\n\n"
                    "Uzupełnij tablice w Z2 albo wróć do grafu. Dopiero poprawne źródło pozwoli przejść do pracy nad znakami."
                ),
                tone="error",
                default_action="repair",
                allow_retry=False,
            )
            return
        if bool(getattr(self, "_campaign_detect_splash_visible", False)):
            self._set_extraction_status(
                "Brak plików wejściowych. annotations.xml i katalog obrazów muszą pochodzić z tego samego zestawu.",
                "error",
            )
            return
        return messagebox.showerror(
            "Błąd",
            "Brak plików wejściowych. annotations.xml i katalog obrazów muszą pochodzić z tego samego zestawu."
        )

    base_out_dir = self._get_step3_chars_root_dir(ensure_exists=True)

    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 1
    while True:
        run_dir = base_out_dir / f"run_{counter:03d}_{timestamp}"
        if not run_dir.exists():
            break
        counter += 1

    run_dir.mkdir(parents=True, exist_ok=True)
    self.preview_dir_var.set(str(run_dir))
    self._reset_preview_cache()
    self._force_save_all()
    self._refresh_extract_workflow_ui()

    try:
        tree = ET.parse(xml_path)
        xml_images = {img.get("name"): img for img in tree.getroot().findall(".//image") if img.get("name")}
    except Exception:
        if campaign_step3_active:
            self._show_campaign_extract_failure_modal(
                title="Nie mogę odczytać anotacji tablic",
                message=(
                    "Plik annotations.xml nie został odczytany poprawnie, więc nie mogę przygotować wyodrębnionych tablic.\n\n"
                    "Wróć do Z2 i zapisz albo popraw anotacje, a potem ponów przejście do pracy nad znakami."
                ),
                tone="error",
                default_action="repair",
                allow_retry=False,
            )
            return
        if bool(getattr(self, "_campaign_detect_splash_visible", False)):
            self._set_extraction_status("Zły plik XML.", "error")
            return
        return messagebox.showerror("Błąd", "Zły plik XML.")

    if hasattr(self.app, "try_begin_exclusive_operation"):
        ok, busy_message = self.app.try_begin_exclusive_operation("pz1.extract.run", "PZ1: wyodrębnianie tablic")
        if not ok:
            if campaign_step3_active:
                self._show_campaign_extract_failure_modal(
                    title="Inny proces jest w toku",
                    message=str(busy_message or "Zaczekaj na zakończenie bieżącej operacji i ponów wyodrębnianie."),
                    tone="warning",
                    default_action="retry",
                    allow_retry=True,
                )
                return
            if bool(getattr(self, "_campaign_detect_splash_visible", False)):
                self._set_extraction_status(str(busy_message or "Inny proces jest w toku."), "error")
                return
            return messagebox.showinfo("Proces w toku", busy_message)
    self.btn_extract.config(state=tk.DISABLED)
    self.btn_ext_stop.config(state=tk.NORMAL)
    self.is_processing = True
    self._set_extraction_status("Start wyodrębniania...", "neutral")

    def worker():
        campaign_success_to_pz2 = False
        campaign_step3_active = _campaign_step3_context_active(self)
        try:
            if session_token != self._project_reset_token:
                return

            self._log(self.ext_log, f"\nROZPOCZETO RUN WYCINANIA PZ1: {run_dir.name}\n", "HEADER")
            generator = PlateGenerator(run_dir)
            total = len(xml_images)
            processed = 0

            for img_name, img_el in xml_images.items():
                if (not self.is_processing) or session_token != self._project_reset_token:
                    break
                img_path = images_dir / img_name
                if not img_path.exists():
                    processed += 1
                    continue

                plates = []
                for poly in img_el.findall(".//polygon[@label='plate']"):
                    pts = [tuple(map(float, p.split(","))) for p in poly.get("points", "").split(";")]
                    if len(pts) >= 4:
                        plates.append(
                            Detection(
                                "plate", 1.0,
                                (min(x for x, y in pts), min(y for x, y in pts), max(x for x, y in pts), max(y for x, y in pts)),
                                polygon=pts
                            )
                        )

                if plates:
                    plates = self._prepare_plate_cut_detections_for_source(img_name, plates)
                    ann = ImageAnnotation(img_name, int(img_el.get("width", 0)), int(img_el.get("height", 0)), plates)
                    generator.generate_from_annotations(
                        img_path, ann,
                        rectify=True,
                        do_deskew=False,
                        enhance_contrast=False,
                        interpolation=self.interpolation_var.get()
                    )

                processed += 1
                self.frame.after(
                    0,
                    lambda p=(processed / max(1, total)) * 100: (
                        self.ext_progress.config(value=p),
                        setattr(self, "_campaign_detect_splash_progress_value", p),
                        self._show_campaign_detect_splash(
                            title=str(getattr(self, "_campaign_detect_splash_title_text", "") or "Wyodrębniam tablice dla Z3"),
                            body=str(getattr(self, "ext_status", None).cget("text") if getattr(self, "ext_status", None) is not None else ""),
                            tone="info",
                            progress=p,
                            show_progress=True,
                            show_return=False,
                        ) if (
                            session_token == self._project_reset_token
                            and (
                                bool(getattr(self, "_campaign_detect_splash_visible", False))
                                or (
                                    bool(getattr(self, "_step3_linear_mode", False))
                                    and bool(CAMPAIGN.get_active_project_name())
                                )
                            )
                        ) else None
                    )
                )
                self.frame.after(
                    0,
                    lambda c=processed, t=total: (
                        self._set_extraction_status(f"{c}/{t} obrazów...", "neutral")
                        if session_token == self._project_reset_token else None
                    )
                )

            generator.save_metadata()
            generated_count = int(getattr(generator, "plate_counter", 0) or 0)
            if generated_count > 0:
                self._write_extract_source_manifest(run_dir, plate_count=generated_count)
            merge_summary = {"matched": 0, "total": 0}
            try:
                in_campaign_reextract = bool(
                    getattr(self, "_step3_linear_mode", False)
                    and CAMPAIGN.get_active_project_name()
                )
                merge_summary = self._merge_reextract_seed_metadata_into_preview(
                    run_dir,
                    allow_historical_fallback=in_campaign_reextract,
                )
            except Exception as merge_exc:
                logger.debug(f"Nie udało się zachować poprzednich boxów przy reextractcie Z3: {merge_exc}")
            finally:
                self._campaign_step3_reextract_seed_metadata = {}
            if self.is_processing and session_token == self._project_reset_token:
                campaign_step3_active = bool(
                    getattr(self, "_step3_linear_mode", False)
                    and CAMPAIGN.get_active_project_name()
                )
                campaign_below_minimum = bool(
                    campaign_step3_active
                    and str(CAMPAIGN.get_iteration_target() or "").strip().lower() == "char"
                    and generated_count < self._get_campaign_step3_min_extracted_plate_count()
                )
                self.frame.after(
                    0,
                    lambda: self._set_extraction_status(
                        (
                            f"Wycinanie zakończone, ale powstało za mało tablic ({generated_count})."
                            if campaign_below_minimum
                            else (
                            f"Wycinanie zakończone. Zachowano poprzednie boxy dla {int(merge_summary.get('matched', 0) or 0)} tablic."
                            if int(merge_summary.get("matched", 0) or 0) > 0
                            else "Wycinanie zakończone. Możesz przejść do PZ2."
                            )
                        ),
                        "warning" if campaign_below_minimum else "success",
                    ),
                )
                if not (campaign_step3_active and not campaign_below_minimum):
                    self.frame.after(0, self._schedule_detection_preview_autoload)
                if campaign_below_minimum:
                    self.frame.after(
                        0,
                        lambda count=generated_count, path=run_dir: self._show_campaign_extract_below_minimum_modal(count, path),
                    )
                elif campaign_step3_active:
                    try:
                        self._campaign_step3_hold_pz2_after_reextract = False
                    except Exception:
                        pass
                    def _commit_preview_and_return_t05(path=run_dir):
                        preview_dir_raw = str(path or "").strip()
                        if preview_dir_raw:
                            try:
                                if str(self.preview_dir_var.get() or "").strip() != preview_dir_raw:
                                    self.preview_dir_var.set(preview_dir_raw)
                            except Exception:
                                pass
                            try:
                                CAMPAIGN.set_step3_preview_dir(preview_dir_raw)
                            except Exception:
                                pass
                            try:
                                sync_registry = getattr(self, "_sync_campaign_step3_preview_artifact_registry", None)
                                if callable(sync_registry):
                                    sync_registry(preview_dir_raw)
                            except Exception:
                                pass
                            try:
                                self._force_save_all()
                            except Exception:
                                pass
                        self._return_to_t05_work_after_step3_pz1()

                    self.frame.after(0, _commit_preview_and_return_t05)
                else:
                    self.frame.after(0, self.unlock_detection_subtab)
                if generated_count > 0 and not campaign_below_minimum and not campaign_step3_active:
                    self.frame.after(
                        0,
                        lambda count=generated_count, path=run_dir: self._show_extract_completed_modal(count, path),
                    )
                if campaign_step3_active and not campaign_below_minimum:
                    self.frame.after(
                        0,
                        lambda: self.app.update_status(
                            "Wyodrębnianie tablic zakończone. W pracy bramki T05 wybierz PZ2 jako następny krok.",
                            "success",
                        ) if hasattr(self.app, "update_status") else None,
                    )
        except Exception as e:
            if session_token == self._project_reset_token:
                self.frame.after(0, lambda: self._set_extraction_status("Błąd wyodrębniania", "error"))
                if campaign_step3_active:
                    self.frame.after(
                        0,
                        lambda err=str(e): self._show_campaign_extract_failure_modal(
                            title="Wyodrębnianie tablic nie powiodło się",
                            message=(
                                "Nie udało się przygotować wyodrębnionych tablic dla pracy nad znakami.\n\n"
                                f"Szczegóły błędu: {err}\n\n"
                                "Możesz ponowić wyodrębnianie, wrócić do Z2 i poprawić tablice albo wrócić do grafu."
                            ),
                            tone="error",
                            default_action="retry",
                            allow_retry=True,
                        ),
                    )
                self._log(self.ext_log, f"\n❌ BŁĄD: {e}\n", "ERROR")
        finally:
            if session_token == self._project_reset_token:
                self.is_processing = False
                self._campaign_step3_reextract_seed_metadata = {}
                if hasattr(self.app, "end_exclusive_operation"):
                    self.app.end_exclusive_operation("pz1.extract.run")
                if not campaign_success_to_pz2:
                    try:
                        self._campaign_pz2_sync_loading = False
                    except Exception:
                        pass
                    self.frame.after(0, self._refresh_extract_action_state)
                self.frame.after(0, lambda: self.btn_ext_stop.config(state=tk.DISABLED))

    threading.Thread(target=worker, daemon=True).start()


def build_extraction_tab(host, parent, SlimProgressBar, nav_button_width):
    self = host
    NAV_BUTTON_WIDTH = nav_button_width
    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")

    def ensure_wrap(widget, container, *, padding=28, min_wrap=220):
        try:
            self.app.ensure_adaptive_wrap(widget, container=container, padding=padding, min_wrap=min_wrap)
        except Exception:
            pass

    parent.grid_rowconfigure(0, weight=1)
    parent.grid_rowconfigure(1, weight=0)
    parent.grid_columnconfigure(0, weight=1)

    self._extract_content_inset = 14
    self._extract_content_max_width = 760
    self._extract_left_panel_width = self._extract_content_max_width + (2 * self._extract_content_inset) + 24

    left_frame = ttk.Frame(parent, style="Panel.TFrame", width=self._extract_left_panel_width)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
    self.extract_main_left_frame = left_frame
    self.extract_main_pane = None
    self.extract_main_right_frame = None

    left_frame.grid_rowconfigure(0, weight=1)
    left_frame.grid_rowconfigure(1, weight=0)
    left_frame.grid_columnconfigure(0, weight=1)

    self.extract_left_scroll_host = ttk.Frame(left_frame, style="Panel.TFrame")
    self.extract_left_scroll_host.grid(row=0, column=0, sticky="nsew")
    self.extract_left_scroll_host.grid_rowconfigure(0, weight=1)
    self.extract_left_scroll_host.grid_columnconfigure(0, weight=1)

    self.extract_left_canvas = tk.Canvas(
        self.extract_left_scroll_host,
        bg=panel_bg,
        bd=0,
        highlightthickness=0,
        width=self._extract_content_max_width + (2 * self._extract_content_inset),
    )
    self.extract_left_canvas.grid(row=0, column=0, sticky="nsew")

    self.extract_left_scrollbar = WebSlimScrollbar(
        self.extract_left_scroll_host,
        command=self.extract_left_canvas.yview,
        width=10,
    )
    self.extract_left_scrollbar.grid(row=0, column=1, sticky="ns", padx=(6, 0))
    self.extract_left_canvas.configure(yscrollcommand=self.extract_left_scrollbar.set)

    self.extract_left_content = ttk.Frame(self.extract_left_canvas, style="Panel.TFrame")
    self.extract_left_content.grid_columnconfigure(0, weight=1)
    self.extract_left_content_window = self.extract_left_canvas.create_window(
        (self._extract_content_inset, 0),
        window=self.extract_left_content,
        anchor="nw",
        width=self._extract_content_max_width,
    )
    self.extract_left_content.bind("<Configure>", self._sync_extract_left_scrollregion, add="+")
    self.extract_left_canvas.bind("<Configure>", self._sync_extract_left_canvas_width, add="+")
    self.extract_left_canvas.bind(
        "<MouseWheel>",
        lambda e: self._redirect_child_mousewheel_to_canvas(e, self.extract_left_canvas, self._extract_left_canvas_overflows),
        add="+",
    )
    self.extract_left_canvas.bind(
        "<Button-4>",
        lambda e: self._redirect_child_mousewheel_to_canvas(e, self.extract_left_canvas, self._extract_left_canvas_overflows),
        add="+",
    )
    self.extract_left_canvas.bind(
        "<Button-5>",
        lambda e: self._redirect_child_mousewheel_to_canvas(e, self.extract_left_canvas, self._extract_left_canvas_overflows),
        add="+",
    )

    settings_col = ttk.Frame(self.extract_left_content, style="Panel.TFrame")
    settings_col.pack(fill=tk.X, expand=True, padx=12, pady=(14, 20))

    extract_workflow_shell_border = blend_hex_colors(
        palette.get("accent", "#4f8de3"),
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        0.62,
    )
    extract_workflow_shell_fill = blend_hex_colors(
        palette.get("surface_info", panel_bg),
        panel_bg,
        0.80,
    )
    self.extract_workflow_shell = tk.Frame(
        settings_col,
        bg=extract_workflow_shell_border,
        bd=0,
        highlightthickness=0,
        padx=1,
        pady=1,
    )
    # Pokazujemy shell dopiero po pierwszym przeliczeniu VM/geometrii, żeby
    # użytkownik nie widział półzłożonych kafli podczas leniwego ładowania Z3.
    self._extract_workflow_shell_pack_parent = settings_col

    self.extract_workflow_shell_inner = tk.Frame(
        self.extract_workflow_shell,
        bg=extract_workflow_shell_fill,
        bd=0,
        highlightthickness=0,
        padx=10,
        pady=12,
    )
    self.extract_workflow_shell_inner.pack(fill=tk.X, expand=True)

    self.extract_entry_section = tk.Frame(
        self.extract_workflow_shell_inner,
        bg=extract_workflow_shell_fill,
        bd=0,
        highlightthickness=0,
    )
    self.extract_entry_section.pack(fill=tk.X)

    self.extract_entry_title_lbl = SectionHeaderLabel(
        self.extract_entry_section,
        self.app,
        text="Jak chcesz wejść do PZ1?",
    )
    self.extract_entry_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 10))

    self.extract_entry_cards_frame = tk.Frame(
        self.extract_entry_section,
        bg=extract_workflow_shell_fill,
        bd=0,
        highlightthickness=0,
    )
    self.extract_entry_cards_frame.pack(fill=tk.X)

    card_bg = palette.get("panel_alt", palette.get("panel", "#252526"))
    card_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    continue_card = tk.Frame(
        self.extract_entry_cards_frame,
        bd=0,
        highlightthickness=1,
        highlightbackground=card_border,
        highlightcolor=card_border,
        bg=card_bg,
        padx=14,
        pady=12,
        cursor="hand2",
    )
    continue_card.pack(fill=tk.X, pady=(0, 8))
    continue_badge = tk.Label(
        continue_card,
        text="WEJŚCIE Z Z2",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 9, "bold"),
        bd=0,
        highlightthickness=0,
        bg=card_bg,
    )
    continue_badge.pack(anchor=tk.W, fill=tk.X)
    continue_title = tk.Label(
        continue_card,
        text="Kontynuuj na runie anotacji",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI Semibold", 11),
        bd=0,
        highlightthickness=0,
        bg=card_bg,
    )
    continue_title.pack(anchor=tk.W, fill=tk.X)
    continue_desc = tk.Label(
        continue_card,
        text=(
            "Przejmij aktywny albo ostatni run anotacji Z2. "
            "W tym torze nie wskazujesz ręcznie folderu runu, XML ani obrazów."
        ),
        anchor="w",
        justify=tk.LEFT,
        wraplength=336,
        bd=0,
        highlightthickness=0,
        bg=card_bg,
    )
    continue_desc.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))

    manual_card = tk.Frame(
        self.extract_entry_cards_frame,
        bd=0,
        highlightthickness=1,
        highlightbackground=card_border,
        highlightcolor=card_border,
        bg=card_bg,
        padx=14,
        pady=12,
        cursor="hand2",
    )
    manual_card.pack(fill=tk.X)
    manual_badge = tk.Label(
        manual_card,
        text="XML + OBRAZY",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 9, "bold"),
        bd=0,
        highlightthickness=0,
        bg=card_bg,
    )
    manual_badge.pack(anchor=tk.W, fill=tk.X)
    manual_title = tk.Label(
        manual_card,
        text="Wskaż anotacje do wyodrębnienia",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI Semibold", 11),
        bd=0,
        highlightthickness=0,
        bg=card_bg,
    )
    manual_title.pack(anchor=tk.W, fill=tk.X)
    manual_desc = tk.Label(
        manual_card,
        text=(
            "Podaj annotations.xml oraz folder oryginalnych obrazów, z których mam "
            "wyciąć tablice. To dobry tor dla importu z zewnątrz."
        ),
        anchor="w",
        justify=tk.LEFT,
        wraplength=336,
        bd=0,
        highlightthickness=0,
        bg=card_bg,
    )
    manual_desc.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))

    self._extract_entry_cards = {
        "continue": {
            "frame": continue_card,
            "badge": continue_badge,
            "title": continue_title,
            "desc": continue_desc,
        },
        "manual": {
            "frame": manual_card,
            "badge": manual_badge,
            "title": manual_title,
            "desc": manual_desc,
        },
    }
    self._bind_extract_entry_card(continue_card, "continue")
    self._bind_extract_entry_card(manual_card, "manual")
    self._paint_extract_entry_cards_first()

    self.extract_source_section = ttk.Frame(self.extract_workflow_shell_inner, style="Panel.TFrame")
    self.extract_source_section.pack(fill=tk.X, pady=(18, 0))
    self.extract_source_section.pack_forget()
    self.extract_source_title_lbl = SectionHeaderLabel(
        self.extract_source_section,
        self.app,
        text="Źródła wejścia",
    )
    self.extract_source_title_lbl.pack(anchor=tk.W, fill=tk.X)

    self.extract_source_intro_lbl = tk.Label(
        self.extract_source_section,
        text="Tutaj dopinasz konkretne źródła wejścia do PZ1.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=760,
        bd=0,
        highlightthickness=0,
    )
    self.extract_source_intro_lbl.pack(anchor=tk.W, fill=tk.X, pady=(6, 12))
    self._set_inline_status_label_state(self.extract_source_intro_lbl, text=self.extract_source_intro_lbl.cget("text"), tone="muted", emphasis=False)
    ensure_wrap(self.extract_source_intro_lbl, self.extract_source_section, padding=28, min_wrap=240)

    self.extract_continue_source_frame = ttk.LabelFrame(
        self.extract_source_section,
        text=" Źródło przejęte z Z2 ",
        padding=12,
    )
    self.extract_continue_source_frame.pack(fill=tk.X, pady=(0, 12))
    self.extract_continue_source_frame.pack_forget()

    self.annotation_run_dir_entry = ttk.Entry(
        self.extract_continue_source_frame,
        textvariable=self.annotation_run_dir_var,
        state="readonly",
    )
    self.annotation_run_dir_entry.pack(fill=tk.X)

    self.extract_continue_buttons_row = ttk.Frame(self.extract_continue_source_frame)
    self.extract_continue_buttons_row.pack(fill=tk.X, pady=(8, 0))

    self.extract_use_z2_source_btn = ttk.Button(
        self.extract_continue_buttons_row,
        text="Użyj źródła z Z2",
        command=self._use_z2_source_on_demand,
    )
    self.extract_use_z2_source_btn.pack_forget()

    self.annotation_run_browse_btn = ttk.Button(
        self.extract_continue_buttons_row,
        text="Wskaż folder runu anotacji",
        command=self._pick_annotation_run_dir,
    )
    # Tryb kontynuacji ma korzystać z aktywnego źródła Z2. Ręczne wskazanie
    # XML + obrazów jest obsługiwane przez drugi kafel, żeby nie mieszać flow.
    self.annotation_run_browse_btn.pack_forget()

    self.extract_run_hint_lbl = tk.Label(
        self.extract_continue_source_frame,
        text="Kliknij przycisk, aby przejąć aktywne lub ostatnie źródło z Z2.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
        bd=0,
        highlightthickness=0,
    )
    self.extract_run_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))
    self._set_inline_status_label_state(self.extract_run_hint_lbl, text=self.extract_run_hint_lbl.cget("text"), tone="muted", emphasis=False)
    ensure_wrap(self.extract_run_hint_lbl, self.extract_continue_source_frame, padding=28, min_wrap=240)

    self.extract_continue_summary_frame = ttk.LabelFrame(
        self.extract_continue_source_frame,
        text=" Źródła wyprowadzone z runu ",
        padding=10,
    )
    self.extract_continue_summary_frame.pack(fill=tk.X, pady=(10, 0))
    self.extract_continue_summary_frame.grid_columnconfigure(1, weight=1)

    ttk.Label(
        self.extract_continue_summary_frame,
        text="XML:",
        font=("Segoe UI", 8, "bold"),
    ).grid(row=0, column=0, sticky="nw", padx=(0, 10), pady=(0, 4))
    self.extract_continue_xml_value_lbl = tk.Label(
        self.extract_continue_summary_frame,
        text="Po użyciu runu Z2 pojawi się ścieżka annotations.xml.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=620,
        bd=0,
        highlightthickness=0,
    )
    self.extract_continue_xml_value_lbl.grid(row=0, column=1, sticky="ew", pady=(0, 4))
    ensure_wrap(self.extract_continue_xml_value_lbl, self.extract_continue_summary_frame, padding=120, min_wrap=220)

    ttk.Label(
        self.extract_continue_summary_frame,
        text="Obrazy:",
        font=("Segoe UI", 8, "bold"),
    ).grid(row=1, column=0, sticky="nw", padx=(0, 10))
    self.extract_continue_images_value_lbl = tk.Label(
        self.extract_continue_summary_frame,
        text="Po użyciu runu Z2 pojawi się katalog obrazów źródłowych.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=620,
        bd=0,
        highlightthickness=0,
    )
    self.extract_continue_images_value_lbl.grid(row=1, column=1, sticky="ew")
    ensure_wrap(self.extract_continue_images_value_lbl, self.extract_continue_summary_frame, padding=120, min_wrap=220)
    self._refresh_continue_source_summary()

    self.extract_source_fields_frame = ttk.LabelFrame(
        self.extract_source_section,
        text=" annotations.xml i katalog obrazów ",
        padding=12,
    )
    self.extract_source_fields_frame.pack(fill=tk.X)

    ttk.Label(
        self.extract_source_fields_frame,
        text="Plik annotations.xml:",
        font=("Segoe UI", 9, "bold"),
    ).pack(anchor=tk.W, pady=(0, 2))
    row_xml = ttk.Frame(self.extract_source_fields_frame)
    row_xml.pack(fill=tk.X, pady=(0, 10))
    self.xml_path_entry = ttk.Entry(row_xml, textvariable=self.xml_path_var)
    self.xml_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.xml_path_browse_btn = ttk.Button(
        row_xml,
        text="Wybierz XML",
        command=self._pick_xml_file,
    )
    self.xml_path_browse_btn.pack(side=tk.RIGHT, padx=(5, 0))

    self.extract_xml_hint_lbl = tk.Label(
        self.extract_source_fields_frame,
        text="To plik z polygonami tablic wygenerowany w Z2 albo poprawiony dalej w CVAT.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
        bd=0,
        highlightthickness=0,
    )
    self.extract_xml_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
    self._set_inline_status_label_state(self.extract_xml_hint_lbl, text=self.extract_xml_hint_lbl.cget("text"), tone="muted", emphasis=False)
    ensure_wrap(self.extract_xml_hint_lbl, self.extract_source_fields_frame, padding=28, min_wrap=240)

    ttk.Label(
        self.extract_source_fields_frame,
        text="Folder oryginalnych obrazów:",
        font=("Segoe UI", 9, "bold"),
    ).pack(anchor=tk.W, pady=(0, 2))
    row_img = ttk.Frame(self.extract_source_fields_frame)
    row_img.pack(fill=tk.X, pady=(0, 10))
    self.images_dir_entry = ttk.Entry(row_img, textvariable=self.images_dir_var)
    self.images_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    self.images_dir_browse_btn = ttk.Button(
        row_img,
        text="Wybierz folder",
        command=self._pick_images_dir,
    )
    self.images_dir_browse_btn.pack(side=tk.RIGHT, padx=(5, 0))

    self.extract_images_hint_lbl = tk.Label(
        self.extract_source_fields_frame,
        text="To musi być ten sam katalog zdjęć, na których wykonano anotacje zapisane w XML.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
        bd=0,
        highlightthickness=0,
    )
    self.extract_images_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
    self._set_inline_status_label_state(self.extract_images_hint_lbl, text=self.extract_images_hint_lbl.cget("text"), tone="muted", emphasis=False)
    ensure_wrap(self.extract_images_hint_lbl, self.extract_source_fields_frame, padding=28, min_wrap=240)

    self.extract_source_fields_hint_lbl = tk.Label(
        self.extract_source_fields_frame,
        text="Po wskazaniu XML mogę dodatkowo spróbować dopasować właściwy katalog obrazów.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
        bd=0,
        highlightthickness=0,
    )
    self.extract_source_fields_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
    self._set_inline_status_label_state(self.extract_source_fields_hint_lbl, text=self.extract_source_fields_hint_lbl.cget("text"), tone="muted", emphasis=False)
    ensure_wrap(self.extract_source_fields_hint_lbl, self.extract_source_fields_frame, padding=28, min_wrap=240)

    self.source_binding_status_frame = tk.Frame(
        self.extract_source_section,
        bg="#153424",
        bd=0,
        highlightthickness=1,
        highlightbackground="#2ecc71",
        highlightcolor="#2ecc71",
    )
    self.source_binding_status_title_lbl = tk.Label(
        self.source_binding_status_frame,
        text="Status źródeł i plan wyodrębniania",
        justify=tk.LEFT,
        anchor="w",
        font=("Segoe UI", 9, "bold"),
        bg="#153424",
        fg="#86efac",
        bd=0,
        highlightthickness=0,
    )
    self.source_binding_status_title_lbl.pack(anchor=tk.W, fill=tk.X, padx=10, pady=(8, 2))
    self.source_binding_status_lbl = tk.Label(
        self.source_binding_status_frame,
        text="",
        justify=tk.LEFT,
        anchor="w",
        wraplength=720,
        bg="#153424",
        fg="#dcfce7",
        bd=0,
        highlightthickness=0,
        padx=10,
        pady=0,
    )
    self.source_binding_status_lbl.pack(fill=tk.X, padx=0, pady=(0, 8))
    ensure_wrap(self.source_binding_status_lbl, self.source_binding_status_frame, padding=28, min_wrap=240)

    extract_start_border = extract_workflow_shell_fill
    extract_start_fill = extract_workflow_shell_fill
    self.extract_start_section = tk.Frame(
        self.extract_workflow_shell_inner,
        bg=extract_start_border,
        bd=0,
        highlightthickness=0,
        padx=1,
        pady=1,
    )
    self.extract_start_section.pack(fill=tk.X, pady=(18, 0))
    self.extract_start_section.pack_forget()
    self.extract_start_section_inner = tk.Frame(
        self.extract_start_section,
        bg=extract_start_fill,
        bd=0,
        highlightthickness=0,
        padx=9,
        pady=11,
    )
    self.extract_start_section_inner.pack(fill=tk.X, expand=True)
    self.extract_start_title_lbl = SectionHeaderLabel(
        self.extract_start_section_inner,
        self.app,
        text="Uruchom wyodrębnianie",
    )
    self.extract_start_title_lbl.pack(anchor=tk.W, fill=tk.X)

    self.extract_start_hint_lbl = tk.Label(
        self.extract_start_section_inner,
        text="Najpierw potwierdź zgodność źródeł, a potem uruchom wyodrębnianie.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=760,
        bd=0,
        highlightthickness=0,
    )
    self.extract_start_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(6, 12))
    self._set_inline_status_label_state(self.extract_start_hint_lbl, text=self.extract_start_hint_lbl.cget("text"), tone="muted", emphasis=False)
    ensure_wrap(self.extract_start_hint_lbl, self.extract_start_section_inner, padding=28, min_wrap=240)

    start_card_border = palette.get("success", "#2ecc71")
    start_card_fill = blend_hex_colors(palette.get("surface_success", panel_bg), panel_bg, 0.72)
    self.extract_start_action_card = tk.Frame(
        self.extract_start_section_inner,
        bg=start_card_border,
        bd=0,
        highlightthickness=0,
        padx=1,
        pady=1,
    )
    self.extract_start_action_card.pack(fill=tk.X)
    self.extract_start_action_inner = tk.Frame(
        self.extract_start_action_card,
        bg=start_card_fill,
        bd=0,
        highlightthickness=0,
        padx=10,
        pady=12,
    )
    self.extract_start_action_inner.pack(fill=tk.X)

    self.extract_start_header_row = tk.Frame(self.extract_start_action_inner, bg=start_card_fill, bd=0, highlightthickness=0)
    self.extract_start_header_row.pack(fill=tk.X)
    self.extract_start_ready_badge_lbl = tk.Label(
        self.extract_start_header_row,
        text="SPRAWDZAM",
        anchor="center",
        justify=tk.CENTER,
        padx=8,
        pady=3,
        bd=0,
        highlightthickness=0,
    )
    self.extract_start_action_title_lbl = tk.Label(
        self.extract_start_header_row,
        text="Przygotowuję PZ1",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.extract_start_action_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self.extract_start_ready_badge_lbl.pack(anchor=tk.W, pady=(6, 0))
    self.extract_start_action_desc_lbl = tk.Label(
        self.extract_start_action_inner,
        text="Sprawdzam źródła i stan wyodrębnionych tablic. Za chwilę pokażę pełne podsumowanie.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
        bd=0,
        highlightthickness=0,
    )
    self.extract_start_action_desc_lbl.pack(anchor=tk.W, fill=tk.X, pady=(6, 10))
    ensure_wrap(self.extract_start_action_desc_lbl, self.extract_start_action_inner, padding=28, min_wrap=240)

    self.extract_start_summary_frame = tk.Frame(
        self.extract_start_action_inner,
        bg=start_card_border,
        bd=0,
        highlightthickness=1,
        highlightbackground=start_card_border,
        highlightcolor=start_card_border,
    )
    self.extract_start_summary_frame.pack(fill=tk.X, pady=(0, 12))
    self.extract_start_summary_frame.pack_forget()
    self.extract_start_summary_frame.grid_columnconfigure(0, weight=0, minsize=116)
    self.extract_start_summary_frame.grid_columnconfigure(1, weight=5, minsize=390)
    self.extract_start_summary_frame.grid_columnconfigure(2, weight=1, minsize=138)
    self._extract_start_summary_header_widgets = []
    self._extract_start_summary_rows = []

    for column, header_text in enumerate(("Pozycja", "Wartość", "Znaczenie")):
        header_lbl = tk.Label(
            self.extract_start_summary_frame,
            text=header_text,
            anchor="w",
            justify=tk.LEFT,
            bg=start_card_fill,
            bd=0,
            highlightthickness=0,
            padx=6,
            pady=4,
        )
        header_lbl.grid(
            row=0,
            column=column,
            sticky="ew",
            padx=(1 if column == 0 else 0, 1),
            pady=(1, 1),
        )
        self._extract_start_summary_header_widgets.append(header_lbl)

    def _add_start_summary_row(row_index: int, title: str, title_attr: str, value_attr: str):
        row_bg = start_card_fill
        title_lbl = tk.Label(
            self.extract_start_summary_frame,
            text=title,
            anchor="w",
            justify=tk.LEFT,
            bg=row_bg,
            bd=0,
            highlightthickness=0,
            padx=6,
            pady=3,
        )
        title_lbl.grid(row=row_index + 1, column=0, sticky="nsew", padx=(1, 1), pady=(0, 1))
        value_lbl = tk.Label(
            self.extract_start_summary_frame,
            text="Przygotowuję...",
            anchor="w",
            justify=tk.LEFT,
            bg=row_bg,
            bd=0,
            highlightthickness=0,
            padx=6,
            pady=3,
            wraplength=560,
        )
        value_lbl.grid(row=row_index + 1, column=1, sticky="nsew", padx=(0, 1), pady=(0, 1))
        note_lbl = tk.Label(
            self.extract_start_summary_frame,
            text="Czekam na dane",
            anchor="w",
            justify=tk.LEFT,
            bg=row_bg,
            bd=0,
            highlightthickness=0,
            padx=6,
            pady=3,
            wraplength=145,
        )
        note_lbl.grid(row=row_index + 1, column=2, sticky="nsew", padx=(0, 1), pady=(0, 1))
        setattr(self, title_attr, title_lbl)
        setattr(self, value_attr, value_lbl)
        self._extract_start_summary_rows.append((title_lbl, value_lbl, note_lbl))
        ensure_wrap(value_lbl, self.extract_start_summary_frame, padding=230, min_wrap=360)
        ensure_wrap(note_lbl, self.extract_start_summary_frame, padding=610, min_wrap=120)

    _add_start_summary_row(0, "XML", "extract_start_xml_title_lbl", "extract_start_xml_value_lbl")
    _add_start_summary_row(1, "Obrazy", "extract_start_images_title_lbl", "extract_start_images_value_lbl")
    _add_start_summary_row(2, "Tablice", "extract_start_plates_title_lbl", "extract_start_plates_value_lbl")
    _add_start_summary_row(3, "Wynik", "extract_start_output_title_lbl", "extract_start_output_value_lbl")
    _add_start_summary_row(4, "", "extract_start_extra_a_title_lbl", "extract_start_extra_a_value_lbl")
    _add_start_summary_row(5, "", "extract_start_extra_b_title_lbl", "extract_start_extra_b_value_lbl")

    action_panel_fill = blend_hex_colors(palette.get("surface_info", panel_bg), start_card_fill, 0.62)
    self.extract_start_actions_panel = tk.Frame(
        self.extract_start_action_inner,
        bg=action_panel_fill,
        bd=0,
        highlightthickness=0,
        padx=12,
        pady=10,
    )
    self.extract_start_actions_panel.pack(fill=tk.X)

    self.extract_start_actions_row = tk.Frame(self.extract_start_actions_panel, bg=action_panel_fill, bd=0, highlightthickness=0)
    self.extract_start_actions_row.pack(anchor=tk.W, fill=tk.X)

    self.btn_extract = ttk.Button(
        self.extract_start_actions_row,
        text="Wyodrębnij tablice do PZ2",
        command=self._run_extraction,
        style="WorkflowCardPrimary.TButton",
    )
    self.btn_extract.pack(side=tk.LEFT)

    self.btn_cancel_continue_extract = ttk.Button(
        self.extract_start_actions_row,
        text="Zrezygnuj i wróć",
        command=self._cancel_continue_extract_flow,
        style="WorkflowCard.TButton",
    )
    self.btn_cancel_continue_extract.pack(side=tk.LEFT, padx=(8, 0))
    self.btn_cancel_continue_extract.pack_forget()

    self.btn_ext_stop = ttk.Button(
        self.extract_start_actions_row,
        text="Zatrzymaj",
        command=lambda: setattr(self, 'is_processing', False),
        state=tk.DISABLED,
        style="WorkflowCard.TButton",
    )
    self.btn_ext_stop.pack(side=tk.LEFT, padx=(8, 0))

    self.ext_progress = SlimProgressBar(
        self.extract_start_actions_panel,
        maximum=100,
        value=0,
        thickness=2,
        trough_color=palette.get("panel_alt", palette.get("panel", "#252526")),
        fill_color=palette.get("success", "#2ecc71"),
        bg=palette.get("panel", "#252526"),
        height=6,
    )
    self.ext_progress.pack(fill=tk.X, pady=(12, 4))
    self.ext_status = tk.Label(
        self.extract_start_actions_panel,
        text="Przygotowuję PZ1...",
        anchor="w",
        font=("Segoe UI", 9),
        bd=0,
        highlightthickness=0
    )
    self.ext_status.pack(anchor=tk.W)
    self._set_inline_status_label_state(self.ext_status, text="Przygotowuję PZ1...", tone="neutral", emphasis=True)

    self.extract_start_action_note_lbl = tk.Label(
        self.extract_start_actions_panel,
        text="Sprawdzam źródła. Akcje pojawią się po pierwszym odświeżeniu stanu PZ1.",
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
        bd=0,
        highlightthickness=0,
    )
    self.extract_start_action_note_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 0))
    ensure_wrap(self.extract_start_action_note_lbl, self.extract_start_actions_panel, padding=28, min_wrap=240)

    def _bind_extract_help_later():
        for widget, key in (
            (self.extract_entry_section, "t2_sources"),
            (self.extract_source_section, "t2_sources"),
            (self.source_binding_status_frame, "t2_sources"),
            (self.source_binding_status_title_lbl, "t2_sources"),
            (self.source_binding_status_lbl, "t2_sources"),
            (row_xml, "t2_xml"),
            (row_img, "t2_img"),
            (self.extract_use_z2_source_btn, "t2_sources"),
            (self.annotation_run_browse_btn, "t2_sources"),
            (self.xml_path_browse_btn, "t2_xml"),
            (self.images_dir_browse_btn, "t2_img"),
            (self.extract_continue_summary_frame, "t2_sources"),
            (self.extract_continue_xml_value_lbl, "t2_sources"),
            (self.extract_continue_images_value_lbl, "t2_sources"),
            (self.extract_start_action_card, "t2_cut_start"),
            (self.extract_start_action_inner, "t2_cut_start"),
            (self.btn_extract, "t2_cut_start"),
            (self.btn_ext_stop, "t2_cut_stop"),
        ):
            try:
                HELP.bind_help(widget, key)
            except Exception:
                pass

    def _bind_extract_scroll_later():
        try:
            self._bind_scroll_canvas_children(
                self.extract_left_content,
                self.extract_left_canvas,
                self._extract_left_canvas_overflows,
            )
        except Exception:
            pass

    def _finish_extract_entry_cards_first_paint():
        try:
            self._sync_extract_left_canvas_width()
            self._sync_extract_left_scrollregion()
        except Exception:
            pass
        for widget in (
            getattr(self, "extract_entry_cards_frame", None),
            getattr(self, "extract_workflow_shell_inner", None),
            getattr(self, "extract_left_canvas", None),
        ):
            if widget is None:
                continue
            try:
                widget.update_idletasks()
            except Exception:
                pass

    self.frame.after(900, _bind_extract_help_later)
    self.frame.after(1100, _bind_extract_scroll_later)
    self.frame.after_idle(self._sync_extract_left_scrollregion)
    self.frame.after_idle(self._sync_extract_left_canvas_width)

    self.extract_right_panel = None

    self.extract_log_section = ttk.Frame(left_frame, style="Panel.TFrame")
    self.ext_log_host = ttk.Frame(self.extract_log_section, style="Panel.TFrame")
    self.ext_log_host.pack(fill=tk.BOTH, expand=True)

    self.ext_log = tk.Text(
        self.ext_log_host,
        wrap=tk.WORD,
        font=("Consolas", 10),
        bg="#161616",
        fg="#f3f3f3",
        insertbackground="#f3f3f3",
        bd=0,
        relief=tk.FLAT,
        highlightthickness=0,
        height=1,
    )
    self.ext_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    self.ext_log_scrollbar = WebSlimScrollbar(
        self.ext_log_host,
        orient=tk.VERTICAL,
        command=self.ext_log.yview,
        auto_hide=False,
    )
    self.ext_log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self.ext_log.configure(yscrollcommand=self.ext_log_scrollbar.set)
    self.ext_log.web_vbar = self.ext_log_scrollbar

    nav_panel = ttk.Frame(left_frame, style="Panel.TFrame")
    nav_panel.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

    self.extract_step_nav_row = ttk.Frame(nav_panel, style="Panel.TFrame")
    self.extract_step_nav_row.pack(fill=tk.X)
    self.extract_step_nav_row.pack_forget()

    self.extract_step_nav_frame = ttk.Frame(self.extract_step_nav_row, style="Panel.TFrame")
    self.extract_step_nav_frame.pack(side=tk.LEFT)

    self.extract_step_back_btn = ttk.Button(
        self.extract_step_nav_frame,
        text="Wstecz",
        command=self._go_to_previous_extract_step,
        state=tk.DISABLED,
        style="WorkflowCard.TButton",
    )
    self.extract_step_back_btn.pack(side=tk.LEFT)
    self.extract_step_back_btn.config(text="Wstecz", padding=(8, 2), width=NAV_BUTTON_WIDTH)

    self.extract_step_next_btn = ttk.Button(
        self.extract_step_nav_frame,
        text="Dalej",
        command=self._go_to_next_extract_step,
        state=tk.DISABLED,
        style="Accent.TButton",
    )
    self.extract_step_next_btn.pack(side=tk.LEFT, padx=(8, 0))
    self.extract_step_next_btn.config(text="Dalej", padding=(8, 2), width=NAV_BUTTON_WIDTH)

    self.extract_main_nav_panel = ttk.Frame(parent, style="Panel.TFrame")
    self.extract_main_nav_panel.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
    self.extract_main_nav_panel.grid_remove()

    self.extract_nav_divider = tk.Frame(
        self.extract_main_nav_panel,
        height=1,
        bg=palette.get("panel_border", palette.get("border", "#3c3c3c")),
        bd=0,
        highlightthickness=0,
    )
    self.extract_nav_divider.pack(fill=tk.X, pady=(0, 8))

    self.extract_tab_nav_row = ttk.Frame(self.extract_main_nav_panel, style="Panel.TFrame")
    self.extract_tab_nav_row.pack(fill=tk.X)
    self.extract_tab_nav_row.grid_columnconfigure(0, weight=0)
    self.extract_tab_nav_row.grid_columnconfigure(1, weight=1)
    self.extract_tab_nav_row.grid_columnconfigure(2, weight=0)

    self.btn_back_to_wizard_step3 = ttk.Button(
        self.extract_tab_nav_row,
        text="← Wstecz",
        command=self._return_to_t05_work_after_step3_pz1,
        state=tk.DISABLED,
        style="WorkflowCard.TButton"
    )
    self.btn_back_to_wizard_step3.grid(row=0, column=0, sticky="w")
    self.btn_back_to_wizard_step3.grid_remove()
    self.btn_back_to_wizard_step3.configure(
        text="Zatwierdź powyższą pracę i przejdź do następnego kroku",
        padding=(8, 2),
        width=max(NAV_BUTTON_WIDTH, 42),
    )

    self.btn_to_detect_frame = tk.Frame(self.extract_tab_nav_row, bd=0, highlightthickness=0)
    self.btn_to_detect_frame.grid(row=0, column=2, sticky="e")

    self.btn_to_detect_pulse_frame = tk.Frame(
        self.btn_to_detect_frame,
        bd=0,
        highlightthickness=0
    )
    self.btn_to_detect_pulse_frame.pack(anchor=tk.E)

    self.btn_to_detect = ttk.Button(
        self.btn_to_detect_pulse_frame,
        text="Dalej → Wykrywanie Znaków i Analiza",
        command=self.go_to_substep_2,
        state=tk.DISABLED,
        style="WorkflowCardPrimary.TButton"
    )
    self.btn_to_detect.pack()
    self.btn_to_detect.config(text="Przejdź do PZ2")
    self.btn_to_detect_frame.grid_remove()
    self.btn_to_detect.config(text="Przejdź do PZ2 →")

    self.btn_to_detect.config(text="Przejdź do PZ2")

    self.btn_to_detect.config(text="Dalej do PZ2", padding=(8, 2), width=NAV_BUTTON_WIDTH)

    self.btn_back_to_wizard_step3.config(
        text="Zatwierdź powyższą pracę i przejdź do następnego kroku",
        width=max(NAV_BUTTON_WIDTH, 42),
    )
    HELP.bind_help(self.extract_step_nav_row, "t2_extract_nav")
    HELP.bind_help(self.extract_step_back_btn, "t2_extract_nav")
    HELP.bind_help(self.extract_step_next_btn, "t2_extract_nav")
    HELP.bind_help(self.btn_back_to_wizard_step3, "t2_extract_nav")
    HELP.bind_help(self.btn_to_detect, "t2_to_dataset")
    try:
        self._refresh_extract_workflow_ui()
        if not str(self.extract_workflow_shell.winfo_manager()):
            self.extract_workflow_shell.pack(fill=tk.X)
        _finish_extract_entry_cards_first_paint()
    except Exception as exc:
        logger.debug(f"Nie udało się wykonać pierwszego odświeżenia PZ1: {exc}")
