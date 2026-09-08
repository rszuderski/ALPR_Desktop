from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

from ..campaign_manager import CAMPAIGN
from ..config import logger
from .web_slim_scrollbar import blend_hex_colors
from .z3_view_models import (
    Step3ExtractStepCardViewModel,
    Step3ExtractWorkflowViewModel,
    Step3Pz3DatasetModeViewModel,
    Step3Pz3StatusPanelViewModel,
)

if TYPE_CHECKING:
    from .tab_character_annotation import CharacterAnnotationTab


def create_step3_widgets(host: "CharacterAnnotationTab") -> None:
    host.main_nb = ttk.Notebook(host.frame)
    host.main_nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 0))

    host.tab_extract = ttk.Frame(host.main_nb)
    host.main_nb.add(host.tab_extract, text="[PZ1] Wyodrębnianie zaanotowanych tablic")
    host._build_extraction_tab(host.tab_extract)

    host.tab_detect = ttk.Frame(host.main_nb)
    host.main_nb.add(host.tab_detect, text="[PZ2] Wykrywanie znaków i analiza")
    host._build_lazy_subtab_placeholder(
        host.tab_detect,
        "PZ2 zostanie przygotowane przy pierwszym wejściu do wykrywania znaków.",
    )

    host.tab_dataset = ttk.Frame(host.main_nb)
    host.main_nb.add(host.tab_dataset, text="[PZ3] Integracje i dataset (YOLO)")
    host._build_lazy_subtab_placeholder(
        host.tab_dataset,
        "PZ3 zostanie przygotowane przy pierwszym wejściu do eksportu datasetu znaków.",
    )
    host.main_nb.bind("<<NotebookTabChanged>>", host._on_main_nb_tab_changed, add="+")


def clear_lazy_subtab_placeholder(host: "CharacterAnnotationTab", parent) -> None:
    try:
        for child in parent.winfo_children():
            child.destroy()
    except Exception:
        pass


def build_lazy_subtab_placeholder(host: "CharacterAnnotationTab", parent, message: str) -> None:
    host._clear_lazy_subtab_placeholder(parent)
    palette = getattr(host.app, "palette", {})
    shell_bg = palette.get("panel", "#252526")
    text_fg = palette.get("muted", "#9ca3af")
    frame = ttk.Frame(parent, style="Panel.TFrame")
    frame.pack(fill=tk.BOTH, expand=True)
    inner = tk.Frame(frame, bg=shell_bg, bd=0, highlightthickness=0)
    inner.place(relx=0.5, rely=0.42, anchor=tk.CENTER)
    tk.Label(
        inner,
        text=message,
        bg=shell_bg,
        fg=text_fg,
        font=("Segoe UI", 10),
        justify=tk.CENTER,
        wraplength=560,
    ).pack(padx=24, pady=18)


def is_step3_campaign_runtime(host: "CharacterAnnotationTab") -> bool:
    try:
        return bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
    except Exception:
        return False


def get_step3_pz2_intro_text(host: "CharacterAnnotationTab") -> str:
    if is_step3_campaign_runtime(host):
        return (
            "T05 składa się z dwóch kroków. Tutaj, w PZ2, przygotowujesz anotacje znaków na tablicach: "
            "poprawiasz ramki, wpisujesz znaki i doprowadzasz tablice do statusu perfect. "
            "Gdy zbiór PZ2 jest sensowny, użyj przycisku „Krok 2: dataset PZ3”, aby przejść do PZ3 i utworzyć źródłowy dataset znaków."
        )
    return (
        "PZ2 przygotowuje anotacje znaków na wyodrębnionych tablicach: poprawiasz ramki, wpisujesz znaki "
        "i doprowadzasz tablice do statusu perfect. Gdy zbiór PZ2 jest sensowny, przejdź do PZ3 i utwórz źródłowy dataset znaków."
    )


def ensure_step3_subtab_built(host: "CharacterAnnotationTab", tab_widget) -> bool:
    if tab_widget is getattr(host, "tab_detect", None):
        return host._ensure_detect_tab_built()
    if tab_widget is getattr(host, "tab_dataset", None):
        return host._ensure_dataset_tab_built()
    return True


def ensure_detect_tab_built(host: "CharacterAnnotationTab") -> bool:
    if bool(getattr(host, "_detect_tab_built", False)):
        return True

    build_start = time.perf_counter()
    ui_ms = refresh_ms = restore_ms = autoload_ms = 0.0
    host._clear_lazy_subtab_placeholder(host.tab_detect)
    host._detect_tab_built = True
    try:
        phase_start = time.perf_counter()
        host._build_detection_tab(host.tab_detect)
        ui_ms = (time.perf_counter() - phase_start) * 1000.0
    except Exception as exc:
        host._detect_tab_built = False
        logger.error(f"Nie udało się zbudować Z3/PZ2: {exc}")
        host._build_lazy_subtab_placeholder(
            host.tab_detect,
            "Nie udało się przygotować PZ2. Wróć do tej podzakładki po sprawdzeniu logu.",
        )
        return False

    for refresh in (
        host._update_yolo_visibility,
        host._update_preview_path_lock,
        host._refresh_preview_bound_action_states,
        host._refresh_step3_mode_specific_ui,
    ):
        try:
            phase_start = time.perf_counter()
            refresh()
            refresh_ms += (time.perf_counter() - phase_start) * 1000.0
        except Exception:
            pass

    campaign_controlled_preview = False
    try:
        campaign_controlled_preview = bool(
            CAMPAIGN.get_active_project_name()
            and getattr(host, "_step3_linear_mode", False)
        )
    except Exception:
        campaign_controlled_preview = False

    try:
        if CAMPAIGN.get_active_project_name() and not campaign_controlled_preview:
            phase_start = time.perf_counter()
            host._restore_preview_context_from_project(require_plates=True)
            restore_ms = (time.perf_counter() - phase_start) * 1000.0
    except Exception:
        pass

    if host.preview_dir_var.get().strip() and not campaign_controlled_preview:
        try:
            phase_start = time.perf_counter()
            host._schedule_detection_preview_autoload()
            autoload_ms = (time.perf_counter() - phase_start) * 1000.0
        except Exception:
            pass
    elapsed_ms = (time.perf_counter() - build_start) * 1000.0
    if elapsed_ms >= 250.0:
        try:
            logger.info(
                "[Z3/PZ2 build] total=%.1fms ui=%.1fms refresh=%.1fms restore=%.1fms autoload=%.1fms campaign_controlled=%s",
                elapsed_ms,
                ui_ms,
                refresh_ms,
                restore_ms,
                autoload_ms,
                bool(campaign_controlled_preview),
            )
        except Exception:
            pass
    return True


def ensure_dataset_tab_built(host: "CharacterAnnotationTab") -> bool:
    if bool(getattr(host, "_dataset_tab_built", False)):
        return True

    host._clear_lazy_subtab_placeholder(host.tab_dataset)
    host._dataset_tab_built = True
    try:
        host._build_cvat_tab(host.tab_dataset)
    except Exception as exc:
        host._dataset_tab_built = False
        logger.error(f"Nie udało się zbudować Z3/PZ3: {exc}")
        host._build_lazy_subtab_placeholder(
            host.tab_dataset,
            "Nie udało się przygotować PZ3. Wróć do tej podzakładki po sprawdzeniu logu.",
        )
        return False

    for refresh_name in (
        "_refresh_preview_bound_action_states",
        "_refresh_pz3_dataset_mode_ui",
        "_refresh_pz3_cards_ui",
        "_refresh_pz3_status_panel_ui",
        "_refresh_step3_mode_specific_ui",
    ):
        refresh = getattr(host, refresh_name, None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass
    return True


def paint_extract_entry_cards_first(host: "CharacterAnnotationTab") -> None:
    cards = getattr(host, "_extract_entry_cards", {}) or {}
    if not cards:
        return

    palette = getattr(host.app, "palette", {})
    card_bg = palette.get("panel_alt", palette.get("panel", "#252526"))
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")

    for widgets in cards.values():
        for widget_name in ("frame", "badge", "title", "desc"):
            widget = widgets.get(widget_name)
            if widget is None:
                continue
            try:
                widget.configure(
                    bg=card_bg,
                    highlightbackground=border,
                    highlightcolor=border,
                )
            except Exception:
                pass
        for widget_name, color in (("badge", muted), ("title", fg), ("desc", muted)):
            widget = widgets.get(widget_name)
            if widget is None:
                continue
            try:
                widget.configure(fg=color)
            except Exception:
                pass


def build_step3_local_success_message(title: str, target_path, next_hint: str) -> str:
    lines = [f"Proces zakończył się sukcesem: {title}"]
    if target_path:
        lines.append(f"Ścieżka: {target_path}")
    if next_hint:
        lines.append(f"Dalej: {next_hint}")
    return "\n".join(lines)


def hide_pz3_status_summary_section(host: "CharacterAnnotationTab") -> None:
    if bool(getattr(host, "_pz3_status_section_persistent", False)):
        return
    section = getattr(host, "pz3_status_section", None)
    if section is None:
        return
    try:
        if str(section.winfo_manager()):
            section.pack_forget()
    except Exception:
        pass


def show_pz3_operation_summary_modal(host: "CharacterAnnotationTab", title: str, message: str, tone: str = "info") -> None:
    body = str(message or "").strip()
    if not body:
        return
    dedupe_key = (str(title or ""), body, str(tone or "info"))
    if getattr(host, "_last_pz3_operation_modal_key", None) == dedupe_key:
        return
    host._last_pz3_operation_modal_key = dedupe_key
    try:
        presenter = getattr(host.app, "themed_info", None)
        if callable(presenter):
            presenter(title or "PZ3", body, parent=host.frame, tone=tone or "info")
        else:
            messagebox.showinfo(title or "PZ3", body)
    except Exception:
        pass


def remember_pz3_operation_message(host: "CharacterAnnotationTab", console_widget, message: str, tone: str = "info") -> None:
    if console_widget is getattr(host, "export_console", None):
        host._pz3_last_export_summary = str(message or "")
        status_prefix = "PZ3 eksport"
    elif console_widget is getattr(host, "import_console", None):
        host._pz3_last_import_summary = str(message or "")
        status_prefix = "PZ3 import"
    else:
        return

    first_line = next((line.strip() for line in str(message or "").splitlines() if line.strip()), "")
    if not first_line:
        return
    try:
        host.app.update_status(f"{status_prefix}: {first_line}", tone or "info")
    except Exception:
        pass


def log_message(host: "CharacterAnnotationTab", txt_widget, msg: str, tag: str = "INFO") -> None:
    try:
        if hasattr(host.app, "append_global_terminal"):
            host.app.append_global_terminal(msg, source="PZ2")
    except Exception:
        pass

    if txt_widget in (getattr(host, "export_console", None), getattr(host, "import_console", None)):
        host._hide_pz3_status_summary_section()
        if not isinstance(txt_widget, tk.Text):
            return

    def do_log():
        try:
            txt_widget.configure(state="normal")

            if not txt_widget.tag_names():
                txt_widget.tag_config("SUCCESS", foreground="#27ae60", font=("Consolas", 9, "bold"))
                txt_widget.tag_config("ERROR", foreground="#c0392b", font=("Consolas", 9, "bold"))
                txt_widget.tag_config("WARNING", foreground="#d35400", font=("Consolas", 9, "bold"))
                txt_widget.tag_config("INFO", foreground="#2980b9", font=("Consolas", 9))
                txt_widget.tag_config("HEADER", foreground="#8e44ad", font=("Consolas", 10, "bold"))

            final_tag = tag
            if tag == "INFO":
                if "✅" in msg:
                    final_tag = "SUCCESS"
                elif "❌" in msg:
                    final_tag = "ERROR"

            txt_widget.insert(tk.END, msg + "\n", final_tag)
            txt_widget.see(tk.END)
            txt_widget.update_idletasks()

        finally:
            try:
                txt_widget.configure(state="disabled")
            except Exception:
                pass

    host.frame.after(0, do_log)


def set_console_text(host: "CharacterAnnotationTab", console_widget, text) -> None:
    if console_widget in (getattr(host, "export_console", None), getattr(host, "import_console", None)):
        host._hide_pz3_status_summary_section()
    if isinstance(console_widget, tk.Text):
        console_widget.config(state=tk.NORMAL)
        console_widget.delete(1.0, tk.END)
        console_widget.insert(tk.END, text)
        console_widget.config(state=tk.DISABLED)
        host.frame.update()
        return

    msg = str(text or "").strip()
    tone = "muted"
    if any(token in msg for token in ("❌", "BŁĄD", "ERROR")):
        tone = "danger"
    elif any(token in msg for token in ("⚠", "OSTRZEŻ", "WARNING")):
        tone = "warning"
    elif any(token in msg for token in ("✅", "sukcesem", "gotowy", "gotowa", "gotowe")):
        tone = "success"

    try:
        host._set_inline_status_label_state(console_widget, text=msg, tone=tone, emphasis=False)
        console_widget.update_idletasks()
    except Exception:
        try:
            console_widget.configure(text=msg)
        except Exception:
            pass

    if console_widget in (getattr(host, "export_console", None), getattr(host, "import_console", None)):
        host._remember_pz3_operation_message(console_widget, msg, tone)
        if tone == "danger":
            title = "Błąd importu PZ3" if console_widget is getattr(host, "import_console", None) else "Błąd eksportu PZ3"
            host._show_pz3_operation_summary_modal(title, msg, tone="error")
        try:
            host._hide_pz3_status_summary_section()
            host._refresh_pz3_status_panel_ui()
        except Exception:
            pass


def set_button_state(host: "CharacterAnnotationTab", attr_name: str, enabled: bool) -> None:
    btn = getattr(host, attr_name, None)
    if btn is None:
        return

    try:
        btn.config(state="normal" if enabled else "disabled")
    except Exception as e:
        logger.debug(f"Nie udało się ustawić stanu przycisku '{attr_name}': {e}")


def set_subtab_state(host: "CharacterAnnotationTab", tab_widget, state: str) -> None:
    normalized_state = str(state or "").strip().lower() or "normal"
    try:
        if normalized_state == "disabled":
            selected = ""
            try:
                selected = str(host.main_nb.select())
            except Exception:
                selected = ""
            if selected == str(tab_widget):
                fallback_selected = False
                for fallback in (
                    getattr(host, "tab_extract", None),
                    getattr(host, "tab_detect", None),
                    getattr(host, "tab_dataset", None),
                ):
                    if fallback is None or fallback is tab_widget:
                        continue
                    try:
                        if str(host.main_nb.tab(str(fallback), "state")) == "normal":
                            host.main_nb.select(str(fallback))
                            fallback_selected = True
                            break
                    except Exception:
                        continue
                if not fallback_selected:
                    return
        host.main_nb.tab(str(tab_widget), state=normalized_state)
    except Exception as e:
        logger.debug(f"Nie udało się ustawić stanu podzakładki: {e}")


def get_subtab_state(host: "CharacterAnnotationTab", tab_widget) -> str:
    try:
        return str(host.main_nb.tab(str(tab_widget), "state"))
    except Exception:
        return "normal"


def _get_pz3_existing_dataset_status_from_summary(host: "CharacterAnnotationTab") -> tuple[str, str]:
    try:
        summary = dict(host._read_step3_export_summary() or {})
    except Exception:
        summary = {}

    dataset_path = str(summary.get("gold_dataset_path") or "").strip()
    if not dataset_path:
        return "", "muted"

    try:
        dataset_exists = Path(dataset_path).exists()
    except Exception:
        dataset_exists = False

    created = bool(summary.get("gold_dataset_created")) and bool(summary.get("gold_dataset_valid", True))
    if created and dataset_exists:
        plates = int(summary.get("exportable_plate_count", 0) or 0)
        chars = int(summary.get("exportable_char_count", 0) or 0)
        details = []
        if plates > 0:
            details.append(f"{plates} tablic")
        if chars > 0:
            details.append(f"{chars} znaków")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"Istniejący dataset PZ3: {Path(dataset_path).name}{suffix}.", "success"

    if bool(summary.get("gold_dataset_created")) and not dataset_exists:
        return "Podsumowanie wskazuje dataset PZ3, ale katalog nie istnieje na dysku.", "warning"

    if bool(summary.get("gold_dataset_created")) and not bool(summary.get("gold_dataset_valid", True)):
        message = str(summary.get("gold_dataset_validation_message") or "").strip()
        if message:
            return f"Dataset PZ3 wymaga sprawdzenia: {message}", "warning"
        return "Dataset PZ3 wymaga sprawdzenia.", "warning"

    return "", "muted"


def build_step3_pz3_dataset_mode_view_model(
    host: "CharacterAnnotationTab",
) -> Step3Pz3DatasetModeViewModel:
    in_campaign = is_step3_campaign_runtime(host)

    mode = "perfect"

    perfect_selected = mode == "perfect"

    if perfect_selected:
        try:
            host._get_active_preview_context()
        except Exception:
            pass
        export_pool = host._build_campaign_aware_gold_export_counts(
            selected_strategies=host._get_selected_gold_export_strategy_buckets(),
            selected_sources=host._get_selected_gold_export_source_buckets(),
        )
        selected_plate_count = int(export_pool.get("selected_plate_count", 0) or 0)
        selected_char_count = int(export_pool.get("selected_char_count", 0) or 0)
        export_ready = bool(selected_plate_count > 0 and selected_char_count > 0)
        existing_dataset_status, existing_dataset_status_tone = _get_pz3_existing_dataset_status_from_summary(host)
        try:
            preview_context = host._get_active_preview_context()
        except Exception:
            preview_context = {
                "ready": False,
                "message": "Nie udało się odczytać aktywnego preview PZ2.",
                "preview_dir": None,
                "plate_count": 0,
            }
        preview_ready = bool(preview_context.get("ready"))
        if preview_ready:
            preview_dir = preview_context.get("preview_dir")
            preview_name = preview_dir.name if isinstance(preview_dir, Path) else str(preview_dir or "aktywny wynik")
            source_preview_text = f"Wynik PZ2: {preview_name}."
            source_preview_tone = "success"
        else:
            source_preview_text = str(preview_context.get("message") or "Brak aktywnego wyniku PZ2.")
            source_preview_tone = "warning"

        if selected_plate_count > 0 or selected_char_count > 0:
            source_pool_text = "Materiał do datasetu: gotowy. Zakres wynika z aktualnych ustawień eksportu."
            source_pool_tone = "success" if export_ready else "warning"
        else:
            source_pool_text = (
                "Materiał do datasetu: brak gotowego zakresu. Wróć do PZ2 i doprowadź tablice "
                "do statusu perfect albo wczytaj poprawki z CVAT."
            )
            source_pool_tone = "warning"
        source_next_text = (
            "Następny krok: utwórz źródłowy dataset znaków. Zakres poniżej zmieniaj tylko świadomie."
            if export_ready
            else (
                "Następny krok: przygotuj wynik PZ2 z wyodrębnionymi tablicami. PZ3 nie wybiera "
                "źródła samodzielnie."
                if not preview_ready
                else (
                    "Następny krok: wróć do PZ2 i uzupełnij poprawne boxy znaków. Dataset wymaga "
                    "tablic perfect z co najmniej jednym eksportowalnym znakiem."
                    if selected_plate_count > 0
                    else "Następny krok: wróć do PZ2 i przygotuj co najmniej jedną tablicę perfect."
                )
            )
        )
        source_next_tone = "muted" if export_ready else "warning"
        return Step3Pz3DatasetModeViewModel(
            in_campaign=in_campaign,
            mode="perfect",
            dataset_source_title="",
            dataset_source_intro="",
            source_preview_text=source_preview_text,
            source_preview_tone=source_preview_tone,
            source_pool_text=source_pool_text,
            source_pool_tone=source_pool_tone,
            source_next_text=source_next_text,
            source_next_tone=source_next_tone,
            show_dataset_source_cards=False,
            perfect_selected=True,
            existing_selected=False,
            perfect_badge_text="PZ2",
            perfect_title_text="Materiał z PZ2",
            perfect_desc_text="Źródłem są wyodrębnione tablice z PZ2 oznaczone jako perfect.",
            existing_badge_text="Z4",
            existing_title_text="Warianty w Z4",
            existing_desc_text="Gotowe datasety i ich warianty wybierzesz w Z4.",
            show_existing_dataset_panel=False,
            cvat_option2_title="Stan materiału z PZ2",
            cvat_option2_tone="success",
            cvat_option2_desc=(
                "PZ3 korzysta z wyodrębnionych tablic z PZ2. Do datasetu wejdą tablice perfect "
                "z poprawnymi boxami znaków; wariant treningowy i split ustawisz później w Z4."
            ),
            show_gold_filters=True,
            split_title="",
            split_label="",
            primary_export_label="UTWÓRZ ŹRÓDŁOWY DATASET ZNAKÓW",
            primary_export_command_id="run_yolo_gold_export",
            primary_export_enabled=export_ready,
            primary_export_columnspan=1,
            show_classifier_export=True,
            classifier_export_enabled=bool(selected_char_count > 0),
            action_hint=(
                (
                    "Brama PZ3: materiał jest gotowy do utworzenia datasetu."
                    if in_campaign
                    else "Źródłowy dataset: materiał z aktualnej puli jest gotowy."
                )
                if (selected_plate_count > 0 or selected_char_count > 0)
                else (
                    "Brama PZ3: brak tablic perfect gotowych do datasetu."
                    if in_campaign
                    else "Źródłowy dataset: brak tablic perfect gotowych do utworzenia."
                )
            ),
            action_hint_tone=("muted" if export_ready else "warning"),
            existing_dataset_status=existing_dataset_status,
            existing_dataset_status_tone=existing_dataset_status_tone,
        )

    source_dir_raw = str(host.pz3_existing_dataset_var.get() or "").strip()
    source_info = host._inspect_pz3_dataset_source_dir(Path(source_dir_raw) if source_dir_raw else None)
    status_tone = "success" if source_info.get("ok") else ("warning" if source_dir_raw else "muted")
    action_hint = "Gotowe datasety i warianty treningowe obsługuje Z4."
    action_tone = status_tone if source_info.get("ok") else "muted"

    return Step3Pz3DatasetModeViewModel(
        in_campaign=in_campaign,
        mode="existing",
        dataset_source_title="",
        dataset_source_intro=(
            "PZ3 pracuje na tablicach perfect z aktywnego wyniku PZ2 oraz ręcznych importach. "
            "Gotowe datasety, warianty treningowe i split przygotujesz w Z4."
        ),
        source_preview_text="Ten tryb został przeniesiony do Z4.",
        source_preview_tone="muted",
        source_pool_text="W PZ3 zostaje tylko budowa datasetu źródłowego znaków.",
        source_pool_tone="muted",
        source_next_text="Wybierz źródło w Z4, jeśli chcesz pracować na gotowym datasecie.",
        source_next_tone="muted",
        show_dataset_source_cards=True,
        perfect_selected=False,
        existing_selected=True,
        perfect_badge_text="PZ2",
        perfect_title_text="Materiał z PZ2",
        perfect_desc_text="Źródłem są wyodrębnione tablice z PZ2 oznaczone jako perfect.",
        existing_badge_text="Z4",
        existing_title_text="Warianty w Z4",
        existing_desc_text="Gotowe datasety i ich warianty wybierzesz w Z4.",
        show_existing_dataset_panel=True,
        cvat_option2_title="Warianty w Z4",
        cvat_option2_tone="default",
        cvat_option2_desc="Ten tryb jest przeniesiony do Z4. W PZ3 tworzysz tylko źródłowy dataset znaków.",
        show_gold_filters=False,
        split_title="",
        split_label="",
        primary_export_label="WRÓĆ DO ŹRÓDŁOWEGO DATASETU PZ3",
        primary_export_command_id="run_yolo_gold_export",
        primary_export_enabled=False,
        primary_export_columnspan=2,
        show_classifier_export=False,
        action_hint=action_hint,
        action_hint_tone=action_tone,
        existing_dataset_status=str(source_info.get("message") or ""),
        existing_dataset_status_tone=status_tone,
    )


def build_step3_extract_workflow_view_model(
    host: "CharacterAnnotationTab",
) -> Step3ExtractWorkflowViewModel:
    current = host._get_extract_workflow_step(lightweight=True)
    route = host._get_extract_entry_mode()
    source_ready = bool(getattr(host, "_extract_last_source_binding_result", {}).get("ok"))
    try:
        has_preview = bool(host._is_extract_preview_ready_fast())
    except Exception:
        has_preview = bool(str(host.preview_dir_var.get() or "").strip())
    has_run = bool(str(host.annotation_run_dir_var.get() or "").strip())
    linear_mode = is_step3_campaign_runtime(host)
    candidate = host._get_preferred_z2_source_candidate()
    step3_status = ""
    if linear_mode:
        try:
            step3_status = str(CAMPAIGN.get_step3_status() or "").strip().lower()
        except Exception:
            step3_status = ""

    if linear_mode and step3_status == "needs_rework":
        source_intro = (
            "Wróciłeś do etapu 3 po poprawkach tablic w Z2. Z3 nadal pracuje na tej samej iteracji "
            "i użyje odświeżonego źródła tablic automatycznie."
        )
        source_hint = (
            "Nie musisz ponownie wskazywać runu, XML ani folderu obrazów. "
            "W tym kroku przebudujesz tylko zestaw wyodrębnionych tablic potrzebny do dalszej pracy nad znakami."
        )
    elif linear_mode:
        source_intro = (
            "Kampania ma już przypięte źródło tablic dla tego etapu. "
            "Nie musisz ręcznie wybierać runu Z2 ani przepisywać ścieżek."
        )
        source_hint = (
            "Gdy źródło tablic będzie spójne, Z3 samo przygotuje zestaw wyodrębnionych tablic dla PZ2."
        )
    elif route == "continue":
        if has_preview:
            source_intro = (
                "PZ2 jest już gotowe dla przejętego runu Z2. "
                "PZ1 pokazuje tutaj tabelę kontrolną źródła i wynik wyodrębniania, ale nie wymaga ponownej pracy."
            )
            source_hint = (
                "Sprawdź tabelę i przejdź do PZ2. Jeśli chcesz świadomie przygotować inny zestaw wyodrębnionych tablic, wróć do wyboru "
                "i użyj kafla 'Wskaż anotacje do wyodrębnienia'."
            )
        else:
            source_intro = (
                "Kontynuacja przejmuje aktywne albo ostatnie źródło z Z2. "
                "Nie wskazujesz tutaj ręcznie folderu runu, XML ani obrazów; PZ1 tylko pokazuje, co zostało przejęte."
            )
            source_hint = (
                "Jeśli chcesz pracować na innym XML lub innym katalogu obrazów, wróć do wyboru i użyj kafla "
                "'Wskaż anotacje do wyodrębnienia'."
            )
    else:
        source_intro = "W tym trybie wskazujesz annotations.xml oraz oryginalny katalog obrazów. System dopilnuje zgodności XML z katalogiem."
        source_hint = "Po wskazaniu XML mogę dodatkowo spróbować dopasować katalog obrazów z Workspace/1_raw_images/."

    if linear_mode:
        if has_preview:
            start_hint = (
                "Zestaw wyodrębnionych tablic dla znaków jest już odświeżony. "
                "Za chwilę otwieram PZ2, aby kontynuować OCR i korektę znaków."
            )
        elif source_ready:
            start_hint = (
                "Źródło tablic tej iteracji jest już gotowe. "
                "Wyodrębnianie uruchomi się automatycznie i odświeży zestaw wyodrębnionych tablic dla PZ2."
            )
        else:
            start_hint = (
                "Czekam na spójne źródło tablic dla tej iteracji. "
                "Gdy będzie gotowe, ponowne wyodrębnianie uruchomi się automatycznie."
            )
    elif route == "continue":
        if has_run and source_ready:
            run_name = Path(str(host.annotation_run_dir_var.get() or "")).name
            if has_preview:
                start_hint = (
                    f"Przejęty run Z2: {run_name}. Tablice są już wyodrębnione, a tabela poniżej potwierdza źródło i gotowy wynik dla PZ2."
                )
            else:
                start_hint = (
                    f"Przejęty run Z2: {run_name}. Sprawdź tabelę i wybierz jedną z akcji na dole karty."
                )
        else:
            start_hint = (
                "Nie widzę jeszcze aktywnego źródła z Z2. "
                "Wróć do wyboru i użyj trybu ręcznego wskazania XML oraz obrazów."
            )
    else:
        if has_preview:
            start_hint = (
                "Katalog wyodrębnionych tablic dla PZ2 jest już gotowy. "
                "Ponowne wyodrębnianie jest zablokowane dla tego zestawu."
            )
        else:
            start_hint = (
                "Źródła są gotowe. Możesz uruchomić wyodrębnianie i przygotować katalog wyodrębnionych tablic do PZ2."
                if source_ready
                else "Najpierw potwierdź zgodność źródeł, a potem uruchom wyodrębnianie."
            )

    if linear_mode and step3_status == "needs_rework":
        run_hint = (
            "Źródło tablic dla tej iteracji jest już podpięte automatycznie. "
            "Po tym kroku odświeżysz zestaw wyodrębnionych tablic, a potem wrócisz do OCR i korekty znaków."
        )
        run_hint_tone = "success"
    elif linear_mode:
        run_hint = "Źródło tablic dla tej iteracji jest obsługiwane automatycznie przez kampanię."
        run_hint_tone = "success"
    elif route == "continue":
        if has_run:
            run_name = Path(str(host.annotation_run_dir_var.get() or "")).name
            if has_preview:
                run_hint = (
                    f"Wybrany run anotacji: {run_name}. PZ2 jest już gotowe; XML, obrazy i wynik wyodrębniania "
                    "są pokazane w tabeli kontrolnej."
                )
            else:
                run_hint = f"Wybrany run anotacji: {run_name}. XML i katalog obrazów są wyprowadzone z tego runu."
            run_hint_tone = "success"
        elif candidate:
            run_hint = (
                f"Dostępne jest {candidate.get('label', 'źródło z Z2')}. "
                "PZ1 przejmie je automatycznie w trybie kontynuacji."
            )
            run_hint_tone = "muted"
        else:
            run_hint = (
                "Nie znaleziono aktywnego runu Z2. Zakończ pracę w Z2 albo wróć do wyboru i użyj trybu ręcznego "
                "z annotations.xml oraz zgodnym katalogiem obrazów."
            )
            run_hint_tone = "muted"
    else:
        run_hint = "Ten tryb nie wymaga folderu runu anotacji. Wystarczy annotations.xml oraz zgodny katalog obrazów."
        run_hint_tone = "muted"

    step_state = {
        "entry": "done" if route else "active",
        "source": "active",
        "start": "pending",
    }
    if source_ready:
        step_state["source"] = "done"
        step_state["start"] = "done" if has_preview else "active"

    if linear_mode:
        source_title = "Potwierdź źródło tablic"
        start_title = "Przebuduj zestaw wyodrębnionych tablic"
        entry_card_description = "Kampania prowadzi ten etap na gotowym źródle tablic."
        source_card_description = "Źródło tablic tej iteracji jest obsługiwane automatycznie."
        start_card_description = (
            "Zestaw wyodrębnionych tablic dla PZ2 jest już odświeżony." if has_preview
            else "Odśwież zestaw wyodrębnionych tablic, aby wrócić do OCR i korekty znaków."
        )
    else:
        source_title = "Źródła wejścia"
        start_title = (
            "Tablice wyodrębnione dla PZ2" if route == "continue" and has_preview
            else "Przejęty run anotacji Z2" if route == "continue"
            else "Uruchom wyodrębnianie"
        )
        if route == "continue" and has_preview:
            entry_card_description = (
                "Tablice są już wyodrębnione. PZ1 jest teraz podsumowaniem źródła i wyniku, "
                "a dalsza praca nad znakami odbywa się w PZ2."
            )
        elif route == "continue":
            entry_card_description = "Kontynuacja po runie anotacji Z2 bez ręcznego wybierania źródeł."
        else:
            entry_card_description = "Wskażesz annotations.xml i obrazy."
        if route == "continue" and has_preview:
            source_card_description = (
                "Tabela pokazuje, z którego runu Z2 powstał aktywny zestaw wyodrębnionych tablic "
                "oraz ile materiału trafiło do PZ2."
            )
        elif route == "continue" and source_ready:
            source_card_description = "Run anotacji jest potwierdzony i gotowy do wyodrębniania tablic."
        elif route == "continue":
            source_card_description = "Podłącz run Z2; XML i obrazy zostaną wyprowadzone automatycznie."
        else:
            source_card_description = "Powiąż annotations.xml z katalogiem obrazów."
        if route == "continue":
            start_card_description = (
                "Wyodrębnianie zostało już wykonane. Przejdź do PZ2, aby oznaczać znaki na wyodrębnionych tablicach." if has_preview
                else "Sprawdź przejęty run i wyodrębnij tablice do PZ2."
            )
        else:
            start_card_description = (
                "Katalog wyodrębnionych tablic do PZ2 jest już gotowy." if has_preview
                else "Wyodrębnij tablice i przygotuj preview do PZ2."
            )

    next_enabled = False
    if current == "entry":
        next_enabled = bool(route)
    elif current == "source":
        next_enabled = bool(route and (source_ready or has_preview))

    return Step3ExtractWorkflowViewModel(
        route=route,
        current_step=current,
        linear_mode=linear_mode,
        show_entry=(current == "entry" and not linear_mode),
        show_source=(current == "source" and (route in {"manual", "continue"} or linear_mode)),
        show_start=(current == "start" or (linear_mode and current == "entry")),
        source_title=source_title,
        start_title=start_title,
        source_intro=source_intro,
        source_hint=source_hint,
        run_hint=run_hint,
        run_hint_tone=run_hint_tone,
        start_hint=start_hint,
        start_hint_tone=("success" if (source_ready or has_preview) else "muted"),
        show_step_nav=(current in {"source", "start"} and not (route == "continue" and current == "start")),
        prev_enabled=(current in {"source", "start"} and not (route == "continue" and current == "start")),
        next_enabled=next_enabled,
        next_visible=current != "start",
        show_tab_nav=(
            not linear_mode
            and (not (route == "continue" and current == "start") or has_preview)
        ),
        show_back_nav=False,
        show_detect_nav=not linear_mode,
        clear_detect_emphasis=not linear_mode,
        step_cards=[
            Step3ExtractStepCardViewModel(
                key="entry",
                state=str(step_state.get("entry", "pending")),
                title="Wybierz wejście",
                description=entry_card_description,
            ),
            Step3ExtractStepCardViewModel(
                key="source",
                state=str(step_state.get("source", "pending")),
                title="Potwierdź źródła",
                description=source_card_description,
            ),
            Step3ExtractStepCardViewModel(
                key="start",
                state=str(step_state.get("start", "pending")),
                title=("PZ2 gotowe" if route == "continue" and has_preview else "Uruchom wyodrębnianie"),
                description=start_card_description,
            ),
        ],
    )


def refresh_extract_step_nav_buttons(
    host: "CharacterAnnotationTab",
    vm: Step3ExtractWorkflowViewModel | None = None,
):
    workflow_vm = vm or host._get_step3_extract_workflow_view_model()
    nav_row = getattr(host, "extract_step_nav_row", None)
    prev_btn = getattr(host, "extract_step_back_btn", None)
    next_btn = getattr(host, "extract_step_next_btn", None)
    main_nav_panel = getattr(host, "extract_main_nav_panel", None)
    back_btn = getattr(host, "btn_back_to_wizard_step3", None)
    detect_frame = getattr(host, "btn_to_detect_frame", None)

    if prev_btn is not None:
        try:
            prev_btn.config(state=tk.NORMAL if workflow_vm.prev_enabled else tk.DISABLED)
        except Exception:
            pass

    if next_btn is not None:
        try:
            next_btn.config(state=tk.NORMAL if workflow_vm.next_enabled else tk.DISABLED)
        except Exception:
            pass

        try:
            if not workflow_vm.next_visible:
                if str(next_btn.winfo_manager()):
                    next_btn.pack_forget()
            elif not str(next_btn.winfo_manager()):
                next_btn.pack(side=tk.LEFT, padx=(8, 0))
        except Exception:
            pass

    if back_btn is not None:
        try:
            if workflow_vm.show_back_nav:
                if not str(back_btn.winfo_manager()):
                    back_btn.grid(row=0, column=0, sticky="w")
            elif str(back_btn.winfo_manager()):
                back_btn.grid_remove()
        except Exception:
            pass

    if detect_frame is not None:
        try:
            if not workflow_vm.show_detect_nav:
                if str(detect_frame.winfo_manager()):
                    detect_frame.grid_remove()
            elif not str(detect_frame.winfo_manager()):
                detect_frame.grid(row=0, column=2, sticky="e")
        except Exception:
            pass

    try:
        if nav_row is not None and str(nav_row.winfo_manager()):
            nav_row.pack_forget()
        if main_nav_panel is not None and str(main_nav_panel.winfo_manager()):
            main_nav_panel.grid_remove()

        if workflow_vm.show_step_nav and nav_row is not None:
            nav_row.pack(fill=tk.X, pady=(0, 8))

        if workflow_vm.show_tab_nav and main_nav_panel is not None:
            main_nav_panel.grid()
    except Exception:
        pass

    if workflow_vm.clear_detect_emphasis:
        try:
            host._set_button_emphasis("btn_to_detect_frame", False)
        except Exception:
            pass
    try:
        host._refresh_campaign_step3_navigation_visibility()
    except Exception:
        pass


def refresh_extract_step_cards(
    host: "CharacterAnnotationTab",
    vm: Step3ExtractWorkflowViewModel | None = None,
):
    cards = getattr(host, "_extract_step_cards", []) or []
    if not cards:
        return

    palette = getattr(host.app, "palette", {})
    panel = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    hover_bg = palette.get("button_hover", panel_alt)
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    active_bg = blend_hex_colors(panel_alt, hover_bg, 0.42)
    done_bg = blend_hex_colors(panel_alt, hover_bg, 0.18)
    workflow_vm = vm or host._get_step3_extract_workflow_view_model()
    step_cards_by_key = {
        str(card.key or ""): card
        for card in list(getattr(workflow_vm, "step_cards", []) or [])
    }

    for entry in cards:
        key = str(entry.get("key") or "").strip()
        card_vm = step_cards_by_key.get(key)
        state = str(getattr(card_vm, "state", "pending") or "pending")
        frame = entry.get("frame")
        badge = entry.get("badge")
        title_lbl = entry.get("title")
        desc_lbl = entry.get("desc")

        if state == "done":
            card_bg = done_bg
            border_color = border
            badge_fg = fg
            title_fg = fg
            desc_fg = fg
        elif state == "active":
            card_bg = active_bg
            border_color = border
            badge_fg = fg
            title_fg = fg
            desc_fg = fg
        else:
            card_bg = panel_alt
            border_color = border
            badge_fg = muted
            title_fg = fg
            desc_fg = muted

        for widget in (frame, badge, title_lbl, desc_lbl):
            if widget is None:
                continue
            try:
                widget.configure(bg=card_bg, highlightbackground=border_color, highlightcolor=border_color)
            except Exception:
                pass

        try:
            badge.configure(fg=badge_fg)
        except Exception:
            pass
        try:
            title_lbl.configure(
                fg=title_fg,
                text=str(getattr(card_vm, "title", "") or entry.get("default_title", "")),
            )
        except Exception:
            pass
        try:
            desc_lbl.configure(
                fg=desc_fg,
                text=str(getattr(card_vm, "description", "") or entry.get("default_desc", "")),
            )
        except Exception:
            pass


def refresh_pz3_status_panel_ui(
    host: "CharacterAnnotationTab",
    vm: Step3Pz3StatusPanelViewModel | None = None,
):
    if vm is None and getattr(host, "_dataset_tab_built", None) is False:
        return
    status_vm = vm or host._get_step3_pz3_status_panel_view_model()

    try:
        title_lbl = getattr(host, "dataset_status_title_lbl", None)
        if title_lbl is not None:
            title_lbl.configure(text=str(status_vm.title or "Podsumowanie"))
    except Exception:
        pass

    for key_widget_name, value_widget_name, row_vm in (
        ("run_status_key_lbl", "run_console", status_vm.run_row),
        ("dataset_status_key_lbl", "dataset_console", status_vm.dataset_row),
        ("export_status_key_lbl", "export_console", status_vm.export_row),
        ("readiness_status_key_lbl", "readiness_console", status_vm.readiness_row),
        ("import_status_key_lbl", "import_console", status_vm.import_row),
    ):
        key_widget = getattr(host, key_widget_name, None)
        if key_widget is not None:
            try:
                key_widget.configure(text=str(row_vm.label or ""))
            except Exception:
                pass

        widget = getattr(host, value_widget_name, None)
        if widget is None:
            continue
        try:
            host._set_inline_status_label_state(
                widget,
                text=str(row_vm.text or ""),
                tone=str(row_vm.tone or "muted"),
                emphasis=False,
            )
        except Exception:
            try:
                widget.configure(text=str(row_vm.text or ""))
            except Exception:
                pass

    btn = getattr(host, "btn_finish_step3", None)
    btn_frame = getattr(host, "btn_finish_step3_frame", None)
    card_frame = getattr(host, "step3_finish_card", None)
    action_card = getattr(host, "step3_finish_action_card", None)
    finish_vm = status_vm.finish_action
    if btn is None:
        return

    try:
        if card_frame is not None:
            if bool(finish_vm.visible):
                if not str(card_frame.winfo_manager()):
                    card_frame.pack(anchor=tk.E)
            elif str(card_frame.winfo_manager()):
                card_frame.pack_forget()
        if btn_frame is not None:
            if bool(finish_vm.visible):
                if not str(btn_frame.winfo_manager()):
                    btn_frame.pack(fill=tk.X)
            elif str(btn_frame.winfo_manager()):
                btn_frame.pack_forget()
    except Exception:
        pass

    if not bool(finish_vm.visible):
        try:
            if action_card is not None:
                action_card.set_emphasis(False)
            else:
                host._set_button_emphasis("btn_finish_step3_frame", False)
        except Exception:
            pass
        host._set_step3_finish_hint("")
        return

    try:
        command = host._resolve_step3_campaign_action_command(finish_vm.command_id)
        label_text = str(finish_vm.label or "")
        target_style = "Accent.TButton" if bool(finish_vm.emphasize) else "WorkflowCard.TButton"
        base_width = host.NAV_BUTTON_WIDTH + 2 if hasattr(host, "NAV_BUTTON_WIDTH") else 20
        button_width = max(base_width, min(42, max(28, len(label_text) + 2)))
        if action_card is not None:
            action_card.configure_action(
                text=label_text,
                command=command,
                state=("normal" if bool(finish_vm.enabled) else "disabled"),
                width=button_width,
                style=target_style,
                padding=(10, 3),
            )
        else:
            btn.config(
                text=label_text,
                command=command,
                state=("normal" if bool(finish_vm.enabled) else "disabled"),
                width=button_width,
                style=target_style,
            )
            try:
                btn.configure(padding=(8, 2))
            except Exception:
                pass

        if bool(finish_vm.emphasize):
            if action_card is not None:
                action_card.set_emphasis(True)
            else:
                host._set_button_emphasis("btn_finish_step3_frame", True)
            host._set_step3_finish_hint("")
        else:
            if action_card is not None:
                action_card.set_emphasis(False)
            else:
                host._set_button_emphasis("btn_finish_step3_frame", False)
            host._set_step3_finish_hint(str(finish_vm.hint or ""), tone=str(finish_vm.hint_tone or "muted"))
    except Exception as exc:
        try:
            logger.debug(f"Nie udało się odświeżyć panelu statusu PZ3: {exc}")
        except Exception:
            pass


def refresh_pz3_dataset_source_card(
    host: "CharacterAnnotationTab",
    card_info,
    *,
    selected: bool,
    hovered: bool,
    badge_text: str,
    title_text: str,
    desc_text: str,
):
    card_palette = host._get_pz3_card_palette()
    current_card_bg = card_palette["card_active_bg"] if selected else (
        card_palette["card_hover_bg"] if hovered else card_palette["card_bg"]
    )
    border_color = card_palette["card_border"]
    for widget in (
        card_info.get("frame"),
        card_info.get("top_row"),
        card_info.get("badge"),
        card_info.get("state_badge"),
        card_info.get("title"),
        card_info.get("desc"),
    ):
        if widget is None:
            continue
        try:
            widget.configure(
                bg=current_card_bg,
                highlightbackground=border_color,
                highlightcolor=border_color,
                cursor="hand2",
            )
        except Exception:
            pass
    try:
        card_info["badge"].configure(
            text=host._normalize_pz3_card_badge_text(badge_text),
            fg=(card_palette["card_fg"] if selected else card_palette["card_muted"]),
        )
    except Exception:
        pass
    host._refresh_pz3_card_state_badge(
        card_info,
        selected=bool(selected),
        current_card_bg=current_card_bg,
        card_palette=card_palette,
    )
    try:
        card_info["title"].configure(text=title_text, fg=card_palette["card_fg"])
    except Exception:
        pass
    host._set_inline_status_label_state(
        card_info["desc"],
        text=desc_text,
        tone=("default" if selected else "muted"),
        emphasis=False,
    )
    try:
        card_info["desc"].configure(fg=(card_palette["card_fg"] if selected else card_palette["card_muted"]))
    except Exception:
        pass


def refresh_pz3_dataset_mode_ui(host: "CharacterAnnotationTab"):
    vm = host._get_step3_pz3_dataset_mode_view_model()
    dataset_source_lf = getattr(host, "_pz3_dataset_source_lf", None)
    dataset_source_cards = getattr(host, "_pz3_dataset_source_cards", None)
    dataset_source_perfect_card = getattr(host, "_pz3_dataset_source_perfect_card", None)
    dataset_source_existing_card = getattr(host, "_pz3_dataset_source_existing_card", None)
    dataset_source_card_state = getattr(host, "_pz3_dataset_source_card_state", {}) or {}

    try:
        if dataset_source_lf is not None:
            dataset_source_lf.configure(text=str(vm.dataset_source_title or ""))
    except Exception:
        pass
    intro_widget = getattr(host, "pz3_dataset_source_intro_lbl", None)
    intro_text = str(vm.dataset_source_intro or "").strip()
    if intro_widget is not None:
        if intro_text:
            try:
                if not str(intro_widget.winfo_manager()):
                    intro_widget.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))
            except Exception:
                pass
            host._set_inline_status_label_state(
                intro_widget,
                text=intro_text,
                tone="muted",
                emphasis=False,
            )
        else:
            try:
                if str(intro_widget.winfo_manager()):
                    intro_widget.pack_forget()
            except Exception:
                pass
    for widget_name, text_value, tone_value in (
        ("pz3_source_preview_status_lbl", vm.source_preview_text, vm.source_preview_tone),
        ("pz3_source_pool_status_lbl", vm.source_pool_text, vm.source_pool_tone),
        ("pz3_source_next_status_lbl", vm.source_next_text, vm.source_next_tone),
    ):
        widget = getattr(host, widget_name, None)
        if widget is None:
            continue
        host._set_inline_status_label_state(
            widget,
            text=str(text_value or ""),
            tone=str(tone_value or "muted"),
            emphasis=False,
        )

    try:
        if dataset_source_cards is not None:
            if not bool(vm.show_dataset_source_cards):
                if str(dataset_source_cards.winfo_manager()):
                    dataset_source_cards.pack_forget()
            elif not str(dataset_source_cards.winfo_manager()):
                dataset_source_cards.pack(fill=tk.X)
    except Exception:
        pass

    if dataset_source_perfect_card is not None:
        refresh_pz3_dataset_source_card(
            host,
            dataset_source_perfect_card,
            selected=bool(vm.perfect_selected),
            hovered=bool(dataset_source_card_state.get("perfect_hovered", False)),
            badge_text=str(vm.perfect_badge_text or ""),
            title_text=str(vm.perfect_title_text or ""),
            desc_text=str(vm.perfect_desc_text or ""),
        )
    if dataset_source_existing_card is not None:
        refresh_pz3_dataset_source_card(
            host,
            dataset_source_existing_card,
            selected=bool(vm.existing_selected),
            hovered=bool(dataset_source_card_state.get("existing_hovered", False)),
            badge_text=str(vm.existing_badge_text or ""),
            title_text=str(vm.existing_title_text or ""),
            desc_text=str(vm.existing_desc_text or ""),
        )

    if bool(vm.perfect_selected):
        if str(host.pz3_existing_dataset_panel.winfo_manager()):
            host.pz3_existing_dataset_panel.pack_forget()
        try:
            host.cvat_option2_title_lbl.configure(text=str(vm.cvat_option2_title or ""))
        except Exception:
            pass
        host._set_inline_status_label_state(
            host.cvat_option2_title_lbl,
            tone=str(vm.cvat_option2_tone or "muted"),
            emphasis=str(vm.cvat_option2_tone or "").strip().lower() == "success",
        )
        host._set_inline_status_label_state(host.cvat_option2_desc_lbl, text=str(vm.cvat_option2_desc or ""), tone="muted", emphasis=False)
        try:
            goldpack_lf = getattr(host, "gold_export_goldpack_lf", None) or getattr(host, "gold_export_filters_lf", None)
            if bool(vm.show_gold_filters) and goldpack_lf is not None:
                goldpack_lf.grid()
            elif goldpack_lf is not None and str(goldpack_lf.winfo_manager()):
                goldpack_lf.grid_remove()
        except Exception:
            pass
        try:
            if str(host.gold_export_split_lf.winfo_manager()):
                host.gold_export_split_lf.grid_remove()
        except Exception:
            pass
        try:
            host.btn_yolo_gold_export.configure(
                text=str(vm.primary_export_label or ""),
                command=(
                    host._run_yolo_gold_export
                    if str(vm.primary_export_command_id or "") == "run_yolo_gold_export"
                    else host._run_pz3_existing_dataset_split
                ),
                style="Accent.TButton",
                state=(tk.NORMAL if bool(vm.primary_export_enabled) else tk.DISABLED),
            )
            host.btn_yolo_gold_export.grid_configure(
                column=0,
                columnspan=max(1, int(vm.primary_export_columnspan or 1)),
                padx=((0, 4) if int(vm.primary_export_columnspan or 1) == 1 else (0, 0)),
            )
        except Exception:
            pass
        try:
            if bool(vm.show_classifier_export) and not str(host.btn_char_classifier_export.winfo_manager()):
                host.btn_char_classifier_export.grid(row=0, column=1, sticky="ew", padx=(4, 0), ipady=4)
            elif (not bool(vm.show_classifier_export)) and str(host.btn_char_classifier_export.winfo_manager()):
                host.btn_char_classifier_export.grid_remove()
        except Exception:
            pass
        try:
            host.btn_char_classifier_export.configure(
                state=(tk.NORMAL if bool(vm.classifier_export_enabled) else tk.DISABLED)
            )
        except Exception:
            pass
        host._set_inline_status_label_state(
            host.pz3_dataset_action_hint_lbl,
            text=str(vm.action_hint or ""),
            tone=str(vm.action_hint_tone or "muted"),
            emphasis=False,
        )
        return

    if bool(vm.show_existing_dataset_panel) and not str(host.pz3_existing_dataset_panel.winfo_manager()):
        host.pz3_existing_dataset_panel.pack(fill=tk.X, pady=(10, 0))

    try:
        host.cvat_option2_title_lbl.configure(text=str(vm.cvat_option2_title or ""))
    except Exception:
        pass
    host._set_inline_status_label_state(
        host.cvat_option2_title_lbl,
        tone=str(vm.cvat_option2_tone or "default"),
        emphasis=str(vm.cvat_option2_tone or "").strip().lower() == "success",
    )
    host._set_inline_status_label_state(host.cvat_option2_desc_lbl, text=str(vm.cvat_option2_desc or ""), tone="muted", emphasis=False)
    host._set_inline_status_label_state(
        host.pz3_existing_dataset_status_lbl,
        text=str(vm.existing_dataset_status or ""),
        tone=str(vm.existing_dataset_status_tone or "muted"),
        emphasis=False,
    )
    try:
        goldpack_lf = getattr(host, "gold_export_goldpack_lf", None) or getattr(host, "gold_export_filters_lf", None)
        if not bool(vm.show_gold_filters) and goldpack_lf is not None and str(goldpack_lf.winfo_manager()):
            goldpack_lf.grid_remove()
        elif bool(vm.show_gold_filters) and goldpack_lf is not None:
            goldpack_lf.grid()
    except Exception:
        pass
    try:
        if str(host.gold_export_split_lf.winfo_manager()):
            host.gold_export_split_lf.grid_remove()
    except Exception:
        pass
    try:
        if not bool(vm.show_classifier_export):
            host.btn_char_classifier_export.grid_remove()
    except Exception:
        pass
    try:
        host.btn_yolo_gold_export.configure(
            text=str(vm.primary_export_label or ""),
            command=(
                host._run_yolo_gold_export
                if str(vm.primary_export_command_id or "") == "run_yolo_gold_export"
                else host._run_pz3_existing_dataset_split
            ),
            style="Accent.TButton",
            state=(tk.NORMAL if bool(vm.primary_export_enabled) else tk.DISABLED),
        )
        host.btn_yolo_gold_export.grid_configure(
            column=0,
            columnspan=max(1, int(vm.primary_export_columnspan or 1)),
            padx=((0, 4) if int(vm.primary_export_columnspan or 1) == 1 else (0, 0)),
        )
    except Exception:
        pass
    host._set_inline_status_label_state(
        host.pz3_dataset_action_hint_lbl,
        text=str(vm.action_hint or ""),
        tone=str(vm.action_hint_tone or "muted"),
        emphasis=False,
    )


def refresh_pz3_dataset_card(host: "CharacterAnnotationTab"):
    panel_vm = host._get_step3_pz3_path_selection_view_model()
    card_info = getattr(host, "_pz3_dataset_card", None)
    state = getattr(host, "_pz3_dataset_card_state", {}) or {}
    if card_info is None:
        return
    selected = bool(panel_vm.dataset_card_selected)
    hovered = bool(state.get("hovered", False))
    card_palette = host._get_pz3_card_palette()
    current_card_bg = card_palette["card_active_bg"] if selected else (
        card_palette["card_hover_bg"] if hovered else card_palette["card_bg"]
    )
    border_color = card_palette["card_border"]

    for widget in (
        card_info.get("frame"),
        card_info.get("top_row"),
        card_info.get("badge"),
        card_info.get("state_badge"),
        card_info.get("title"),
        card_info.get("desc"),
    ):
        if widget is None:
            continue
        try:
            widget.configure(bg=current_card_bg, highlightbackground=border_color, highlightcolor=border_color, cursor="hand2")
        except Exception:
            pass
    try:
        card_info["badge"].configure(
            text=host._normalize_pz3_card_badge_text(str(panel_vm.dataset_badge_text or "")),
            fg=(card_palette["card_fg"] if selected else card_palette["card_muted"]),
        )
    except Exception:
        pass
    host._refresh_pz3_card_state_badge(
        card_info,
        selected=bool(selected),
        current_card_bg=current_card_bg,
        card_palette=card_palette,
    )
    try:
        card_info["title"].configure(text=str(panel_vm.dataset_title_text or ""), fg=card_palette["card_fg"])
    except Exception:
        pass
    host._set_inline_status_label_state(
        card_info["desc"],
        text=str(panel_vm.dataset_desc_text or ""),
        tone=("default" if selected else "muted"),
        emphasis=False,
    )
    try:
        card_info["desc"].configure(fg=(card_palette["card_fg"] if selected else card_palette["card_muted"]))
    except Exception:
        pass


def refresh_pz3_cvat_card(host: "CharacterAnnotationTab"):
    panel_vm = host._get_step3_pz3_path_selection_view_model()
    card_info = getattr(host, "_pz3_review_card", None)
    state = getattr(host, "_pz3_cvat_card_state", {}) or {}
    if card_info is None:
        return
    expanded = bool(panel_vm.cvat_card_selected)
    hovered = bool(state.get("hovered", False))
    palette_getter = getattr(host, "_get_pz3_action_card_palette", host._get_pz3_card_palette)
    card_palette = palette_getter()
    current_card_bg = card_palette["card_active_bg"] if expanded else (
        card_palette["card_hover_bg"] if hovered else card_palette["card_bg"]
    )
    border_color = card_palette["card_border"]
    desc_text = str(panel_vm.cvat_desc_text or "")
    badge_text = str(panel_vm.cvat_badge_text or "")
    title_text = str(panel_vm.cvat_title_text or "")

    for widget in (
        card_info.get("frame"),
        card_info.get("top_row"),
        card_info.get("badge"),
        card_info.get("state_badge"),
        card_info.get("title"),
        card_info.get("desc"),
    ):
        if widget is None:
            continue
        try:
            widget.configure(bg=current_card_bg, highlightbackground=border_color, highlightcolor=border_color, cursor="hand2")
        except Exception:
            pass
    try:
        card_info["badge"].configure(
            text=host._normalize_pz3_card_badge_text(badge_text),
            fg=(card_palette["card_fg"] if expanded else card_palette.get("card_accent", card_palette["card_muted"])),
        )
    except Exception:
        pass
    host._refresh_pz3_card_state_badge(
        card_info,
        selected=bool(expanded),
        current_card_bg=current_card_bg,
        card_palette=card_palette,
    )
    try:
        card_info["title"].configure(text=title_text, fg=card_palette["card_fg"])
    except Exception:
        pass
    host._set_inline_status_label_state(card_info["desc"], text=desc_text, tone=("default" if expanded else "muted"), emphasis=False)
    try:
        card_info["desc"].configure(fg=(card_palette["card_fg"] if expanded else card_palette["card_muted"]))
    except Exception:
        pass


def refresh_pz3_cards_ui(host: "CharacterAnnotationTab"):
    panel_vm = host._get_step3_pz3_path_selection_view_model()
    dataset_section = getattr(host, "pz3_dataset_section", None)
    overview_section = getattr(host, "_pz3_overview_section", None)
    review_card_frame = None
    try:
        review_card_frame = (getattr(host, "_pz3_review_card", {}) or {}).get("frame")
    except Exception:
        review_card_frame = None
    host._pz3_cvat_expanded = str(panel_vm.selected_path or "") == "cvat"
    try:
        if dataset_section is not None:
            if not str(dataset_section.winfo_manager()):
                dataset_section.pack(fill=tk.X, pady=(12, 0), after=overview_section)
        if bool(panel_vm.show_cvat_section):
            if not str(host.pz3_cvat_section.winfo_manager()):
                if review_card_frame is not None:
                    host.pz3_cvat_section.pack(fill=tk.X, pady=(12, 0), after=review_card_frame)
                else:
                    host.pz3_cvat_section.pack(fill=tk.X, pady=(12, 0))
        else:
            if str(host.pz3_cvat_section.winfo_manager()):
                host.pz3_cvat_section.pack_forget()
        import_host = getattr(host, "gold_export_sources_import_host", None)
        if import_host is not None:
            if not str(import_host.winfo_manager()):
                import_host.pack(fill=tk.X, pady=(12, 0))
            if str(host.pz3_cvat_section.winfo_manager()):
                import_host.pack_configure(after=host.pz3_cvat_section)
            elif review_card_frame is not None:
                import_host.pack_configure(after=review_card_frame)
    except Exception:
        pass
    try:
        status_section_widget = getattr(host, "pz3_status_section", None)
        if status_section_widget is not None:
            if bool(getattr(host, "_pz3_status_section_persistent", False)):
                if not str(status_section_widget.winfo_manager()):
                    status_section_widget.pack(fill=tk.BOTH, expand=True)
            elif str(status_section_widget.winfo_manager()):
                status_section_widget.pack_forget()
    except Exception:
        pass
    refresh_pz3_dataset_card(host)
    refresh_pz3_cvat_card(host)
    try:
        refresh_pz3_status_panel_ui(host)
    except Exception:
        pass


def refresh_step3_mode_specific_ui(host: "CharacterAnnotationTab"):
    if (getattr(host, "_detect_tab_built", None) is False
            and getattr(host, "_dataset_tab_built", None) is False):
        return
    in_campaign = is_step3_campaign_runtime(host)

    host._set_grid_visibility(getattr(host, "preview_source_lf", None), (not in_campaign))
    host._set_grid_visibility(getattr(host, "preview_counts_frame", None), True)
    host._set_grid_visibility(getattr(host, "preview_layout_summary_lbl", None), True)
    host._set_grid_visibility(getattr(host, "preview_fusion_info_lbl", None), False)
    host._set_grid_visibility(getattr(host, "preview_box_mode_info_lbl", None), False)
    for attr_name in (
        "preview_repair_progress_title_lbl",
        "preview_repair_progress",
    ):
        host._set_grid_visibility(getattr(host, attr_name, None), True)
    host._set_grid_visibility(getattr(host, "preview_repair_progress_status_lbl", None), False)

    try:
        title = getattr(host, "preview_status_title_lbl", None)
        if title is not None:
            title.configure(text=("Status pracy" if in_campaign else "Podsumowanie listy"))
    except Exception:
        pass

    try:
        note = getattr(host, "preview_load_note_lbl", None)
        if note is not None:
            note_text = (
                "Status pracy, liczniki i jakość zbioru są widoczne tutaj. Szuflada na canvasie służy jako szybki skrót w trakcie edycji."
                if in_campaign
                else (
                    "Poniżej zostaje krótkie podsumowanie listy. Do rysowania i korekty boxów najlepiej użyj pełnego canvasu."
                )
            )
            host._set_inline_status_label_state(
                note,
                text=note_text,
                tone="muted",
                emphasis=False,
            )
    except Exception:
        pass

    try:
        intro = getattr(host, "preview_list_intro_lbl", None)
        if intro is not None:
            host._set_inline_status_label_state(
                intro,
                text=get_step3_pz2_intro_text(host),
                tone="muted",
                emphasis=False,
            )
    except Exception:
        pass

    host._set_grid_visibility(getattr(host, "test_progress_row", None), False)

    try:
        refresh = getattr(host, "_refresh_pz3_dataset_mode_ui", None)
        if callable(refresh):
            refresh()
    except Exception:
        pass

    try:
        refresh = getattr(host, "_refresh_pz3_cards_ui", None)
        if callable(refresh):
            refresh()
    except Exception:
        pass

    for refresh_name in (
        "_refresh_gold_export_filter_labels",
        "_refresh_gold_export_source_labels",
        "_refresh_gold_export_scope_label",
    ):
        try:
            refresh_fn = getattr(host, refresh_name, None)
            if callable(refresh_fn):
                refresh_fn()
        except Exception:
            pass

    try:
        host._update_preview_repair_progress_ui()
        if not in_campaign:
            host._set_test_progress_counter()
    except Exception:
        pass


def sync_step3_nav_buttons(host: "CharacterAnnotationTab"):
    detect_enabled = host._get_subtab_state(host.tab_detect) == "normal"
    dataset_enabled = host._get_subtab_state(host.tab_dataset) == "normal"

    if hasattr(host, "btn_to_detect"):
        host.btn_to_detect.config(state=tk.NORMAL if detect_enabled else tk.DISABLED)

    if hasattr(host, "btn_to_dataset"):
        host.btn_to_dataset.config(state=tk.NORMAL if dataset_enabled else tk.DISABLED)
