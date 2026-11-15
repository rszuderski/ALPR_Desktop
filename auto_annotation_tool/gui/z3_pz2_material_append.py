#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AZ009D — wejście UI appendu materiału do aktywnego PZ2."""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..campaign_manager import CAMPAIGN
from ..config import CONFIG
from ..registry.az_package_transport import (
    analyze_az_package,
    append_az_package,
    load_az_package,
)
from ..registry.az_registry import (
    AZRegistry,
    ensure_campaign_project,
)
from ..registry.pz2_append_materializer import (
    materialize_az_append_to_preview,
)



def open_pz2_add_material(host) -> bool:
    """Wybierz sposób ROZSZERZENIA aktywnego zbioru projektu."""
    context = _resolve_active_project_context(host)
    if context is None:
        return False

    mode = _choose_add_material_kind(host)
    if mode == "az":
        return _open_pz2_add_az_package(host, context=context)
    if mode == "images_at":
        return _open_pz2_add_images_at(host, context=context)
    return False


def _choose_add_material_kind(host) -> str | None:
    parent = _parent(host)
    dialog = tk.Toplevel(parent)
    dialog.title("Dodaj materiał do zbioru")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    result = {"mode": None}

    shell = ttk.Frame(dialog, padding=14)
    shell.grid(row=0, column=0, sticky="nsew")
    shell.grid_columnconfigure(0, weight=1)

    ttk.Label(
        shell,
        text="Jak chcesz rozszerzyć bieżący zbiór?",
        font=("Segoe UI", 10, "bold"),
    ).grid(row=0, column=0, sticky="w", pady=(0, 8))

    ttk.Label(
        shell,
        text=(
            "Obie opcje dodają materiał do aktualnego projektu. "
            "Nie zastępują bieżącego zbioru PZ2."
        ),
        justify="left",
        wraplength=470,
    ).grid(row=1, column=0, sticky="ew", pady=(0, 12))

    def finish(value):
        result["mode"] = value
        try:
            dialog.grab_release()
        except Exception:
            pass
        dialog.destroy()

    ttk.Button(
        shell,
        text="Gotowe tablice + AZ",
        command=lambda: finish("az"),
    ).grid(row=2, column=0, sticky="ew", pady=(0, 5))

    ttk.Label(
        shell,
        text=(
            "Dopnij gotowe cropy tablic bezpośrednio do bieżącego PZ2. "
            "AZ zostanie oznaczone do lokalnej kontroli."
        ),
        justify="left",
        wraplength=470,
    ).grid(row=3, column=0, sticky="ew", pady=(0, 12))

    ttk.Button(
        shell,
        text="Obrazy + AT",
        command=lambda: finish("images_at"),
    ).grid(row=4, column=0, sticky="ew", pady=(0, 5))

    ttk.Label(
        shell,
        text=(
            "Dodaj nowe obrazy z annotations.xml. Najpierw sprawdzisz je w Z2 "
            "i jawnie nadasz [OK]; dopiero potem trafią przez PZ1 do PZ2."
        ),
        justify="left",
        wraplength=470,
    ).grid(row=5, column=0, sticky="ew", pady=(0, 12))

    ttk.Button(
        shell,
        text="Anuluj",
        command=lambda: finish(None),
    ).grid(row=6, column=0, sticky="e")

    dialog.protocol("WM_DELETE_WINDOW", lambda: finish(None))
    dialog.bind("<Escape>", lambda _event: finish(None), add="+")
    dialog.wait_window()
    return result["mode"]


def _open_pz2_add_images_at(host, *, context: dict) -> bool:
    """Append-safe routing nowych obrazów + AT do istniejącego Z2."""
    parent = _parent(host)
    app = getattr(host, "app", None)

    try:
        _flush_preview_metadata(host)
    except Exception as exc:
        _show_error(
            host,
            "Dodawanie obrazów + AT",
            f"Nie udało się zapisać odłożonych zmian PZ2.\n\n{exc}",
        )
        return False

    selected_xml = filedialog.askopenfilename(
        parent=parent,
        initialdir=str(context.get("project_root") or context["preview_dir"]),
        title="Wskaż annotations.xml z nowymi tablicami",
        filetypes=[
            ("Plik annotations.xml", "annotations.xml"),
            ("Pliki XML", "*.xml"),
            ("Wszystkie pliki", "*.*"),
        ],
    )
    if not selected_xml:
        return False

    xml_path = Path(selected_xml)
    if not xml_path.is_file() or xml_path.name.lower() != "annotations.xml":
        _show_error(
            host,
            "Dodawanie obrazów + AT",
            "Wskaż właściwy plik annotations.xml.",
        )
        return False

    images_dir = xml_path.parent / "images"
    if not _dir_has_images(images_dir):
        selected_images = filedialog.askdirectory(
            parent=parent,
            initialdir=str(xml_path.parent),
            title="Wskaż folder obrazów zgodnych z annotations.xml",
        )
        if not selected_images:
            return False
        images_dir = Path(selected_images)

    if not _dir_has_images(images_dir):
        _show_error(
            host,
            "Dodawanie obrazów + AT",
            "W wybranym katalogu nie znaleziono obsługiwanych obrazów.",
        )
        return False

    campaign_tab = None
    try:
        campaign_tab = getattr(app, "tabs", {}).get("campaign")
    except Exception:
        campaign_tab = None
    if campaign_tab is None:
        try:
            loader = getattr(app, "_ensure_tab_loaded", None)
            if callable(loader):
                campaign_tab = loader("campaign", select=False)
        except Exception:
            campaign_tab = None

    importer = getattr(campaign_tab, "_import_project_start_plate_run", None)
    if not callable(importer):
        _show_error(
            host,
            "Dodawanie obrazów + AT",
            "Nie udało się przygotować istniejącego importera AT projektu.",
        )
        return False

    previous_master_pool = None
    try:
        previous_master_pool = CAMPAIGN.get_master_pool_dir()
    except Exception:
        previous_master_pool = None

    import_ok = False
    try:
        # Istniejący importer E1 dopasowuje AT do bieżącego źródła O.
        # Wskazujemy nowe obrazy tylko na czas przygotowania draftu.
        if not CAMPAIGN.set_master_pool_dir(images_dir):
            raise RuntimeError(
                "Nie udało się tymczasowo wskazać nowych obrazów jako źródła AT."
            )

        import_ok = bool(
            importer(
                selected_xml_path=xml_path,
                parent=parent,
                refresh_dashboard_after_import=False,
                confirm_import=True,
            )
        )
    except Exception as exc:
        _show_error(
            host,
            "Dodawanie obrazów + AT",
            f"Nie udało się przygotować draftu AT w Z2.\n\n{exc}",
        )
        import_ok = False
    finally:
        _restore_master_pool(previous_master_pool)

    if not import_ok:
        return False

    try:
        plate_source = dict(CAMPAIGN.get_project_start_plate_source() or {})
    except Exception:
        plate_source = {}

    source_run = str(plate_source.get("source_run_path") or "").strip()
    source_xml = str(plate_source.get("source_xml_path") or "").strip()
    source_input = str(plate_source.get("source_input_path") or "").strip()

    if not source_run and source_xml:
        try:
            source_run = str(Path(source_xml).parent)
        except Exception:
            source_run = ""

    if not source_run:
        _show_error(
            host,
            "Dodawanie obrazów + AT",
            (
                "AT zostało przygotowane, ale nie udało się ustalić katalogu "
                "draftu Z2 do lokalnej kontroli."
            ),
        )
        return False

    annotation_tab = None
    try:
        annotation_tab = getattr(app, "tabs", {}).get("annotation")
    except Exception:
        annotation_tab = None
    if annotation_tab is None:
        try:
            loader = getattr(app, "_ensure_tab_loaded", None)
            if callable(loader):
                annotation_tab = loader("annotation", select=False)
        except Exception:
            annotation_tab = None

    open_review = getattr(
        annotation_tab,
        "open_existing_run_for_campaign_review",
        None,
    )
    if not callable(open_review):
        _show_error(
            host,
            "Dodawanie obrazów + AT",
            "Draft AT jest gotowy, ale nie udało się otworzyć kontroli Z2.",
        )
        return False

    review_context = {
        "source": "pz2_append_images_at",
        "graph_edge_key": "e3_to_e4",
        "graph_gate_id": "T05",
        "graph_visible_gate_id": "T05",
        "graph_display_gate_id": "T05",
        "graph_transition_source": "E3",
        "graph_transition_target": "E4Z",
        "graph_gate_label": "Dataset znaków",
        "z2_work_mode": "pz2_append_images_at_review",
        "restore_run_dir": source_run,
        "xml_path": source_xml,
        "input_dir": source_input or str(images_dir),
        "input_source": "pz2_append_images_at",
    }
    try:
        annotation_tab._campaign_graph_entry_context = dict(review_context)
        annotation_tab._campaign_context_project_name = str(
            CAMPAIGN.get_active_project_name() or ""
        ).strip()
    except Exception:
        pass

    try:
        opened = bool(
            open_review(
                Path(source_run),
                iteration_target="char",
                manual_template=False,
                defer_ui_restore=False,
            )
        )
    except Exception as exc:
        _show_error(
            host,
            "Dodawanie obrazów + AT",
            f"Draft AT jest gotowy, ale nie udało się otworzyć go w Z2.\n\n{exc}",
        )
        return False

    if not opened:
        _show_error(
            host,
            "Dodawanie obrazów + AT",
            "Draft AT jest gotowy, ale Z2 odmówiło jego otwarcia.",
        )
        return False

    try:
        app.open_controlled_tab("annotation")
    except Exception as exc:
        _show_error(
            host,
            "Dodawanie obrazów + AT",
            f"Draft AT jest gotowy, ale nie udało się przełączyć widoku na Z2.\n\n{exc}",
        )
        return False

    try:
        update_status = getattr(app, "update_status", None)
        if callable(update_status):
            update_status(
                "Dodano nowe obrazy + AT do kontroli w Z2. "
                "Sprawdź nowe tablice i nadaj [OK] właściwym obrazom.",
                "info",
            )
    except Exception:
        pass
    return True


def _restore_master_pool(previous_master_pool) -> None:
    try:
        if previous_master_pool is not None:
            CAMPAIGN.set_master_pool_dir(previous_master_pool)
        else:
            clear = getattr(CAMPAIGN, "clear_master_pool_dir", None)
            if callable(clear):
                clear()
    except Exception:
        pass


def _dir_has_images(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_dir():
            return False
        extensions = {
            str(ext).lower()
            for ext in getattr(CONFIG, "IMAGE_EXTENSIONS", ())
        }
        return any(
            item.is_file()
            and item.suffix.lower() in extensions
            for item in path.iterdir()
        )
    except Exception:
        return False


def _open_pz2_add_az_package(
    host,
    *,
    context: dict | None = None,
) -> bool:
    context = context or _resolve_active_project_context(host)
    if context is None:
        return False

    parent = _parent(host)
    preview_dir = context["preview_dir"]

    selected = filedialog.askdirectory(
        parent=parent,
        initialdir=str(preview_dir),
        title="Wybierz pakiet gotowych tablic + AZ",
    )
    if not selected:
        return False

    try:
        package = load_az_package(selected)
    except Exception as exc:
        _show_error(
            host,
            "Dodawanie materiału do zbioru",
            f"Wybrany katalog nie jest poprawnym pakietem cropów + AZ.\n\n{exc}",
        )
        return False

    registry = context["registry"]
    project_id = context["project_id"]
    iteration_num = context["iteration_num"]
    metadata_path = preview_dir / "metadata.json"
    images_dir = preview_dir / "images"

    try:
        _flush_preview_metadata(host)
        plan = analyze_az_package(
            registry,
            package,
            target_project_id=project_id,
        )
    except Exception as exc:
        _show_error(
            host,
            "Dodawanie materiału do zbioru",
            f"Nie udało się wykonać preflightu pakietu.\n\n{exc}",
        )
        return False

    if plan.new_count <= 0:
        if plan.already_present_count > 0:
            question = (
                "Pakiet nie zawiera nowych logical cropów do przypięcia, ale część materiału "
                "jest już zarejestrowana w tym projekcie.\n\n"
                f"Już przypięte: {plan.already_present_count}\n"
                f"Konflikty: {plan.conflict_count}\n"
                f"Nieprawidłowe: {plan.invalid_count}\n\n"
                "Sprawdzić, czy poprzedni append wymaga dokończenia w aktywnym PZ2?"
            )
            if not messagebox.askyesno(
                "Dokończ append materiału",
                question,
                parent=parent,
            ):
                return False
            try:
                result = materialize_az_append_to_preview(
                    registry,
                    package,
                    preview_dir=preview_dir,
                    target_project_id=project_id,
                    iteration_num=iteration_num,
                    append_result=None,
                )
                _refresh_after_append(host, metadata_path, result, package.package_id)
            except Exception as exc:
                _show_error(
                    host,
                    "Dokończenie appendu",
                    f"Nie udało się dokończyć appendu w PZ2.\n\n{exc}",
                )
                return False

            if result.added <= 0:
                _show_info(
                    host,
                    "Materiał już w zbiorze",
                    (
                        "Pakiet jest już w pełni obecny w aktywnym zbiorze PZ2.\n\n"
                        f"Obecnych rekordów: {result.after_count}\n"
                        f"Rozpoznanych rekordów pakietu: {result.candidates}"
                    ),
                )
            else:
                _show_success(host, result, plan)
            return True

        _show_info(
            host,
            "Brak nowego materiału",
            (
                "Pakiet nie zawiera cropów, które można dopiąć do bieżącego projektu.\n\n"
                f"Konflikty: {plan.conflict_count}\n"
                f"Nieprawidłowe: {plan.invalid_count}"
            ),
        )
        return False

    current_count = _current_metadata_count(host, metadata_path)
    projected_count = current_count + int(plan.new_count)
    confirmation = (
        "Ta operacja ROZSZERZY istniejący zbiór aktywnego projektu. "
        "Nie zastąpi bieżącego runu PZ2 i nie nadpisze lokalnej pracy.\n\n"
        f"Obecny zbiór: {current_count}\n"
        f"Nowe cropy do dopięcia: {plan.new_count}\n"
        f"Już obecne / bez duplikacji: {plan.already_present_count}\n"
        f"Konflikty pomijane: {plan.conflict_count}\n"
        f"Nieprawidłowe pomijane: {plan.invalid_count}\n"
        f"Z AZ w poprawnych rekordach: {plan.with_az_count}\n"
        f"Bez AZ w poprawnych rekordach: {plan.without_az_count}\n"
        f"Po operacji: {projected_count}\n\n"
        "Nowe AZ zostaną oznaczone jako wymagające lokalnej kontroli w tym projekcie."
    )
    if not messagebox.askyesno(
        "Dodaj materiał do zbioru",
        confirmation,
        parent=parent,
    ):
        return False

    try:
        append_result = append_az_package(
            registry,
            package,
            target_project_id=project_id,
            iteration_num=iteration_num,
            target_artifact_dir=images_dir,
        )
        result = materialize_az_append_to_preview(
            registry,
            package,
            preview_dir=preview_dir,
            target_project_id=project_id,
            iteration_num=iteration_num,
            append_result=append_result,
        )
        _refresh_after_append(host, metadata_path, result, package.package_id)
    except Exception as exc:
        _show_error(
            host,
            "Dodawanie materiału do zbioru",
            (
                "Append nie został poprawnie domknięty.\n\n"
                f"{exc}\n\n"
                "Istniejąca praca PZ2 nie została zastąpiona."
            ),
        )
        return False

    _show_success(host, result, plan)
    return True


def _resolve_active_project_context(host) -> dict | None:
    parent = _parent(host)
    try:
        project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
        project_root = CAMPAIGN.get_active_project_root_dir()
        iteration_num = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        project_name = ""
        project_root = None
        iteration_num = 0

    if not project_name or project_root is None or iteration_num <= 0:
        messagebox.showwarning(
            "Dodaj materiał do zbioru",
            "Ta operacja wymaga aktywnego projektu kampanii.",
            parent=parent,
        )
        return None

    if not bool(getattr(host, "_step3_linear_mode", False)):
        messagebox.showwarning(
            "Dodaj materiał do zbioru",
            "Dodawanie materiału do projektu jest dostępne w projektowym PZ2.",
            parent=parent,
        )
        return None

    try:
        preview_raw = str(host.preview_dir_var.get() or "").strip()
    except Exception:
        preview_raw = ""
    if not preview_raw:
        messagebox.showwarning(
            "Dodaj materiał do zbioru",
            "PZ2 nie ma aktywnego runu. Najpierw przygotuj tablice w PZ1.",
            parent=parent,
        )
        return None

    preview_dir = Path(preview_raw)
    metadata_path = preview_dir / "metadata.json"
    images_dir = preview_dir / "images"
    if not metadata_path.is_file() or not images_dir.is_dir():
        messagebox.showwarning(
            "Dodaj materiał do zbioru",
            "Aktywny PZ2 nie ma kompletnego metadata.json + images/.",
            parent=parent,
        )
        return None

    loaded_meta_path = getattr(host, "_loaded_meta_path", None)
    if loaded_meta_path is not None:
        try:
            if Path(loaded_meta_path).resolve() != metadata_path.resolve():
                messagebox.showwarning(
                    "Dodaj materiał do zbioru",
                    "Widok PZ2 nie jest zsynchronizowany z aktywnym metadata.json. "
                    "Odśwież PZ2 przed dodaniem materiału.",
                    parent=parent,
                )
                return None
        except Exception:
            pass

    workspace = Path(CONFIG.WORKSPACE_DIR)
    registry = AZRegistry.for_workspace(workspace)
    project_root = Path(project_root)
    try:
        project_id = ensure_campaign_project(
            registry,
            project_name=project_name,
            folder_name=project_root.name,
        )
    except Exception as exc:
        _show_error(
            host,
            "Dodaj materiał do zbioru",
            f"Nie udało się ustalić project_id w AZ registry.\n\n{exc}",
        )
        return None

    return {
        "project_name": project_name,
        "project_root": project_root,
        "project_id": project_id,
        "iteration_num": iteration_num,
        "preview_dir": preview_dir,
        "registry": registry,
    }


def _flush_preview_metadata(host) -> None:
    flush = getattr(host, "_flush_scheduled_preview_metadata_save", None)
    if callable(flush):
        flush()


def _current_metadata_count(host, metadata_path: Path) -> int:
    live = getattr(host, "preview_metadata", None)
    if isinstance(live, dict):
        return len(live)
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        return len(payload) if isinstance(payload, dict) else 0
    except Exception:
        return 0


def _refresh_after_append(host, metadata_path: Path, result, package_id: str) -> None:
    payload = getattr(result, "verified_metadata", None)
    stat = metadata_path.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    result_path = getattr(result, "metadata_path", None)
    if (not isinstance(payload, dict)
            or getattr(result, "metadata_signature", None) != signature
            or result_path is None
            or Path(result_path).resolve() != metadata_path.resolve()):
        payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError("Po appendzie metadata.json nie jest obiektem JSON.")

    # Ustaw fokus importu PRZED przebudową listy.
    # _apply_preview_metadata_update() wywołuje _rebuild_preview_listbox(),
    # który już respektuje _preview_import_focus_active. Dzięki temu po
    # appendzie budujemy od razu krótką listę nowych rekordów zamiast:
    #   1) pełnej listy całego PZ2,
    #   2) drugiej listy zawężonej do importu.
    new_plate_ids = []
    seen_plate_ids = set()
    for item in getattr(result, "items", ()):
        plate_id = str(getattr(item, "plate_id", "") or "").strip()
        if not plate_id or plate_id in seen_plate_ids:
            continue
        seen_plate_ids.add(plate_id)
        new_plate_ids.append(plate_id)

    if new_plate_ids:
        host._preview_import_focus_plate_ids = list(new_plate_ids)
        host._preview_import_focus_batch_id = str(package_id or "").strip()
        host._preview_import_focus_active = True

    host._apply_preview_metadata_update(
        payload,
        preserve_selection=False,
        render_selection=False,
        recalculate_statuses=False,
    )
    host._loaded_meta_path = metadata_path
    try:
        host._loaded_meta_mtime = metadata_path.stat().st_mtime
    except Exception:
        pass

    for refresh_name in (
        "_set_plates_legend_info",
        "_update_preview_repair_progress_ui",
        "_refresh_preview_source_panel",
    ):
        try:
            fn = getattr(host, refresh_name, None)
            if callable(fn):
                fn()
        except Exception:
            pass

    try:
        sync = getattr(host, "_sync_step3_access_from_preview_state", None)
        if callable(sync):
            sync(getattr(host, "preview_metadata", None))
    except Exception:
        pass


def _show_success(host, result, plan) -> None:
    _show_info(
        host,
        "Materiał dodany do zbioru",
        (
            "Rozszerzono TEN SAM aktywny zbiór PZ2.\n\n"
            f"Przed: {result.before_count}\n"
            f"Dodano do PZ2: {result.added}\n"
            f"Już obecne / bez duplikacji: {result.already_materialized}\n"
            f"AZ odzyskano: {result.with_az}\n"
            f"Bez AZ: {result.without_az}\n"
            f"Konflikty pakietu pominięte: {plan.conflict_count}\n"
            f"Nieprawidłowe pominięte: {plan.invalid_count}\n"
            f"Po: {result.after_count}\n\n"
            "Nowy materiał został ustawiony do lokalnej kontroli w PZ2."
        ),
    )


def _show_info(host, title: str, message: str) -> None:
    app = getattr(host, "app", None)
    themed = getattr(app, "themed_info", None)
    if callable(themed):
        try:
            themed(title, message, parent=_parent(host), tone="info")
            return
        except Exception:
            pass
    messagebox.showinfo(title, message, parent=_parent(host))


def _show_error(host, title: str, message: str) -> None:
    app = getattr(host, "app", None)
    themed = getattr(app, "themed_error", None)
    if callable(themed):
        try:
            themed(title, message, parent=_parent(host))
            return
        except Exception:
            pass
    messagebox.showerror(title, message, parent=_parent(host))


def _parent(host):
    frame = getattr(host, "frame", None)
    try:
        return frame.winfo_toplevel() if frame is not None else None
    except Exception:
        return frame
