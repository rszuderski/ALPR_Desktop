#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AZ009E2 — UI eksportu przenośnego pakietu cropów + AZ z aktywnego PZ2."""

from __future__ import annotations

from datetime import datetime
import re
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

from ..campaign_manager import CAMPAIGN
from ..config import CONFIG
from ..registry.az_registry import AZRegistry, project_id_from_folder_name
from ..registry.az_package_export import export_pz2_az_package
from .z3_pz2_material_append import (
    _flush_preview_metadata,
    _parent,
    _show_error,
    _show_info,
)


def open_pz2_export_material(host) -> bool:
    """Eksportuj materiał z TEGO SAMEGO aktywnego PZ2 do alpr.az_package.v1."""
    context = _resolve_export_context(host)
    if context is None:
        return False

    try:
        if context["project_id"] is not None:
            _flush_preview_metadata(host)
    except Exception as exc:
        _show_error(
            host,
            "Eksport cropów + AZ",
            f"Nie udało się zapisać odłożonych zmian PZ2.\n\n{exc}",
        )
        return False

    scope_info = _collect_export_scopes(host)
    scope = _prompt_export_scope(host, scope_info)
    if not scope:
        return False

    plate_ids = scope_info[scope]["plate_ids"]
    if plate_ids is not None and not plate_ids:
        _show_info(
            host,
            "Eksport cropów + AZ",
            "Wybrany zakres nie zawiera żadnych tablic do eksportu.",
        )
        return False

    parent_dir = filedialog.askdirectory(
        parent=_parent(host),
        initialdir=str(context["output_initial_dir"]),
        title="Wybierz katalog, w którym utworzyć pakiet cropów + AZ",
    )
    if not parent_dir:
        return False

    output_dir = _build_unique_output_dir(
        Path(parent_dir),
        project_name=context["project_name"],
        iteration_num=context["iteration_num"],
        scope=scope,
    )

    try:
        result = export_pz2_az_package(
            context["registry"],
            preview_dir=context["preview_dir"],
            project_id=context["project_id"],
            iteration_num=context["iteration_num"],
            output_dir=output_dir,
            plate_ids=plate_ids,
            metadata=(getattr(host, "preview_metadata", None)
                      if context["project_id"] is None else None),
        )
    except Exception as exc:
        _show_error(
            host,
            "Eksport cropów + AZ",
            (
                "Nie udało się utworzyć przenośnego pakietu.\n\n"
                f"{exc}\n\n"
                "Aktywny PZ2 nie został zmodyfikowany."
            ),
        )
        return False

    _show_info(
        host,
        "Wyeksportowano cropy + AZ",
        (
            "Utworzono przenośny pakiet zgodny z funkcją "
            "„Dodaj tablice lub zdjęcia…” → „Gotowe tablice + AZ”.\n\n"
            f"Zakres: {scope_info[scope]['label']}\n"
            f"Wyeksportowano: {result.exported}\n"
            f"Z istniejącą rewizją AZ: {result.with_bound_revision}\n"
            f"Pakiet: {result.package_id}\n\n"
            f"Katalog:\n{result.output_dir}"
        ),
    )
    try:
        app = getattr(host, "app", None)
        update = getattr(app, "update_status", None)
        if callable(update):
            update(
                f"Wyeksportowano pakiet cropów + AZ: {result.exported} tablic.",
                "success",
            )
    except Exception:
        pass
    return True


def _resolve_export_context(host) -> dict | None:
    """Resolve an existing preview and registry without creating project state."""
    try:
        preview_raw = str(host.preview_dir_var.get() or "").strip()
        if not preview_raw:
            raise ValueError("Najpierw przygotuj tablice w PZ1 i otwórz je w PZ2.")
        preview = Path(preview_raw)
        metadata_path = preview / "metadata.json"
        if not metadata_path.is_file() or not (preview / "images").is_dir():
            raise ValueError("Aktywny PZ2 nie ma kompletnego metadata.json + images/.")
        loaded = getattr(host, "_loaded_meta_path", None)
        if loaded is not None and Path(loaded).resolve() != metadata_path.resolve():
            raise ValueError("Odśwież PZ2: widok nie odpowiada aktywnemu metadata.json.")

        registry = AZRegistry.for_workspace(Path(CONFIG.WORKSPACE_DIR))
        if not registry.database.db_path.is_file():
            raise ValueError("Brak istniejącego AZ registry dla aktywnego Workspace.")
        project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
        free_mode = bool(getattr(getattr(host, "app", None), "campaign_free_mode", False))
        project_id = None
        iteration_num = None
        initial_dir = preview.parent
        display_name = preview.name
        if project_name and not free_mode:
            project_root = CAMPAIGN.get_active_project_root_dir()
            iteration_num = int(CAMPAIGN.get_current_iteration_num() or 0)
            if project_root is None or iteration_num <= 0 or not getattr(host, "_step3_linear_mode", False):
                raise ValueError("Eksport projektu wymaga aktywnego projektowego PZ2 i iteracji.")
            initial_dir = Path(project_root)
            project_id = project_id_from_folder_name(initial_dir.name)
            display_name = project_name
        return {
            "registry": registry,
            "preview_dir": preview,
            "project_id": project_id,
            "iteration_num": iteration_num,
            "project_name": display_name,
            "output_initial_dir": initial_dir,
        }
    except Exception as exc:
        _show_error(host, "Eksport cropów + AZ", str(exc))
        return None


def _collect_export_scopes(host) -> dict[str, dict]:
    metadata = getattr(host, "preview_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}

    all_ids = [
        str(pid)
        for pid, row in metadata.items()
        if isinstance(row, dict)
        and str(row.get("crop_id") or "").strip()
    ]

    approved_ids = []
    for pid in all_ids:
        row = metadata.get(pid) or {}
        gold = row.get("gold_state")
        gold = gold if isinstance(gold, dict) else {}
        status = str(row.get("status") or "").strip().lower()
        if bool(gold.get("approved", False)) or status == "perfect":
            approved_ids.append(pid)

    selected_ids = _selected_plate_ids(host, metadata)

    recent_ids = []
    seen = set()
    for pid in list(
        getattr(host, "_preview_import_focus_plate_ids", []) or []
    ):
        value = str(pid or "").strip()
        if value and value in metadata and value not in seen:
            seen.add(value)
            row = metadata.get(value)
            if isinstance(row, dict) and str(
                row.get("crop_id") or ""
            ).strip():
                recent_ids.append(value)

    return {
        "all": {
            "label": f"Wszystkie tablice ({len(all_ids)})",
            "plate_ids": None,
            "enabled": bool(all_ids),
        },
        "approved": {
            "label": f"Tylko zatwierdzone [OK] ({len(approved_ids)})",
            "plate_ids": approved_ids,
            "enabled": bool(approved_ids),
        },
        "selected": {
            "label": f"Zaznaczone tablice ({len(selected_ids)})",
            "plate_ids": selected_ids,
            "enabled": bool(selected_ids),
        },
        "recent": {
            "label": f"Ostatnio dodany materiał ({len(recent_ids)})",
            "plate_ids": recent_ids,
            "enabled": bool(recent_ids),
        },
    }


def _selected_plate_ids(host, metadata: dict) -> list[str]:
    result = []
    seen = set()
    listbox = getattr(host, "plates_listbox", None)
    mapping = list(getattr(host, "_listbox_pid_by_index", []) or [])
    if listbox is not None:
        try:
            indices = list(listbox.curselection())
        except Exception:
            indices = []
        for raw_index in indices:
            try:
                index = int(raw_index)
            except Exception:
                continue
            if 0 <= index < len(mapping):
                pid = str(mapping[index] or "").strip()
                if pid and pid in metadata and pid not in seen:
                    seen.add(pid)
                    result.append(pid)

    if result:
        return result

    active = str(
        getattr(host, "_preview_active_pid", "") or ""
    ).strip()
    if active and active in metadata:
        return [active]
    return []


def _prompt_export_scope(host, scopes: dict[str, dict]) -> str | None:
    parent = _parent(host)
    dialog = tk.Toplevel(parent)
    dialog.title("Eksportuj cropy + AZ")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    result = {"scope": None}

    shell = ttk.Frame(dialog, padding=14)
    shell.grid(row=0, column=0, sticky="nsew")
    shell.grid_columnconfigure(0, weight=1)

    ttk.Label(
        shell,
        text="Wybierz zakres materiału do przenośnego pakietu:",
        font=("Segoe UI", 10, "bold"),
    ).grid(row=0, column=0, sticky="w", pady=(0, 10))

    ttk.Label(
        shell,
        text=(
            "Eksport nie zmienia aktywnego PZ2. Pakiet można później "
            "dodać do projektu przez „Dodaj tablice lub zdjęcia…” → „Gotowe tablice + AZ”."
        ),
        wraplength=460,
        justify="left",
    ).grid(row=1, column=0, sticky="ew", pady=(0, 12))

    options = ttk.Frame(shell)
    options.grid(row=2, column=0, sticky="ew")
    options.grid_columnconfigure(0, weight=1)

    ordered = ("all", "approved", "selected", "recent")
    row_index = 0
    for key in ordered:
        item = scopes[key]
        button = ttk.Button(
            options,
            text=item["label"],
            command=lambda value=key: _finish(value),
            state=(tk.NORMAL if item["enabled"] else tk.DISABLED),
        )
        button.grid(
            row=row_index,
            column=0,
            sticky="ew",
            pady=(0, 6),
        )
        row_index += 1

    cancel = ttk.Button(
        shell,
        text="Anuluj",
        command=lambda: _finish(None),
    )
    cancel.grid(row=3, column=0, sticky="e", pady=(8, 0))

    def _center():
        try:
            dialog.update_idletasks()
            if parent is not None:
                px = parent.winfo_rootx()
                py = parent.winfo_rooty()
                pw = parent.winfo_width()
                ph = parent.winfo_height()
                dw = dialog.winfo_width()
                dh = dialog.winfo_height()
                x = px + max(0, (pw - dw) // 2)
                y = py + max(0, (ph - dh) // 2)
                dialog.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _finish(value):
        result["scope"] = value
        try:
            dialog.grab_release()
        except Exception:
            pass
        dialog.destroy()

    # Functions are resolved when Tk invokes the callback.
    dialog.protocol("WM_DELETE_WINDOW", lambda: _finish(None))
    dialog.bind("<Escape>", lambda _event: _finish(None), add="+")
    dialog.after_idle(_center)
    dialog.wait_window()
    return result["scope"]


def _build_unique_output_dir(
    parent: Path,
    *,
    project_name: str,
    iteration_num: int | None,
    scope: str,
) -> Path:
    safe_project = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(project_name or "projekt").strip(),
    ).strip("._") or "projekt"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"AZ_FREE_{safe_project}_{scope}_{stamp}" if iteration_num is None
            else f"AZ_{safe_project}_IT{int(iteration_num):03d}_{scope}_{stamp}")
    base = parent / name
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    return candidate
