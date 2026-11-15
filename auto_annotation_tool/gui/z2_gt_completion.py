"""Z2 completion/status UI for optional writable character Ground Truth."""

from __future__ import annotations

import datetime
import os
import shutil
from pathlib import Path
from tkinter import filedialog, messagebox

from ..gt_pack import ALPRGTPack
from ..gt_resource_companions import DEFAULT_WORKING_PACK_NAME, inspect_gt_pack
from . import z2_gt_pack_runtime


def _gt_enabled(host) -> bool:
    # GT is optional data and is always available.
    return True


def _working_pack_path(host) -> Path | None:
    if not _gt_enabled(host):
        return None

    raw_resource = str(
        getattr(host, "_z2_gt_resource_working_pack_path", "") or ""
    ).strip()
    if raw_resource:
        return Path(raw_resource)

    try:
        resolved = z2_gt_pack_runtime.get_working_gt_pack_path(
            host,
            create_parent=False,
        )
    except Exception:
        resolved = None
    if resolved is not None:
        return Path(resolved)

    candidates = [
        getattr(host, "current_input_dir", None),
    ]
    try:
        candidates.append(host.input_dir_var.get())
    except Exception:
        pass

    for raw in candidates:
        if not str(raw or "").strip():
            continue
        path = Path(raw)
        if path.is_dir():
            return path / DEFAULT_WORKING_PACK_NAME
    return None


def build_gt_completion_state(pack_path: Path | str | None) -> dict:
    result = {
        "path": str(pack_path or ""),
        "exists": False,
        "valid": False,
        "complete": False,
        "status": "BRAK PAKIETU",
        "images": 0,
        "plates": 0,
        "gt_set": 0,
        "layout_set": 0,
        "geometry_set": 0,
        "missing_gt": 0,
        "missing_layout": 0,
        "missing_geometry": 0,
        "conflicts": 0,
        "issues": [],
    }
    if pack_path is None:
        return result

    path = Path(pack_path)
    result["path"] = str(path)
    if not path.is_dir() or not (path / "manifest.json").is_file():
        return result

    result["exists"] = True
    info = dict(inspect_gt_pack(path) or {})
    result["valid"] = bool(info.get("valid"))
    result["issues"] = [
        str(value)
        for value in list(info.get("issues", []) or [])
    ]

    plates = int(info.get("plates", 0) or 0)
    gt_set = int(info.get("gt_set", 0) or 0)
    layout_set = int(info.get("layout_set", 0) or 0)
    conflicts = int(info.get("gt_conflicts", 0) or 0)
    conflicts += int(info.get("layout_conflicts", 0) or 0)
    conflicts += int(info.get("geometry_conflicts", 0) or 0)

    geometry_set = 0
    if result["valid"]:
        try:
            pack = ALPRGTPack.open(path)
            for plate in pack.list_plates():
                plate_id = str(plate.get("plate_id") or "").strip()
                if not plate_id:
                    continue
                state = dict(pack.resolve_plate_geometry(plate_id) or {})
                if state.get("resolved") and not state.get("conflict"):
                    geometry_set += 1
        except Exception as exc:
            result["valid"] = False
            result["issues"].append(str(exc))

    missing_gt = max(0, plates - gt_set)
    missing_layout = max(0, plates - layout_set)
    missing_geometry = max(0, plates - geometry_set)

    complete = bool(
        result["valid"]
        and plates > 0
        and missing_gt == 0
        and missing_layout == 0
        and missing_geometry == 0
        and conflicts == 0
    )

    result.update(
        {
            "images": int(info.get("images", 0) or 0),
            "plates": plates,
            "gt_set": gt_set,
            "layout_set": layout_set,
            "geometry_set": geometry_set,
            "missing_gt": missing_gt,
            "missing_layout": missing_layout,
            "missing_geometry": missing_geometry,
            "conflicts": conflicts,
            "complete": complete,
            "status": "GOTOWE" if complete else "WYMAGA UZUPEŁNIENIA",
        }
    )
    return result


def export_gt_snapshot_pack(
    source_pack: Path | str,
    destination_dir: Path | str,
    *,
    timestamp: str | None = None,
) -> dict:
    source = Path(source_pack)
    destination_root = Path(destination_dir)
    source_info = dict(inspect_gt_pack(source) or {})
    if not source_info.get("valid"):
        issues = "; ".join(source_info.get("issues", [])[:3])
        raise ValueError(
            "Roboczy GT Pack nie przeszedł walidacji"
            + (f": {issues}" if issues else ".")
        )
    if not destination_root.exists() or not destination_root.is_dir():
        raise ValueError(
            f"Nieprawidłowy katalog docelowy: {destination_root}"
        )

    stamp = str(
        timestamp
        or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ).strip()
    base_name = f"gt_copy_{stamp}.alprgt"
    target = destination_root / base_name
    suffix = 1
    while target.exists():
        target = destination_root / f"gt_copy_{stamp}_{suffix:02d}.alprgt"
        suffix += 1

    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(".pack.lock"),
    )

    try:
        copied = ALPRGTPack.open(target)
        validation = dict(copied.validate(deep=True) or {})
    except Exception as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise ValueError(
            f"Nie udało się zwalidować snapshotu GT: {exc}"
        ) from exc

    if not validation.get("ok"):
        issues = "; ".join(
            str(value)
            for value in list(validation.get("issues", []) or [])[:3]
        )
        shutil.rmtree(target, ignore_errors=True)
        raise ValueError(
            "Snapshot GT nie przeszedł walidacji"
            + (f": {issues}" if issues else ".")
        )

    return {
        "path": target,
        "validation": validation,
        "summary": dict(inspect_gt_pack(target) or {}),
    }


def build_gt_completion_subpanel(host, parent, *, key: str):
    """Create a compact GT block nested inside an existing correction card."""
    import tkinter as tk
    from tkinter import ttk

    palette = getattr(getattr(host, "app", None), "palette", {}) or {}
    panel_bg = palette.get("panel_alt", palette.get("panel", "#252526"))
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")

    box = tk.Frame(
        parent,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        bg=panel_bg,
        padx=10,
        pady=8,
    )

    title = tk.Label(
        box,
        text="GT znaków",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI Semibold", 9),
        bd=0,
        highlightthickness=0,
        bg=panel_bg,
        fg=fg,
    )
    title.pack(anchor=tk.W, fill=tk.X, pady=(0, 3))

    status = tk.Label(
        box,
        text="Oczekiwanie na dane",
        anchor="w",
        justify=tk.LEFT,
        wraplength=330,
        bd=0,
        highlightthickness=0,
        bg=panel_bg,
        fg=muted,
    )
    status.pack(anchor=tk.W, fill=tk.X)

    details = tk.Label(
        box,
        text="",
        anchor="w",
        justify=tk.LEFT,
        wraplength=330,
        bd=0,
        highlightthickness=0,
        bg=panel_bg,
        fg=muted,
    )
    details.pack(anchor=tk.W, fill=tk.X, pady=(3, 5))

    pack_name = tk.Label(
        box,
        text="",
        anchor="w",
        justify=tk.LEFT,
        wraplength=330,
        bd=0,
        highlightthickness=0,
        bg=panel_bg,
        fg=muted,
    )
    pack_name.pack(anchor=tk.W, fill=tk.X, pady=(0, 6))

    buttons = ttk.Frame(box, style="Panel.TFrame")
    buttons.pack(fill=tk.X)
    buttons.columnconfigure(0, weight=1)
    buttons.columnconfigure(1, weight=1)

    open_button = ttk.Button(
        buttons,
        text="Otwórz pakiet",
        style="WorkflowCard.TButton",
        command=host._open_gt_pack_folder,
    )
    open_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

    snapshot_button = ttk.Button(
        buttons,
        text="Zapisz kopię GT…",
        style="WorkflowCard.TButton",
        command=host._export_gt_snapshot,
    )
    snapshot_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    views = getattr(host, "_gt_completion_views", None)
    if not isinstance(views, dict):
        views = {}
        host._gt_completion_views = views
    views[str(key)] = {
        "box": box,
        "title": title,
        "status": status,
        "details": details,
        "pack_name": pack_name,
        "open_button": open_button,
        "snapshot_button": snapshot_button,
    }

    box.pack(fill=tk.X, pady=(8, 8))
    box.pack_forget()
    return box


def _completion_view_key(host) -> str:
    try:
        if not host._is_free_mode_session_context():
            return ""
    except Exception:
        return ""

    try:
        screen = str(host._coerce_free_mode_screen() or "").strip().lower()
    except Exception:
        screen = ""

    if screen == "manual_review":
        return "manual"
    if screen == "auto_summary":
        return "followup"
    return ""


def refresh_gt_completion_ui(host, *, visible: bool | None = None) -> dict:
    views = getattr(host, "_gt_completion_views", None)
    if not isinstance(views, dict):
        return {}

    # GT is a sub-section of Correction, never a separate workflow card.
    for view in views.values():
        box = (view or {}).get("box")
        if box is not None:
            try:
                box.pack_forget()
            except Exception:
                pass

    if not _gt_enabled(host):
        return {}

    key = _completion_view_key(host)
    if not key:
        return {}

    view = views.get(key)
    if not isinstance(view, dict):
        return {}

    path = _working_pack_path(host)
    state = build_gt_completion_state(path)

    box = view.get("box")
    if box is not None:
        try:
            box.pack(fill="x", pady=(8, 8))
        except Exception:
            pass

    complete = bool(state.get("complete"))
    exists = bool(state.get("exists"))
    valid = bool(state.get("valid"))

    status_text = str(
        state.get("status")
        or ("BRAK PAKIETU" if not exists else "WYMAGA UZUPEŁNIENIA")
    )
    status = view.get("status")
    if status is not None:
        try:
            host._set_inline_label_state(
                status,
                text=status_text,
                tone=("success" if complete else "warning"),
                emphasis=True,
            )
        except Exception:
            try:
                status.configure(text=status_text)
            except Exception:
                pass

    if exists:
        details_text = (
            f"GT: {state['gt_set']}/{state['plates']}  •  "
            f"1R/2R: {state['layout_set']}/{state['plates']}  •  "
            f"geometria: {state['geometry_set']}/{state['plates']}"
        )
        if (
            state.get("missing_gt")
            or state.get("missing_layout")
            or state.get("missing_geometry")
            or state.get("conflicts")
        ):
            details_text += (
                f"\nBraki: GT {state['missing_gt']}, "
                f"układ {state['missing_layout']}, "
                f"geometria {state['missing_geometry']}  •  "
                f"konflikty {state['conflicts']}"
            )
        elif complete:
            details_text += "\nPakiet jest kompletny."
    else:
        details_text = (
            "Pakiet roboczy nie został jeszcze utworzony dla bieżącego źródła."
        )

    details = view.get("details")
    if details is not None:
        try:
            details.configure(text=details_text)
        except Exception:
            pass

    pack_label = view.get("pack_name")
    if pack_label is not None:
        try:
            pack_label.configure(
                text=(
                    f"Pakiet: {Path(state['path']).name}"
                    if state.get("path")
                    else "Pakiet: brak"
                )
            )
        except Exception:
            pass

    open_state = "normal" if exists else "disabled"
    copy_state = (
        "normal"
        if exists and valid and int(state.get("gt_set", 0) or 0) >= 1
        else "disabled"
    )

    open_button = view.get("open_button")
    if open_button is not None:
        try:
            open_button.configure(state=open_state)
        except Exception:
            pass

    snapshot_button = view.get("snapshot_button")
    if snapshot_button is not None:
        try:
            snapshot_button.configure(state=copy_state)
        except Exception:
            pass

    if exists and valid and int(state.get("gt_set", 0) or 0) <= 0:
        if details is not None:
            try:
                details.configure(
                    text=details_text
                    + "\nKopia GT będzie dostępna po zapisaniu pierwszego GT."
                )
            except Exception:
                pass

    return state


def open_gt_pack_folder(host) -> None:
    path = _working_pack_path(host)
    if path is None or not path.is_dir():
        messagebox.showinfo(
            "Ground Truth znaków",
            "Nie znaleziono aktywnego roboczego pakietu GT.",
            parent=getattr(host, "frame", None),
        )
        return

    try:
        if os.name == "nt":
            os.startfile(str(path))
        else:
            messagebox.showinfo(
                "Ground Truth znaków",
                f"Pakiet GT:\n{path}",
                parent=getattr(host, "frame", None),
            )
    except Exception as exc:
        messagebox.showerror(
            "Nie można otworzyć pakietu GT",
            str(exc),
            parent=getattr(host, "frame", None),
        )


def export_gt_snapshot_dialog(host) -> None:
    source = _working_pack_path(host)
    state = build_gt_completion_state(source)
    if not state.get("exists") or not state.get("valid"):
        messagebox.showwarning(
            "Zapisz kopię GT",
            "Roboczy pakiet GT nie jest gotowy do skopiowania.",
            parent=getattr(host, "frame", None),
        )
        return
    if int(state.get("gt_set", 0) or 0) <= 0:
        messagebox.showinfo(
            "Zapisz kopię GT",
            "Kopia GT będzie dostępna po zapisaniu pierwszego GT.",
            parent=getattr(host, "frame", None),
        )
        return

    initial = None
    try:
        current_input = Path(host.input_dir_var.get())
        if current_input.is_dir():
            initial = current_input
    except Exception:
        initial = None

    destination = filedialog.askdirectory(
        parent=getattr(host, "frame", None),
        title="Wybierz folder dla kopii GT",
        initialdir=str(initial or Path.home()),
    )
    if not destination:
        return

    try:
        result = export_gt_snapshot_pack(
            source,
            Path(destination),
        )
    except Exception as exc:
        messagebox.showerror(
            "Nie można zapisać kopii GT",
            str(exc),
            parent=getattr(host, "frame", None),
        )
        return

    completeness = (
        "Pakiet źródłowy był kompletny."
        if state.get("complete")
        else "Kopia zawiera bieżący, jeszcze niepełny stan GT."
    )
    messagebox.showinfo(
        "Kopia GT zapisana",
        f"Utworzono:\n{result['path']}\n\n{completeness}",
        parent=getattr(host, "frame", None),
    )
