"""Small Z2 manager for mounted ALPR GT Packs."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..campaign_manager import CAMPAIGN
from ..gt_pack import ALPRGTPack
from . import z2_gt_pack_runtime


def _validate_pack(path: Path) -> tuple[bool, str]:
    try:
        pack = ALPRGTPack.open(path)
        result = pack.validate(deep=False)
    except Exception as exc:
        return False, str(exc)
    if not result.get("ok"):
        issues = list(result.get("issues", []) or [])
        return False, "; ".join(str(v) for v in issues[:3])
    return True, ""


def _project_default_working_path() -> Path | None:
    try:
        project_name = str(CAMPAIGN.get_active_project_name() or "").strip()
        root = CAMPAIGN.get_active_project_root_dir() if project_name else None
    except Exception:
        root = None
    if root is None:
        return None
    return Path(root) / "_campaign_state" / "ground_truth" / "current_work.alprgt"


def open_gt_pack_manager(host) -> None:
    parent = getattr(host, "frame", None)
    window = tk.Toplevel(parent)
    window.title("ALPR GT Pack")
    try:
        window.transient(parent.winfo_toplevel())
    except Exception:
        pass
    window.resizable(True, True)
    window.minsize(680, 420)

    outer = ttk.Frame(window, padding=12)
    outer.pack(fill=tk.BOTH, expand=True)

    ttk.Label(
        outer,
        text="Źródła GT (read-only)",
        font=("Segoe UI", 10, "bold"),
    ).pack(anchor="w")

    list_frame = ttk.Frame(outer)
    list_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 8))

    source_list = tk.Listbox(
        list_frame,
        selectmode=tk.EXTENDED,
        exportselection=False,
        height=8,
    )
    source_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(
        list_frame,
        orient=tk.VERTICAL,
        command=source_list.yview,
    )
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    source_list.configure(yscrollcommand=scrollbar.set)

    for path in z2_gt_pack_runtime.get_configured_source_pack_paths(host):
        source_list.insert(tk.END, str(path))

    source_buttons = ttk.Frame(outer)
    source_buttons.pack(fill=tk.X, pady=(0, 12))

    def add_source():
        selected = filedialog.askdirectory(
            parent=window,
            title="Wybierz katalog .alprgt",
        )
        if not selected:
            return
        path = Path(selected)
        ok, error = _validate_pack(path)
        if not ok:
            messagebox.showerror(
                "Nieprawidłowy GT Pack",
                error or "Wybrany katalog nie jest poprawnym ALPR GT Pack.",
                parent=window,
            )
            return
        existing = {source_list.get(i) for i in range(source_list.size())}
        if str(path) not in existing:
            source_list.insert(tk.END, str(path))

    def remove_source():
        for index in reversed(source_list.curselection()):
            source_list.delete(index)

    ttk.Button(
        source_buttons,
        text="Dodaj źródło…",
        command=add_source,
    ).pack(side=tk.LEFT)
    ttk.Button(
        source_buttons,
        text="Usuń zaznaczone",
        command=remove_source,
    ).pack(side=tk.LEFT, padx=(8, 0))

    ttk.Separator(outer).pack(fill=tk.X, pady=(0, 12))

    ttk.Label(
        outer,
        text="Roboczy GT Pack (writable)",
        font=("Segoe UI", 10, "bold"),
    ).pack(anchor="w")

    working_var = tk.StringVar(
        value=str(
            z2_gt_pack_runtime.get_working_gt_pack_path(
                host,
                create_parent=False,
            )
            or ""
        )
    )

    working_row = ttk.Frame(outer)
    working_row.pack(fill=tk.X, pady=(6, 6))
    ttk.Entry(
        working_row,
        textvariable=working_var,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def choose_working():
        selected = filedialog.askdirectory(
            parent=window,
            title="Wybierz istniejący .alprgt albo katalog dla current_work.alprgt",
        )
        if not selected:
            return
        path = Path(selected)
        if (path / "manifest.json").is_file():
            ok, error = _validate_pack(path)
            if not ok:
                messagebox.showerror(
                    "Nieprawidłowy GT Pack",
                    error,
                    parent=window,
                )
                return
            working_var.set(str(path))
            return

        candidate = path / "current_work.alprgt"
        if messagebox.askyesno(
            "Nowy GT Pack",
            (
                "Wybrany katalog nie jest GT Packiem.\n\n"
                f"Użyć jako lokalizacji nowego:\n{candidate}?"
            ),
            parent=window,
        ):
            working_var.set(str(candidate))

    def use_default():
        default = _project_default_working_path()
        if default is None:
            working_var.set("")
            messagebox.showinfo(
                "Tryb swobodny",
                "W trybie swobodnym nie ma domyślnego packa. Wskaż własny katalog .alprgt.",
                parent=window,
            )
            return
        working_var.set(str(default))

    ttk.Button(
        working_row,
        text="Wybierz…",
        command=choose_working,
    ).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Button(
        working_row,
        text="Domyślny",
        command=use_default,
    ).pack(side=tk.LEFT, padx=(8, 0))

    status_var = tk.StringVar(value="")
    ttk.Label(outer, textvariable=status_var).pack(anchor="w", pady=(8, 10))

    def refresh_status():
        status = z2_gt_pack_runtime.get_gt_pack_status(host)
        status_var.set(
            " | ".join(
                (
                    f"źródła: {status['source_count']}",
                    f"pending: {status['pending_count']}",
                    f"conflicts: {status['conflict_count']}",
                )
            )
        )

    def retry_sync():
        result = z2_gt_pack_runtime.retry_pending_gt_sync(host)
        refresh_status()
        messagebox.showinfo(
            "Synchronizacja GT",
            (
                f"Ponowiono: {result.get('retried', 0)}\n"
                f"Pozostało: {result.get('remaining', 0)}"
            ),
            parent=window,
        )

    action_row = ttk.Frame(outer)
    action_row.pack(fill=tk.X)
    ttk.Button(
        action_row,
        text="Ponów pending",
        command=retry_sync,
    ).pack(side=tk.LEFT)

    def save_and_close():
        source_values = [
            source_list.get(index)
            for index in range(source_list.size())
        ]
        working = str(working_var.get() or "").strip()

        if working:
            path = Path(working)
            if path.exists() and (path / "manifest.json").is_file():
                ok, error = _validate_pack(path)
                if not ok:
                    messagebox.showerror(
                        "Nieprawidłowy writable GT Pack",
                        error,
                        parent=window,
                    )
                    return

        z2_gt_pack_runtime.set_gt_pack_mounts(
            host,
            source_paths=source_values,
            working_path=working,
            persist=True,
        )
        try:
            z2_gt_pack_runtime.retry_pending_gt_sync(host)
        except Exception:
            pass
        try:
            host._refresh_preview_canvas()
        except Exception:
            pass
        window.destroy()

    ttk.Button(
        action_row,
        text="Zapisz",
        command=save_and_close,
    ).pack(side=tk.RIGHT)
    ttk.Button(
        action_row,
        text="Anuluj",
        command=window.destroy,
    ).pack(side=tk.RIGHT, padx=(0, 8))

    refresh_status()
    try:
        window.grab_set()
    except Exception:
        pass
    window.focus_set()
