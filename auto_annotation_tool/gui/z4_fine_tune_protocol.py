#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jawny protokół wykonania fine-tuningu Z4/PZ2."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

POLICY_FIXED = "Stałe parametry"
POLICY_ADAPTIVE = "Automatyczny RAM-safe"


def _var(host, name, factory):
    current = getattr(host, name, None)
    if current is None:
        current = factory()
        setattr(host, name, current)
    return current


def ensure_vars(host):
    _var(host, "ft_memory_policy_var", lambda: tk.StringVar(value=POLICY_ADAPTIVE))
    _var(host, "ft_amp_var", lambda: tk.BooleanVar(value=True))
    _var(host, "ft_mosaic_var", lambda: tk.DoubleVar(value=1.0))
    _var(host, "ft_cache_var", lambda: tk.BooleanVar(value=False))
    _var(host, "ft_workers_var", lambda: tk.IntVar(value=0))
    _var(host, "ft_plots_var", lambda: tk.BooleanVar(value=True))
    _var(host, "ft_seed_var", lambda: tk.IntVar(value=0))
    _var(host, "ft_optimizer_var", lambda: tk.StringVar(value="auto"))


def _fine_tune_active(host) -> bool:
    return bool(str(getattr(host, "_step4_fine_tune_parent_run_id", "") or "").strip())


def _ensure_panel_managed(host):
    frame = getattr(host, "ft_protocol_frame", None)
    if frame is None:
        return
    try:
        if frame.winfo_manager():
            return
    except Exception:
        pass

    note = getattr(host, "train_recommendation_note_lbl", None)
    kwargs = {"fill": tk.X, "pady": (2, 10)}
    if note is not None:
        kwargs["before"] = note
    try:
        frame.pack(**kwargs)
    except Exception:
        frame.pack(fill=tk.X, pady=(2, 10))


def build_panel(host, parent):
    ensure_vars(host)

    frame = ttk.LabelFrame(parent, text=" Fine-tuning: protokół wykonania ", padding=8)
    host.ft_protocol_frame = frame
    host._ft_protocol_controls = []

    intro = ttk.Label(
        frame,
        text=(
            "Jawne ustawienia nowego etapu dotrenowania. `Stałe parametry` uruchamiają "
            "dokładnie wpisany protokół. `Automatyczny RAM-safe` może zmniejszyć batch/imgsz "
            "i wyłączyć kosztowne opcje przy presji pamięci, ale nie zmienia LR."
        ),
        style="PanelMuted.TLabel",
        wraplength=350,
        justify=tk.LEFT,
    )
    intro.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 6))

    host.ft_protocol_status_lbl = ttk.Label(
        frame,
        text="Opcje uaktywnią się po wybraniu `Dotrenuj wybrany model`.",
        style="PanelMuted.TLabel",
        wraplength=350,
        justify=tk.LEFT,
    )
    host.ft_protocol_status_lbl.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 8))

    ttk.Label(frame, text="Polityka pamięci").grid(row=2, column=0, sticky="w", padx=(0, 6))
    policy_combo = ttk.Combobox(
        frame,
        textvariable=host.ft_memory_policy_var,
        state="readonly",
        values=(POLICY_ADAPTIVE, POLICY_FIXED),
        width=24,
    )
    policy_combo.grid(row=2, column=1, columnspan=3, sticky="ew", pady=2)
    host._ft_protocol_controls.append((policy_combo, "readonly"))

    amp_check = ttk.Checkbutton(frame, text="AMP", variable=host.ft_amp_var)
    amp_check.grid(row=3, column=0, sticky="w", pady=2)
    host._ft_protocol_controls.append((amp_check, tk.NORMAL))

    ttk.Label(frame, text="Mosaic").grid(row=3, column=1, sticky="e", padx=(8, 4))
    mosaic_spin = ttk.Spinbox(
        frame, from_=0.0, to=1.0, increment=0.05, format="%.2f", width=7,
        textvariable=host.ft_mosaic_var,
    )
    mosaic_spin.grid(row=3, column=2, sticky="w", pady=2)
    host._ft_protocol_controls.append((mosaic_spin, tk.NORMAL))

    cache_check = ttk.Checkbutton(frame, text="Cache", variable=host.ft_cache_var)
    cache_check.grid(row=3, column=3, sticky="w", padx=(10, 0), pady=2)
    host._ft_protocol_controls.append((cache_check, tk.NORMAL))

    ttk.Label(frame, text="Workers").grid(row=4, column=0, sticky="w", pady=2)
    workers_spin = ttk.Spinbox(
        frame, from_=0, to=32, increment=1, width=7,
        textvariable=host.ft_workers_var,
    )
    workers_spin.grid(row=4, column=1, sticky="w", pady=2)
    host._ft_protocol_controls.append((workers_spin, tk.NORMAL))

    plots_check = ttk.Checkbutton(frame, text="Wykresy", variable=host.ft_plots_var)
    plots_check.grid(row=4, column=2, sticky="w", pady=2)
    host._ft_protocol_controls.append((plots_check, tk.NORMAL))

    ttk.Label(frame, text="Seed").grid(row=5, column=0, sticky="w", pady=2)
    seed_spin = ttk.Spinbox(
        frame, from_=0, to=2147483647, increment=1, width=10,
        textvariable=host.ft_seed_var,
    )
    seed_spin.grid(row=5, column=1, sticky="w", pady=2)
    host._ft_protocol_controls.append((seed_spin, tk.NORMAL))

    ttk.Label(frame, text="Optimizer").grid(row=5, column=2, sticky="e", padx=(8, 4), pady=2)
    optimizer_combo = ttk.Combobox(
        frame,
        textvariable=host.ft_optimizer_var,
        state="readonly",
        values=("auto", "SGD", "AdamW", "Adam"),
        width=9,
    )
    optimizer_combo.grid(row=5, column=3, sticky="w", pady=2)
    host._ft_protocol_controls.append((optimizer_combo, "readonly"))

    frame.columnconfigure(1, weight=1)
    _ensure_panel_managed(host)
    refresh_visibility(host)
    return frame


def refresh_visibility(host):
    """Odśwież stan kontrolek bez usuwania panelu z geometrii."""
    frame = getattr(host, "ft_protocol_frame", None)
    if frame is None:
        return

    _ensure_panel_managed(host)
    active = _fine_tune_active(host)

    status = getattr(host, "ft_protocol_status_lbl", None)
    if status is not None:
        try:
            if active:
                parent_id = str(getattr(host, "_step4_fine_tune_parent_run_id", "") or "").strip()
                status.configure(
                    text=(
                        f"Aktywne dla dotrenowania od runu {parent_id}. "
                        "Poniższe wartości zostaną zapisane w protokole runu."
                    )
                )
            else:
                status.configure(
                    text="Opcje uaktywnią się po wybraniu `Dotrenuj wybrany model`."
                )
        except Exception:
            pass

    for widget, active_state in list(getattr(host, "_ft_protocol_controls", ()) or ()):
        try:
            widget.configure(state=(active_state if active else tk.DISABLED))
        except Exception:
            pass


def collect(host) -> dict:
    ensure_vars(host)

    try:
        mosaic = max(0.0, min(1.0, float(host.ft_mosaic_var.get())))
    except Exception:
        mosaic = 1.0
    try:
        workers = max(0, int(host.ft_workers_var.get()))
    except Exception:
        workers = 0
    try:
        seed = max(0, int(host.ft_seed_var.get()))
    except Exception:
        seed = 0

    optimizer = str(host.ft_optimizer_var.get() or "auto").strip() or "auto"
    policy = str(host.ft_memory_policy_var.get() or POLICY_ADAPTIVE).strip()
    adaptive = policy != POLICY_FIXED

    return {
        "schema": "alpr.fine_tune_runtime_protocol.v1",
        "ui_fine_tune_protocol": True,
        "memory_policy": "adaptive" if adaptive else "fixed",
        "ram_safe_enabled": adaptive,
        "amp": bool(host.ft_amp_var.get()),
        "mosaic": mosaic,
        "close_mosaic": 0 if mosaic <= 0.0 else None,
        "cache": bool(host.ft_cache_var.get()),
        "workers": workers,
        "plots": bool(host.ft_plots_var.get()),
        "seed": seed,
        "optimizer": optimizer,
    }
