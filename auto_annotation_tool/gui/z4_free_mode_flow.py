from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

from ..campaign_manager import CAMPAIGN

if TYPE_CHECKING:
    from .tab_training import TrainingTab


def refresh_free_training_route_cards(host: "TrainingTab"):
    title = getattr(host, "free_training_route_title_lbl", None)
    if title is None:
        return
    try:
        title.configure(
            text=("Przepływ PZ1 -> PZ2" if not CAMPAIGN.get_active_project_name() else "Wybrany tor treningu")
        )
    except Exception:
        pass


def refresh_free_training_route_ui(host: "TrainingTab"):
    route_host = getattr(host, "free_training_route_host", None)
    if route_host is None:
        return

    try:
        pack_options = {"anchor": tk.W, "fill": tk.X, "pady": (0, 12)}
        parent = route_host.master
        try:
            siblings = [widget for widget in parent.pack_slaves() if widget is not route_host]
        except Exception:
            siblings = []
        if siblings:
            route_host.pack(**pack_options, before=siblings[0])
        else:
            route_host.pack(**pack_options)
    except Exception:
        pass

    refresh_free_training_route_cards(host)
    host._refresh_training_start_state()


def update_step4_notebook_mode(host: "TrainingTab"):
    if not hasattr(host, "main_nb"):
        return

    campaign_active = bool(CAMPAIGN.get_active_project_name())
    dataset_label = "[PZ1] Wariant splitu"
    train_label = "[PZ2] Trening na wariancie"

    try:
        host.main_nb.tab(host.tab_train, text=train_label)
    except Exception:
        pass

    if not getattr(host, "_step4_dataset_tab_visible", False):
        try:
            host.main_nb.insert(0, host.tab_dataset, text=dataset_label)
        except Exception:
            try:
                host.main_nb.add(host.tab_dataset, text=dataset_label)
            except Exception:
                pass
        host._step4_dataset_tab_visible = True
    else:
        try:
            host.main_nb.tab(host.tab_dataset, text=dataset_label)
        except Exception:
            pass

    if campaign_active:
        return

    try:
        current_tab = str(host.main_nb.select() or "").strip()
        if current_tab not in {str(host.tab_dataset), str(host.tab_train)}:
            host.main_nb.select(host.tab_dataset)
    except Exception:
        pass
