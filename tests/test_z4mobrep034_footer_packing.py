"""Z4MOBREP034: axis caption must survive vertical sash/table resizing."""
from __future__ import annotations

import ast
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import pytest

BROWSER = Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/z4_mobile_report_browser.py"


def test_source_reserves_footer_before_stretchable_canvas():
    source = BROWSER.read_text(encoding="utf-8")
    ast.parse(source)
    # The footer is allocated before the canvas using Tk's explicit packing
    # order, even though its construction appears after that of the canvas.
    assert "side=tk.BOTTOM, fill=tk.X, before=self.latency_canvas," in source
    assert "self.latency_canvas.pack(fill=tk.BOTH, expand=True)" in source
    assert "self._latency_axis_footer_label.pack(" in source
    assert "self.latency_split.add(latency_plot, weight=3)" in source
    assert "self.latency_split.add(latency_metrics, weight=1)" in source


def test_tk_footer_is_visible_when_user_expands_table():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Brak serwera graficznego Tk")
    try:
        root.geometry("760x580+0+0")
        split = ttk.PanedWindow(root, orient=tk.VERTICAL)
        split.pack(fill=tk.BOTH, expand=True)
        plot = tk.Frame(split)
        metrics = tk.Frame(split)
        split.add(plot, weight=3)
        split.add(metrics, weight=1)
        bar = tk.Checkbutton(plot, text="Łącz punkty sąsiednich klatek")
        bar.pack(fill=tk.X)
        canvas = tk.Canvas(plot, height=320)
        canvas.pack(fill=tk.BOTH, expand=True)
        # Mirror the production packing order, not only the labels' text.
        footer = tk.Label(
            plot, text="Oś X: ID klatki z pomiarem czasu · punkty w równych "
                       "odstępach (nie jest to oś czasu)",
            wraplength=350, anchor=tk.W, justify=tk.LEFT,
        )
        footer.pack(side=tk.BOTTOM, fill=tk.X, before=canvas,
                    padx=(50, 18), pady=(3, 6))
        root.deiconify()
        for _ in range(6):
            root.update_idletasks()
            root.update()
            if root.winfo_ismapped() and plot.winfo_height() > 100:
                break
        if not root.winfo_ismapped():
            pytest.skip("Tk nie odwzorował okna; brak możliwości pomiaru GUI")
        for height in (400, 260, 185, 135):
            split.sashpos(0, height)
            root.update_idletasks()
            root.update()
            assert plot.winfo_height() >= height - 12
            assert footer.winfo_ismapped(), f"Znikająca stopka przy wysokości {height}"
            assert footer.winfo_height() >= footer.winfo_reqheight(), (
                f"Przycięta stopka dla {height}: "
                f"rzeczywista={footer.winfo_height()} wymagana={footer.winfo_reqheight()}"
            )
            assert (canvas.winfo_y() + canvas.winfo_height()
                    <= footer.winfo_y()), f"Nachodzące płótno przy wysokości {height}"
            assert (footer.winfo_y() + footer.winfo_height()
                    <= plot.winfo_height()), f"Stopka poza panelem przy wysokości {height}"
            assert canvas.winfo_height() > 4
    finally:
        root.destroy()
