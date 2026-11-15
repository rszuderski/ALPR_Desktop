"""Z4MOBREP019 FIX04 — latency plot dominates, metrics stay scrollable.

Presentation-only regression. Existing trace alignment, colors, and summaries
remain outside this change.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BROWSER = ROOT / 'auto_annotation_tool/gui/z4_mobile_report_browser.py'
SOURCE = BROWSER.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)
CLASS = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == 'MobileReportBrowser')
BUILD = next(node for node in CLASS.body if isinstance(node, ast.FunctionDef) and node.name == '_build_tabs')
DRAW = next(node for node in CLASS.body if isinstance(node, ast.FunctionDef) and node.name == '_draw_latency_chart')
BUILD_SRC = ast.get_source_segment(SOURCE, BUILD)
DRAW_SRC = ast.get_source_segment(SOURCE, DRAW)


def test_chart_and_table_have_vertical_resizable_split():
    assert 'self.latency_split = ttk.PanedWindow(latency, orient=tk.VERTICAL)' in BUILD_SRC
    assert 'self.latency_split.grid(row=0, column=0, sticky="nsew")' in BUILD_SRC
    assert 'self.latency_split.add(latency_plot, weight=3)' in BUILD_SRC
    assert 'self.latency_split.add(latency_metrics, weight=1)' in BUILD_SRC
    assert 'self.technical_notebook.add(latency, text="Czasy szczegółowe")' in BUILD_SRC


def test_canvas_is_big_and_expands_with_pane():
    assert 'highlightbackground=self.border, height=320)' in BUILD_SRC
    assert 'self.latency_canvas.pack(fill=tk.BOTH, expand=True)' in BUILD_SRC
    assert 'self.latency_canvas.bind("<Configure>", lambda _event: self._draw_latency_chart(), add="+")' in BUILD_SRC
    assert 'self.latency_canvas.grid(row=0, column=0, sticky="ew"' not in BUILD_SRC


def test_table_gets_less_initial_space_and_preserves_scrolling():
    assert 'palette={**self.palette, "panel": self.panel}, height=5)' in BUILD_SRC
    assert 'self.latency_table.shell.pack(fill=tk.BOTH, expand=True)' in BUILD_SRC
    assert 'self.latency_table.set_rows' in SOURCE
    assert 'scroll = WebSlimScrollbar(' in SOURCE
    assert 'command=self.tree.yview' in SOURCE
    assert 'self.tree.configure(yscrollcommand=scroll.set)' in SOURCE


def test_plot_keeps_evidence_faithful_frame_axis_and_chart_styles():
    assert 'timeline_stages(bundle.trace_rows' in DRAW_SRC
    assert 'axis_ticks(xs, plot_w, timeline["axis_label"])' in DRAW_SRC
    assert 'continuous_segments(values, xs' in DRAW_SRC
    assert 'pixel_column_extrema(xs, values, min_x, max_x, plot_w)' in DRAW_SRC
    assert 'LATENCY_SERIES_STYLES' in DRAW_SRC
    assert 'draw_chart_marker(' in DRAW_SRC
    assert 'smooth=False' in DRAW_SRC


def test_no_extra_buttons_or_source_mutation_in_latency_tab():
    assert 'Z4MOBREP019_FIX04' in BUILD_SRC
    assert 'latency = tab_frame(self.technical_notebook)' in BUILD_SRC
    assert 'command=self.import_report' not in BUILD_SRC[BUILD_SRC.index('latency = tab_frame(self.technical_notebook)'):BUILD_SRC.index('self.technical_notebook.add(latency, text="Czasy szczegółowe")')]
