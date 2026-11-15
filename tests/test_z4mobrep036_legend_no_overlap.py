"""Z4MOBREP036: independent, responsive legend and gesture guide.

Tests the actual _draw_latency_chart() against a real Tk Canvas (when Tk is
available) and proves read-only geometry of time data and zoom controls.
"""
from __future__ import annotations

import ast
from pathlib import Path
import tkinter as tk
import pytest

ROOT = Path(__file__).resolve().parents[1]
BROWSER = ROOT / "auto_annotation_tool/gui/z4_mobile_report_browser.py"


def test_source_keeps_inference_labels_and_moves_hint_below_series():
    src = BROWSER.read_text(encoding="utf-8")
    ast.parse(src)
    assert "legend_cols = 3 if plot_w >= 1000 else 2 if plot_w >= 580 else 1" in src
    assert "legend_rows = max(1, (len(present) + legend_cols - 1) // legend_cols)" in src
    assert "pad_t = hint_y + 20" in src
    assert "canvas.create_text(width - pad_r, hint_y, text=hint" in src
    assert "STAGE_DESCRIPTIONS if legend_column_width >= 280 else STAGE_SHORT" in src
    # Labels describe inference, not unmeasured pre/postprocessing.
    assert "self._latency_x_view = clamp_x_view" in src
    assert "source_min_x, source_max_x = 0.0" in src


def _plot_for_width(width, *, selected_zoom):
    from test_z4mobrep035_session_clock import _actual_plot_stub
    v, draw = _actual_plot_stub(zoom=selected_zoom)
    v._latency_x_view = selected_zoom
    v._latency_effective_mode = 'time'
    return v, draw


@pytest.mark.parametrize('width', [320, 420, 580, 760, 1050, 1200, 1600])
def test_tk_legend_bbox_never_collides_with_other_series_or_help(width):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Brak graficznego serwera Tk")
    try:
        root.geometry(f"{max(width + 20, 360)}x500+0+0")
        canvas = tk.Canvas(root, width=width, height=365, highlightthickness=0)
        canvas.pack()
        root.update()
        v, draw = _plot_for_width(width, selected_zoom=(9, 43))
        v.latency_canvas = canvas
        draw(v)
        assert v._latency_x_view == (9, 43)
        labels, help_lines = [], []
        for item in canvas.find_all():
            if canvas.type(item) != 'text':
                continue
            content = canvas.itemcget(item, 'text')
            bbox = canvas.bbox(item)
            if content.startswith(('ALPR:', 'MT:', 'MZ:')):
                labels.append((content, bbox))
            if content.startswith('Rolka:'):
                help_lines.append((content, bbox))
        assert len(labels) == 3
        assert len(help_lines) == 1
        for name, bounds in labels + help_lines:
            assert bounds is not None
            # Text stays inside the drawing surface, no silent cropping.
            assert bounds[0] >= 0, (width, name, bounds)
            assert bounds[2] <= width + 1, (width, name, bounds)
        all_bboxes = [bbox for _, bbox in labels + help_lines]
        for index, (ax1, ay1, ax2, ay2) in enumerate(all_bboxes):
            for bx1, by1, bx2, by2 in all_bboxes[index + 1:]:
                # No area intersection; touching edges is okay.
                assert ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1, (width, all_bboxes)
        # The first horizontal grid line is strictly below legend and guide.
        plot_bands = [canvas.coords(item) for item in canvas.find_all()
                      if canvas.type(item) == 'line' and len(canvas.coords(item)) == 4
                      and float(canvas.coords(item)[0]) == 50.0
                      and float(canvas.coords(item)[2]) == width-18
                      and float(canvas.coords(item)[1]) == float(canvas.coords(item)[3])]
        assert plot_bands
        assert max(bbox[3] for bbox in all_bboxes) < min(line[1] for line in plot_bands)
    finally:
        root.destroy()


def test_rendered_measurements_and_zoom_do_not_change_when_legend_wraps():
    from test_z4mobrep035_session_clock import _actual_plot_stub
    v, draw = _actual_plot_stub(zoom=(10, 40))
    previous_hits = list(v._latency_hit_points)
    previous_view = v._latency_x_view
    previous_full = v._latency_full_x
    v._latency_connect_neighbors_var.set(False)
    draw(v)
    assert v._latency_hit_points == previous_hits
    assert v._latency_x_view == previous_view
    assert v._latency_full_x == previous_full
    assert len(v._latency_hit_points) == 3 * 5  # 5 visible frames x 3 series
