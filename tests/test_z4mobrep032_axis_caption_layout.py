"""Z4MOBREP032/033: unclipped X label and link-only checkbox semantics."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import tkinter as tk
import pytest

ROOT = Path(__file__).resolve().parents[1]
BROWSER = ROOT / 'auto_annotation_tool' / 'gui' / 'z4_mobile_report_browser.py'


def _fixture_031():
    path = Path(__file__).with_name('test_z4mobrep031_sparse_zoom.py')
    spec = importlib.util.spec_from_file_location('z4mobrep031_for_032', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _browser():
    src = BROWSER.read_text(encoding='utf-8')
    ast.parse(src)
    return src


def test_checkbox_plain_label_only_controls_connectors():
    src = _browser()
    assert 'text="Łącz punkty sąsiednich klatek"' in src
    assert 'command=self._on_latency_connections_change' in src
    assert '_on_latency_spacing_change' not in src
    assert 'Przy dużych lukach:' not in src


def test_axis_description_is_packed_below_canvas_and_has_wrapping():
    src = _browser()
    first = src.index('self.latency_canvas.pack(fill=tk.BOTH, expand=True)')
    footer = src.index('self._latency_axis_footer_label.pack(', first)
    splitter = src.index('self.latency_split.add(latency_plot, weight=3)', footer)
    assert first < footer < splitter
    assert 'side=tk.BOTTOM, fill=tk.X, before=self.latency_canvas' in src
    assert 'wraplength=max(220, int(event.width) - 72)' in src
    assert 'textvariable=self._latency_axis_footer_var' in src
    assert 'canvas.create_text(pad_l, height - 5, text=axis_caption' not in src


def test_equal_spacing_footer_is_outside_canvas_and_not_a_time_axis():
    mod = _fixture_031()
    shapes, markers, fake = mod._render_case(False)
    assert 'Oś X: ID klatki z pomiarem czasu' in fake.footer_text
    assert 'punkty w równych odstępach' in fake.footer_text
    assert 'nie jest to oś czasu' in fake.footer_text
    texts = [shape for shape in shapes if shape[0] == 'text']
    assert all('nie jest to oś czasu' not in str(item[2].get('text')) for item in texts)
    assert all(item[1][1] < 345 for item in texts)
    assert len([x for x in markers if x[0] == 'MT']) == 9


def test_raw_id_spacing_legend_does_not_claim_equal_distances():
    mod = _fixture_031()
    _shapes, markers, fake = mod._render_case(False, frame_ids=list(range(200,208)))
    assert 'odstępy według numerów klatek' in fake.footer_text
    assert 'punkty w równych odstępach' not in fake.footer_text
    assert len([x for x in markers if x[0] == 'MZ']) == 9


def test_zoom_note_is_readable_below_plot_and_reports_hidden_counts():
    mod = _fixture_031()
    shapes, markers, fake = mod._render_case(False, zoom=(2, 5))
    assert 'Widoczne pomiary' not in str([x[2].get('text','') for x in shapes if x[0]=='text'])
    assert '4/8' in fake.footer_text
    assert '◀2' in fake.footer_text and '▶2' in fake.footer_text
    assert len([x for x in markers if x[0] == 'MT']) == 5


def test_zoom_equal_spacing_footer_counts_visible_observations_with_connections_on():
    mod = _fixture_031()
    _shapes, _markers, fake = mod._render_case(True, zoom=(2,5))
    assert '4/8' in fake.footer_text
    assert '◀2' in fake.footer_text and '▶2' in fake.footer_text


def test_tk_label_footer_physically_lies_below_plot_for_narrow_window():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Brak graficznego środowiska Tk; test pozostałych aspektów dostępny')
    try:
        root.geometry('420x330+0+0')
        frame = tk.Frame(root)
        frame.pack(fill='both', expand=True)
        top = tk.Checkbutton(frame, text='Łącz punkty sąsiednich klatek')
        top.pack(anchor='w')
        canvas = tk.Canvas(frame, height=210)
        canvas.pack(fill='both', expand=True)
        footer = tk.Label(frame, text='Oś X: ID klatki z pomiarem czasu · '
                                  'punkty w równych odstępach (nie jest to oś czasu)',
                          wraplength=330, anchor='w', justify='left')
        footer.pack(side=tk.BOTTOM, fill='x', before=canvas, padx=(50,18), pady=(3,6))
        # The Tk window manager can postpone mapping the new top-level window
        # until its event queue is serviced. update_idletasks() alone is not
        # sufficient on every Windows/Tk installation. Comparing coordinates
        # before mapping reports y=0, canvas_height=1 and is not a layout test.
        root.deiconify()
        for _ in range(5):
            root.update_idletasks()
            root.update()
            if (root.winfo_ismapped() and frame.winfo_height() > 1
                    and canvas.winfo_height() > 1 and footer.winfo_height() > 1):
                break
        if not root.winfo_ismapped():
            pytest.skip('Tk nie odwzorował okna; fizycznych współrzędnych nie można zmierzyć')
        assert frame.winfo_height() > 1 and canvas.winfo_height() > 1 and footer.winfo_height() > 1, (
            f'Okno jest odwzorowane, ale geometria niegotowa: frame={frame.winfo_geometry()}, '
            f'canvas={canvas.winfo_geometry()}, footer={footer.winfo_geometry()}')
        # Compare positions after real geometry has been calculated; this
        # still fails if the axis caption overlays or extends below the plot.
        assert canvas.winfo_y() + canvas.winfo_height() <= footer.winfo_y()
        assert footer.winfo_y() + footer.winfo_height() < frame.winfo_height()
    finally:
        root.destroy()
