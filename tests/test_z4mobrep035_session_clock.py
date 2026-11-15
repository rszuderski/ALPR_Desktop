"""Z4MOBREP035: honest time axis, concrete MT/MZ labels, stable connector toggle."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import zipfile
import csv
import io

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / 'auto_annotation_tool/gui'
BROWSER = GUI / 'z4_mobile_report_browser.py'
TIMES = GUI / 'mobile_report_session_timeline.py'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HELPER = _load('z4mobrep035_timeline_test', TIMES)


def _data():
    starts_at = '2026-10-18T23:19:09.654Z'
    ends_at = '2026-10-18T23:20:09.659Z'
    start_ms = int(datetime.fromisoformat(starts_at.replace('Z', '+00:00')).timestamp() * 1000)
    frame_ids = [384, 386, 387, 388, 389, 390, 391, 585]
    elapsed_ms = [737, 2440, 10371, 11957, 20858, 31160, 38560, 51249]
    rows = [{
        'frame_id': str(frame), 'elapsed_ms': str(offset),
        'timestamp_ms': str(start_ms + offset),
        'total_ms': str(1300 + (index * 115)),
        'plate_inference_ms': str(620 + index * 12),
        'character_inference_ms': str(245 + index * 16),
    } for index, (frame, offset) in enumerate(zip(frame_ids, elapsed_ms))]
    bundle = SimpleNamespace(
        collection_session={'started_at': starts_at, 'finished_at': ends_at},
        trace_rows=rows,
    )
    report = SimpleNamespace(raw={
        'session_started_ms': start_ms, 'session_finished_ms': start_ms + 60005,
        'session_duration_ms': 60005,
    })
    return bundle, report


def test_real_session_window_and_all_8_trace_offsets():
    bundle, report = _data()
    context = HELPER.session_time_context(bundle, report)
    assert context['duration_ms'] == 60005
    seconds = HELPER.session_elapsed_seconds(bundle.trace_rows, context)
    assert seconds == (.737, 2.44, 10.371, 11.957, 20.858, 31.16, 38.56, 51.249)
    assert '1:00,005' in HELPER.session_caption(context)
    ticks = HELPER.time_ticks(0, 60.005)
    assert ticks[0] == (0, '0:00')
    assert ticks[-1] == (60, '1:00')


def test_clock_missing_or_inconsistent_must_not_be_guessed():
    bundle, report = _data()
    context = HELPER.session_time_context(bundle, report)
    broken = [dict(row) for row in bundle.trace_rows]
    broken[4]['elapsed_ms'] = ''
    broken[4]['timestamp_ms'] = ''
    assert HELPER.session_elapsed_seconds(broken, context) is None
    broken = [dict(row) for row in bundle.trace_rows]
    broken[2]['timestamp_ms'] = str(int(broken[2]['timestamp_ms']) + 8000)
    assert HELPER.session_elapsed_seconds(broken, context) is None
    broken = [dict(row) for row in bundle.trace_rows]
    broken[3]['elapsed_ms'] = broken[2]['elapsed_ms']
    broken[3]['timestamp_ms'] = broken[2]['timestamp_ms']
    assert HELPER.session_elapsed_seconds(broken, context) is None
    no_window = SimpleNamespace(collection_session={})
    no_report = SimpleNamespace(raw={})
    context = HELPER.session_time_context(no_window, no_report)
    assert context['duration_ms'] is None
    assert HELPER.session_elapsed_seconds(bundle.trace_rows, context) is None


def test_model_inference_is_not_preprocessing_and_point_provenance_is_exact():
    assert HELPER.STAGE_DESCRIPTIONS['plate_inference_ms'] == 'MT: inferencja modelu tablic'
    assert HELPER.STAGE_DESCRIPTIONS['character_inference_ms'] == 'MZ: inferencja modelu znaków'
    assert HELPER.STAGE_DESCRIPTIONS['total_ms'] == 'ALPR: cały proces'
    bundle, report = _data()
    context = HELPER.session_time_context(bundle, report)
    text = HELPER.point_caption(bundle.trace_rows[0], .737, 'MT: inferencja modelu tablic', 673.805769, context)
    assert 'Klatka 384' in text and '+0:00,737' in text and '673.8 ms' in text
    assert 'zapis ' in text
    assert 'kamera' not in text.lower()  # not a sensor exposure timestamp


def test_chart_has_independent_axis_and_link_controls_and_a_session_caption():
    text = BROWSER.read_text(encoding='utf-8')
    ast.parse(text)
    assert 'text=\'Czas sesji\'' in text
    assert 'text=\'ID klatki\'' in text
    assert 'text="Łącz punkty sąsiednich klatek"' in text
    assert 'session_elapsed_seconds(bundle.trace_rows, context)' in text
    assert 'point_caption(bundle.trace_rows[index]' in text
    assert 'source_min_x, source_max_x = 0.0' in text
    assert 'axis_caption = (\'Oś X: czas od startu sesji' in text
    # Old trained model inference measurements are not retitled as full model stages.
    assert "'plate_inference_ms': 'MT: inferencja modelu tablic'" in TIMES.read_text('utf-8')


class _Canvas:
    def __init__(self):
        self.ops = []

    def winfo_width(self):
        return 1200

    def winfo_height(self):
        return 350

    def delete(self, *args, **kwargs):
        self.ops = []

    def create_rectangle(self, *args, **kwargs):
        self.ops.append(('rect', args, kwargs))

    def create_line(self, *args, **kwargs):
        self.ops.append(('line', args, kwargs))

    def create_text(self, *args, **kwargs):
        self.ops.append(('text', args, kwargs))


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Radio:
    def __init__(self):
        self.state = None

    def configure(self, **kwargs):
        self.state = kwargs.get('state')


def _actual_plot_stub(*, connected=True, zoom=None, axis='time', missing=False):
    gui_source = ast.parse(BROWSER.read_text('utf-8'))
    cls = next(node for node in gui_source.body if isinstance(node, ast.ClassDef) and node.name == 'MobileReportBrowser')
    func = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_draw_latency_chart')
    src = ast.unparse(ast.Module(body=[func], type_ignores=[]))
    styles = {
        'total_ms': SimpleNamespace(label='Cały proces', color='#555555', line_width=2),
        'plate_inference_ms': SimpleNamespace(label='MT', color='#1166aa', line_width=2),
        'character_inference_ms': SimpleNamespace(label='MZ', color='#dd4411', line_width=2),
    }
    bundle, report = _data()
    if missing:
        for r in bundle.trace_rows:
            r.pop('elapsed_ms', None)
            r.pop('timestamp_ms', None)
    names = tuple(styles)
    def timeline(rows, fields):
        return {'x': tuple(float(r['frame_id']) for r in rows),
                'axis_label': 'ID klatki',
                'stages': {field: tuple(float(r[field]) for r in rows) for field in fields}}
    def continuous(vals, xs, require_consecutive=False):
        return tuple((i-1, i) for i in range(1, len(vals)) if vals[i] is not None and vals[i-1] is not None
                     and (not require_consecutive or abs(xs[i]-xs[i-1]-1)<1e-9))
    def viewport_counts(xs, lo, hi):
        left=sum(x < lo for x in xs)
        mid=sum(lo <= x <= hi for x in xs)
        return left, mid, len(xs)-left-mid
    def sparse_positions(xs, even_spacing=False):
        return tuple(float(i if even_spacing else x) for i, x in enumerate(xs))
    palette = SimpleNamespace(W='w', E='e', CENTER='center')
    ns = {
        'tk': palette, 'logger': SimpleNamespace(exception=lambda *a: None),
        'LATENCY_SERIES_STYLES': styles,
        'timeline_stages': timeline, 'stage_enabled': lambda *_: True,
        'has_sparse_frame_gap': lambda xs,label: True,
        'display_positions': sparse_positions,
        'session_time_context': HELPER.session_time_context,
        'session_elapsed_seconds': HELPER.session_elapsed_seconds,
        'session_caption': HELPER.session_caption,
        'point_caption': HELPER.point_caption,
        'time_ticks': HELPER.time_ticks,
        'STAGE_DESCRIPTIONS': HELPER.STAGE_DESCRIPTIONS,
        'STAGE_SHORT': HELPER.STAGE_SHORT,
        'continuous_segments': continuous,
        'blend_hex_colors': lambda col,*rest: col,
        'clamp_x_view': lambda whole, sub: sub,
        'viewport_counts': viewport_counts,
        'source_id_ticks': lambda xs,lo,hi,w: tuple((i, str(int(xs[i]))) for i in range(len(xs)) if lo<=i<=hi),
        'axis_ticks': lambda *args: ((0,'0'),),
        'pixel_column_extrema': lambda *args: (),
        'visible_xy': lambda *args: ((),()),
        'draw_chart_marker': lambda *args, **kwargs: None,
    }
    exec(src, ns)
    canvas = _Canvas()
    view = SimpleNamespace(
        latency_canvas=canvas, current_bundle=bundle, current_report=report,
        _latency_connect_neighbors_var=_Var(connected),
        _latency_axis_mode_var=_Var(axis),
        _latency_context_var=_Var(''),
        _latency_axis_footer_var=_Var(''),
        _latency_time_axis_radio=_Radio(),
        panel='#fefefe', border='#888888', fg='#333333', muted='#777777',
        _latency_full_x=None, _latency_x_view=zoom, _latency_pan_origin=None,
        _latency_effective_mode=None,
    )
    ns['_draw_latency_chart'](view)
    return view, ns['_draw_latency_chart']


def test_true_time_axis_spans_full_session_and_uses_true_trace_offsets():
    v, draw = _actual_plot_stub()
    assert v._latency_full_x == (0, 60.005)
    assert len(v._latency_hit_points) == 24  # 3 measurements for each of 8 frames
    positions = [item[0] for item in v._latency_hit_points if item[3] == 'total_ms']
    assert positions[0] < positions[1] < positions[-1]
    last_px = positions[-1]
    assert last_px < 1200-18-30  # last measured frame precedes end of the 60 s session
    assert 'Czas trwania sesji: 1:00,005' in v._latency_hover_baseline
    assert 'Oś X: czas od startu sesji' in v._latency_axis_footer_var.get()
    labels = [row[2]['text'] for row in v.latency_canvas.ops if row[0] == 'text']
    assert any('MT: inferencja modelu tablic' in label for label in labels)
    assert any('MZ: inferencja modelu znaków' in label for label in labels)


def test_changing_links_does_not_change_time_axis_zoom_or_sample_positions():
    v, draw = _actual_plot_stub(connected=True, zoom=(10,45))
    original_positions = list(v._latency_hit_points)
    original_zoom = v._latency_x_view
    original_full = v._latency_full_x
    line_count_before = sum(row[0] == 'line' for row in v.latency_canvas.ops)
    v._latency_connect_neighbors_var.set(False)
    draw(v)
    assert v._latency_x_view == original_zoom
    assert v._latency_full_x == original_full
    assert v._latency_hit_points == original_positions
    line_count_after = sum(row[0] == 'line' for row in v.latency_canvas.ops)
    assert line_count_before > line_count_after


def test_old_archive_falls_back_without_fake_time_axis():
    v, draw = _actual_plot_stub(missing=True)
    assert v._latency_axis_label == 'ID klatki'
    assert v._latency_time_axis_radio.state == 'disabled'
    assert v._latency_full_x == (0, 7)
    assert v._latency_time_seconds is None
    assert 'Oś X: ID klatki' in v._latency_axis_footer_var.get()


def test_hover_describes_actual_selected_frame_not_guessed_intermediate():
    browser_module = ast.parse(BROWSER.read_text('utf-8'))
    cls = next(n for n in browser_module.body if isinstance(n, ast.ClassDef) and n.name == 'MobileReportBrowser')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_on_latency_point_hover')
    source = ast.unparse(ast.Module(body=[method], type_ignores=[]))
    ns = {'point_caption': HELPER.point_caption, 'STAGE_DESCRIPTIONS': HELPER.STAGE_DESCRIPTIONS}
    exec(source, ns)
    v, _ = _actual_plot_stub()
    v._clear_latency_point_hover = lambda event=None: v._latency_context_var.set(v._latency_hover_baseline)
    x, y, index, key, value = v._latency_hit_points[3]
    ns['_on_latency_point_hover'](v, SimpleNamespace(x=x, y=y))
    text = v._latency_context_var.get()
    assert f"Klatka {v.current_bundle.trace_rows[index]['frame_id']}" in text
    assert '+' + HELPER.fmt_elapsed(v._latency_time_seconds[index], milliseconds=True) in text
    assert 'ms' in text
    ns['_on_latency_point_hover'](v, SimpleNamespace(x=10, y=300))
    assert v._latency_context_var.get() == v._latency_hover_baseline


def test_tk_time_context_and_footer_survive_reducing_plot_pane():
    import tkinter as tk
    from tkinter import ttk
    import pytest
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Brak serwera graficznego Tk')
    try:
        root.geometry('790x610+0+0')
        pane = ttk.PanedWindow(root, orient=tk.VERTICAL)
        pane.pack(fill=tk.BOTH, expand=True)
        plot = tk.Frame(pane)
        table = tk.Frame(pane)
        pane.add(plot, weight=3)
        pane.add(table, weight=1)
        axis_bar = tk.Frame(plot)
        axis_bar.pack(fill=tk.X)
        tk.Checkbutton(axis_bar, text='Łącz punkty sąsiednich klatek').pack(side=tk.LEFT)
        tk.Radiobutton(axis_bar, text='Czas sesji').pack(side=tk.RIGHT)
        tk.Radiobutton(axis_bar, text='ID klatki').pack(side=tk.RIGHT)
        context = tk.Label(plot, text='Czas trwania sesji: 1:00,005 · start 19.10 01:19:09 · koniec 19.10 01:20:09', height=1, width=1)
        context.pack(fill=tk.X, padx=(50,18), pady=(0,1))
        canvas = tk.Canvas(plot, height=320)
        canvas.pack(fill=tk.BOTH, expand=True)
        footer = tk.Label(plot, text='Oś X: czas od startu sesji · chwila zapisu pomiaru, nie moment ekspozycji kamery', height=1)
        footer.pack(side=tk.BOTTOM, fill=tk.X, before=canvas, padx=(50,18), pady=(3,6))
        root.deiconify()
        root.update()
        for pane_height in (370, 265, 165, 140):
            pane.sashpos(0, pane_height)
            root.update_idletasks()
            root.update()
            assert context.winfo_ismapped() and footer.winfo_ismapped()
            assert footer.winfo_height() >= footer.winfo_reqheight()
            assert canvas.winfo_y() + canvas.winfo_height() <= footer.winfo_y()
            assert canvas.winfo_height() > 5
    finally:
        root.destroy()
