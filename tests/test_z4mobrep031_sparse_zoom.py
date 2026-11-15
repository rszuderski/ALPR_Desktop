"""Sparse X-axis preserves every observation, regardless of connector visibility.

Z4MOBREP033: changing the checkbox must never change the operator-selected
viewport, plot geometry, numeric value, nor the position of a measured point.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

MODULE = Path(__file__).resolve().parents[1] / 'auto_annotation_tool' / 'gui' / 'mobile_report_sparse_axis.py'
BROWSER = MODULE.parent / 'z4_mobile_report_browser.py'


def _load_helper():
    spec = importlib.util.spec_from_file_location('mobile_report_sparse_axis_fixture', MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_helper()
FRAME_IDS = [384, 386, 387, 388, 389, 390, 391, 585]


def test_sparse_gap_detected_only_for_real_frame_id_series():
    assert MOD.has_sparse_frame_gap(FRAME_IDS, 'ID klatki')
    assert not MOD.has_sparse_frame_gap(FRAME_IDS, 'Czas (ms)')
    assert not MOD.has_sparse_frame_gap([384, 385, 386, 387, 388], 'ID klatki')
    assert not MOD.has_sparse_frame_gap([384, 385, 385, 585], 'ID klatki')
    assert not MOD.has_sparse_frame_gap([384, 385, float('nan'), 585], 'ID klatki')
    assert not MOD.has_sparse_frame_gap([585, 384, 391, 390], 'ID klatki')
    assert not MOD.has_sparse_frame_gap([1, 2, 500], 'ID klatki')


def test_display_positions_do_not_modify_measured_ids():
    source = list(FRAME_IDS)
    assert MOD.display_positions(source, even_spacing=True) == tuple(float(i) for i in range(8))
    assert MOD.display_positions(source, even_spacing=False) == tuple(float(i) for i in source)
    assert source == FRAME_IDS


def test_ticks_are_real_frame_ids_and_never_fabricated_labels():
    ticks = MOD.source_id_ticks(FRAME_IDS, 0, 7, 1200)
    assert [name for _, name in ticks] == list(map(str, FRAME_IDS))
    zoomed = MOD.source_id_ticks(FRAME_IDS, 1.5, 5.5, 800)
    assert [name for _, name in zoomed] == ['387', '388', '389', '390']
    assert '392' not in [name for _, name in ticks]
    assert MOD.source_id_ticks(FRAME_IDS, 6.6, 7, 200) == ((7.0, '585'),)
    narrow = MOD.source_id_ticks(FRAME_IDS, 0, 7, 185)
    assert narrow[0] == (0.0, '384') and narrow[-1] == (7.0, '585')


def test_viewport_counts_distinguish_visible_from_total():
    assert MOD.viewport_counts(FRAME_IDS, 385, 394) == (1, 6, 1)
    assert MOD.viewport_counts(tuple(range(8)), 0, 7) == (0, 8, 0)
    assert MOD.viewport_counts(tuple(range(8)), 2.5, 5.5) == (3, 3, 2)


def test_browser_checkbox_controls_only_segments():
    src = BROWSER.read_text('utf-8')
    assert 'text="Łącz punkty sąsiednich klatek"' in src
    assert 'variable=self._latency_connect_neighbors_var' in src
    assert 'command=self._on_latency_connections_change' in src
    assert 'self._latency_even_spacing_var' not in src
    assert 'use_even_spacing = has_sparse_frame_gap(xs, timeline["axis_label"])' in src
    assert 'if connect_consecutive:' in src
    assert 'require_consecutive=True' in src
    assert 'Oś X: ID klatki z pomiarem czasu' in src
    assert 'self._latency_axis_footer_label.pack' in src
    assert 'measured_visible = sum(' in src
    assert 'left_hidden, visible_count, right_hidden' in src
    ast.parse(src)


class _FakeCanvas:
    def __init__(self):
        self.shapes = []
        self.width = 1200
        self.height = 360

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height

    def delete(self, *args, **kwargs):
        self.shapes.clear()

    def create_rectangle(self, *args, **kwargs):
        self.shapes.append(('rect', args, kwargs))

    def create_line(self, *args, **kwargs):
        self.shapes.append(('line', args, kwargs))

    def create_text(self, *args, **kwargs):
        self.shapes.append(('text', args, kwargs))


class _Flag:
    def __init__(self, value: bool):
        self.value = value

    def get(self):
        return self.value


def _methods():
    tree = ast.parse(BROWSER.read_text('utf-8'))
    clazz = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MobileReportBrowser')
    return {n.name: n for n in clazz.body if isinstance(n, ast.FunctionDef)}


def _render_case(connect: bool, *, zoom=None, frame_ids=None, axis_label='ID klatki'):
    """Run actual production renderer using fake Canvas; not a mock of its geometry."""
    funcs = _methods()
    funcs = [funcs['_draw_latency_chart'], funcs['_on_latency_connections_change']]
    source = ast.unparse(ast.Module(body=funcs, type_ignores=[]))
    root = Path(__file__).resolve().parents[1] / 'auto_annotation_tool' / 'gui'
    axis_spec = importlib.util.spec_from_file_location('axis_tick_fixture', root/'mobile_report_plot_axis.py')
    axis_module = importlib.util.module_from_spec(axis_spec)
    axis_spec.loader.exec_module(axis_module)
    zoom_spec = importlib.util.spec_from_file_location('xzoom_fixture', root/'mobile_report_x_zoom.py')
    zoom_module = importlib.util.module_from_spec(zoom_spec)
    zoom_spec.loader.exec_module(zoom_module)
    time_spec = importlib.util.spec_from_file_location('session_time_fixture', root/'mobile_report_session_timeline.py')
    time_module = importlib.util.module_from_spec(time_spec)
    time_spec.loader.exec_module(time_module)

    ids = tuple(FRAME_IDS if frame_ids is None else frame_ids)
    styles = {
        'total_ms': SimpleNamespace(label='Cały proces', color='#303030', line_width=2),
        'plate_inference_ms': SimpleNamespace(label='MT', color='#0066DD', line_width=2),
        'character_inference_ms': SimpleNamespace(label='MZ', color='#DD6600', line_width=2),
    }
    stages = {key: [float(100*(i+1)) for i in range(len(ids))] for key in styles}
    timeline = {'x': ids, 'axis_label': axis_label, 'stages': stages}
    canvas = _FakeCanvas()
    markers = []

    def draw_marker(canvas_, x, y, style, **kwargs):
        markers.append((style.label, x, y))

    def continuous_segments(values, source_x, *, require_consecutive):
        return tuple((i, i+1) for i in range(len(values)-1) if values[i] is not None and values[i+1] is not None
                     and (not require_consecutive or source_x[i+1] - source_x[i] == 1))

    ns = {
        'tk': SimpleNamespace(W='w', E='e', CENTER='center'),
        'blend_hex_colors': lambda c, *args: c,
        'LATENCY_SERIES_STYLES': styles,
        'draw_chart_marker': draw_marker,
        'timeline_stages': lambda rows, names: timeline,
        'stage_enabled': lambda *_: True,
        'axis_ticks': axis_module.axis_ticks,
        'pixel_column_extrema': axis_module.pixel_column_extrema,
        'clamp_x_view': zoom_module.clamp_x_view,
        'visible_xy': zoom_module.visible_xy,
        'continuous_segments': continuous_segments,
        'has_sparse_frame_gap': MOD.has_sparse_frame_gap,
        'display_positions': MOD.display_positions,
        'source_id_ticks': MOD.source_id_ticks,
        'viewport_counts': MOD.viewport_counts,
        'session_time_context': time_module.session_time_context,
        'session_elapsed_seconds': time_module.session_elapsed_seconds,
        'session_caption': time_module.session_caption,
        'time_ticks': time_module.time_ticks,
        'point_caption': time_module.point_caption,
        'STAGE_DESCRIPTIONS': time_module.STAGE_DESCRIPTIONS,
        'STAGE_SHORT': time_module.STAGE_SHORT,
        'logger': SimpleNamespace(exception=lambda *args: None),
    }
    exec(source, ns)
    footer_values = []
    fake = SimpleNamespace(
        latency_canvas=canvas, current_bundle=SimpleNamespace(trace_rows=[{} for _ in ids]),
        current_report=SimpleNamespace(), _latency_connect_neighbors_var=_Flag(connect),
        panel='#DDDDDD', border='#888888', fg='#000000', muted='#333333',
        _latency_x_view=zoom, _latency_full_x=None,
        _latency_axis_footer_var=SimpleNamespace(set=footer_values.append),
        _latency_pan_origin=(48.0, zoom),
    )
    fake._draw_latency_chart = lambda: ns['_draw_latency_chart'](fake)
    fake._on_latency_connections_change = lambda: ns['_on_latency_connections_change'](fake)
    fake._draw_latency_chart()
    fake.footer_text = footer_values[-1] if footer_values else ''
    fake._markers = markers
    fake._footer_values = footer_values
    return canvas.shapes, markers, fake


def _colored_connectors(shapes):
    return [s for s in shapes if s[0] == 'line' and s[2].get('fill') in ('#303030', '#0066DD', '#DD6600')
            and len(s[1]) == 4 and s[2].get('smooth') is False]


def test_actual_draw_shows_all_8_points_at_same_positions_with_or_without_links():
    lines_on, marks_on, state_on = _render_case(True)
    lines_off, marks_off, state_off = _render_case(False)
    assert state_on._latency_full_x == state_off._latency_full_x == (0.0, 7.0)
    assert state_on._latency_x_view == state_off._latency_x_view is None
    assert state_on._latency_display_even_spacing is state_off._latency_display_even_spacing is True
    assert marks_on == marks_off
    assert len(_colored_connectors(lines_on)) == 15  # 5 consecutive links per 3 series
    assert not _colored_connectors(lines_off)
    assert sum(s[0] == 'line' and s[2].get('fill') in ('#303030', '#0066DD', '#DD6600')
               and s[2].get('smooth') is None for s in lines_on) == 3  # legend samples
    assert not any(s[0] == 'line' and s[2].get('fill') in ('#303030', '#0066DD', '#DD6600')
                   for s in lines_off)
    labels = [s[2].get('text', '') for s in lines_on if s[0] == 'text']
    assert '585' in labels
    assert state_on.footer_text == state_off.footer_text
    assert 'punkty w równych odstępach' in state_on.footer_text
    assert 'nie jest to oś czasu' in state_on.footer_text


def test_zoom_window_and_point_geometry_are_immutable_when_checkbox_changes():
    shapes, markers, fake = _render_case(True, zoom=(2.2, 5.5))
    axis_before = fake._latency_full_x
    window_before = fake._latency_x_view
    pan_before = fake._latency_pan_origin
    points_before = tuple(markers)
    other_shapes = tuple(s for s in shapes if not (s[0] == 'line' and
        s[2].get('fill') in ('#303030', '#0066DD', '#DD6600')))
    assert _colored_connectors(shapes)
    fake._latency_connect_neighbors_var.value = False
    markers.clear()
    fake._on_latency_connections_change()
    assert fake._latency_full_x == axis_before
    assert fake._latency_x_view == window_before
    assert fake._latency_pan_origin == pan_before
    assert tuple(markers) == points_before
    assert not _colored_connectors(shapes)
    assert tuple(s for s in shapes if not (s[0] == 'line' and
        s[2].get('fill') in ('#303030', '#0066DD', '#DD6600'))) == other_shapes
    # Re-enable: identical view, exact same markers, original number of links.
    fake._latency_connect_neighbors_var.value = True
    markers.clear()
    fake._on_latency_connections_change()
    assert fake._latency_full_x == axis_before
    assert fake._latency_x_view == window_before
    assert fake._latency_pan_origin == pan_before
    assert tuple(markers) == points_before
    assert _colored_connectors(shapes)


def test_actual_draw_zoom_counts_original_measured_points_with_links_on_or_off():
    on_shapes, on_marks, on = _render_case(True, zoom=(2, 5))
    off_shapes, off_marks, off = _render_case(False, zoom=(2, 5))
    assert on._latency_x_view == off._latency_x_view == (2, 5)
    assert on_marks == off_marks
    assert '4/8' in on.footer_text and '4/8' in off.footer_text
    assert '◀2' in on.footer_text and '▶2' in on.footer_text
    assert all(len([x for x in markers if x[0] == series]) == 5
               for markers in (on_marks, off_marks) for series in ('Cały proces', 'MT', 'MZ'))
    assert len(_colored_connectors(on_shapes)) == 9  # 4 consecutive frames => 3 links, 3 series
    assert len(_colored_connectors(off_shapes)) == 0


def test_dense_sequential_ids_keep_numeric_spacing_in_both_toggle_states():
    ids = list(range(100,108))
    _, ma, a = _render_case(True, frame_ids=ids)
    _, mb, b = _render_case(False, frame_ids=ids)
    assert a._latency_full_x == b._latency_full_x == (100,107)
    assert a._latency_display_even_spacing is b._latency_display_even_spacing is False
    assert ma == mb
    assert 'odstępy według numerów klatek' in a.footer_text


def test_without_frame_ids_checkbox_does_not_imply_artificial_continuity():
    shapes, markers, state = _render_case(True, axis_label='Czas (ms)')
    assert len(markers) == 27
    assert not _colored_connectors(shapes)
    assert state._latency_display_even_spacing is False
