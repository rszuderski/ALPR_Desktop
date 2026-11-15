"""Frame-axis and dense-view contract: readable at 8/100/1000 frames.

Only view transformations are permitted. Raw samples and sources are immutable.
"""
from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'auto_annotation_tool/gui/mobile_report_plot_axis.py'
BROWSER = ROOT / 'auto_annotation_tool/gui/z4_mobile_report_browser.py'
spec = importlib.util.spec_from_file_location('z4_plot_axis_fix02', HELPER)
axis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(axis)


def test_eight_real_frame_ids_all_visible_on_wide_plot():
    frames = (1, 2, 3, 5, 7, 8, 11, 14)
    assert axis.axis_ticks(frames, 1000, 'ID klatki') == tuple((float(f), str(f)) for f in frames)


def test_hundred_frames_scale_with_numeric_ruler_not_hundred_labels():
    frames = tuple(range(1, 101))
    marks = axis.axis_ticks(frames, 900, 'ID klatki')
    assert 4 <= len(marks) <= 12
    assert marks[0] == (1.0, '1') and marks[-1] == (100.0, '100')
    assert all(x < y for (x, _), (y, _) in zip(marks, marks[1:]))
    assert all((right-left)/99*900 >= 54 for (left,_),(right,_) in zip(marks, marks[1:]))


def test_plot_width_adapts_tick_count_without_losing_source_range():
    frames = list(range(1, 101))
    wide = axis.axis_ticks(frames, 1100, 'ID klatki')
    narrow = axis.axis_ticks(frames, 240, 'ID klatki')
    assert len(narrow) < len(wide)
    assert narrow[0][0] == wide[0][0] == 1
    assert narrow[-1][0] == wide[-1][0] == 100


def test_axis_displays_true_single_frame_instead_of_arbitrary_zero():
    assert axis.axis_ticks((37,), 650, 'ID klatki') == ((37.0, '37'),)


def test_time_fallback_floats_remain_uninvented_coordinates():
    measured = (0.0, 0.2, 1.3, 1.7)
    marks = axis.axis_ticks(measured, 1000, 'Czas od pierwszej klatki [s]')
    assert [v for v, _ in marks] == list(measured)
    assert marks[-1][1] == '1.7'


def test_dense_frames_keep_spike_and_zero_as_real_extrema():
    xs = tuple(range(1, 1001))
    y = [10.0] * 1000
    y[450] = 999.0
    y[451] = 0.0
    original = tuple(y)
    buckets = axis.pixel_column_extrema(xs, y, 1, 1000, 250)
    assert len(buckets) <= 251
    assert max(hi for _, _, hi, _ in buckets) == 999.0
    assert min(lo for _, lo, _, _ in buckets) == 0.0
    assert sum(count for _, _, _, count in buckets) == 1000
    assert tuple(y) == original


def test_missing_mz_is_not_converted_to_zero():
    buckets = axis.pixel_column_extrema((1, 2, 3, 5, 7), (None, 10.0, None, 0.0, None), 1, 7, 100)
    assert sum(count for _, _, _, count in buckets) == 2
    assert [v for _, v, _, _ in buckets] == [10, 0]


def test_one_pixel_column_returns_range_not_mean_or_fake_points():
    bins = axis.pixel_column_extrema((1, 1.01, 1.02), (12.0, 100.0, 5.0), 1, 2, 1)
    assert bins == ((0, 5.0, 100.0, 3),)


def test_bogus_nan_samples_are_skipped_while_real_zero_is_kept():
    bins = axis.pixel_column_extrema((1, 2, 3, 4), (0, float('nan'), None, 10), 1, 4, 10)
    assert sum(v[3] for v in bins) == 2
    assert min(v[1] for v in bins) == 0.0


def test_corrupted_length_rejected_not_silently_zipped():
    with pytest.raises(ValueError, match='X_AND_STAGE_LENGTH_MISMATCH'):
        axis.pixel_column_extrema((1,2,3),(1,2),1,3,250)


def test_browser_uses_original_frame_numbers_and_extrema_without_stride():
    code = BROWSER.read_text('utf-8')
    ast.parse(code)
    assert 'axis_ticks(xs, plot_w, timeline["axis_label"])' in code
    assert 'pixel_column_extrema(xs, values, min_x, max_x, plot_w)' in code
    draw = code.split('    def _draw_latency_chart(self) -> None:',1)[1].split('\ndef open_mobile_report_browser',1)[0]
    assert 'stride =' not in draw
    assert 'smooth=False' in draw
    assert 'continuous_segments(values, xs' in draw
    assert 'dense = len(xs) > 350' in draw


def test_small_and_large_axis_labels_do_not_explode_cardinality():
    assert len(axis.axis_ticks(range(1, 100_001), 800, 'ID klatki')) <= 12
    assert len(axis.axis_ticks(range(1, 100_001), 180, 'ID klatki')) <= 3
