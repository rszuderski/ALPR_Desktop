"""Display-only navigation on a mobile latency plot's original X coordinates.

Viewport calculations neither interpolate, sort nor rewrite telemetry. The full
trace remains the authority for metric aggregation and scientific exports.
"""
from __future__ import annotations

import math
from typing import Sequence


def clamp_x_view(full: tuple[float, float], view: tuple[float, float] | None) -> tuple[float, float] | None:
    """Return a bounded viewport, or None to represent the original full view."""
    lo, hi = map(float, full)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        return None
    if view is None:
        return None
    a, b = map(float, view)
    if not (math.isfinite(a) and math.isfinite(b) and b > a):
        return None
    span = min(hi - lo, b - a)
    if span >= (hi - lo) * (1.0 - 1e-9):
        return None
    start = min(hi - span, max(lo, a))
    return (start, start + span)


def zoom_x_view(full: tuple[float, float], view: tuple[float, float] | None,
                anchor_fraction: float, *, zoom_in: bool,
                integer_axis: bool = False) -> tuple[float, float] | None:
    """Zoom around the actual mouse X coordinate, capped at original limits."""
    lo, hi = map(float, full)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        return None
    old = clamp_x_view(full, view) or (lo, hi)
    anchor = max(0.0, min(1.0, float(anchor_fraction)))
    if not math.isfinite(anchor):
        anchor = 0.5
    factor = 0.80 if zoom_in else 1.25
    minimum = min(hi - lo, max(1.0 if integer_axis else 0.0001, (hi - lo) / 10000))
    span = max(minimum, min(hi - lo, (old[1] - old[0]) * factor))
    if span >= (hi - lo) * (1.0 - 1e-9):
        return None
    pivot = old[0] + (old[1] - old[0]) * anchor
    start = min(hi - span, max(lo, pivot - span * anchor))
    return (start, start + span)


def pan_x_view(full: tuple[float, float], view: tuple[float, float] | None,
               drag_pixels: float, plot_width: float) -> tuple[float, float] | None:
    """Left mouse drag moves the current window, not the measured frame IDs."""
    current = clamp_x_view(full, view)
    if current is None or plot_width <= 0:
        return None
    span = current[1] - current[0]
    offset = -float(drag_pixels) / float(plot_width) * span
    lo, hi = map(float, full)
    start = min(hi - span, max(lo, current[0] + offset))
    return (start, start + span)


def visible_xy(xs: Sequence[float], ys: Sequence[float | None],
               minimum: float, maximum: float) -> tuple[tuple[float, ...], tuple[float | None, ...]]:
    """Select existing measured X positions; no rounding or resampling."""
    if len(xs) != len(ys):
        raise ValueError('X_AND_STAGE_LENGTH_MISMATCH')
    positions = [i for i, x in enumerate(xs) if minimum <= x <= maximum]
    return tuple(xs[i] for i in positions), tuple(ys[i] for i in positions)
