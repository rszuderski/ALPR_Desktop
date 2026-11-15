"""Display-only spacing for sparse mobile latency traces.

The actual measured X coordinates remain untouched. This module may display
samples at equally spaced *positions* while labeling each position with its
real source frame ID. This categorical view is explicitly labeled so that the
spacing cannot be mistaken for elapsed time or numeric frame distance.
"""
from __future__ import annotations

from math import ceil, isfinite
from statistics import median
from typing import Sequence


def has_sparse_frame_gap(xs: Sequence[float], axis_label: str, *, max_samples: int = 160) -> bool:
    """Decide whether a sparse, strictly increasing frame-ID trace merits categorical spacing.

    Never resample or reorder; return False for ambiguous, duplicated, nonfinite,
    timestamp, very short and dense sequences.
    """
    if axis_label != "ID klatki" or not 4 <= len(xs) <= max_samples:
        return False
    try:
        x = tuple(float(value) for value in xs)
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(isfinite(v) for v in x):
        return False
    gaps = tuple(b - a for a, b in zip(x, x[1:]))
    if not gaps or any(g <= 0 for g in gaps):
        return False
    central_gap = median(gaps)
    return max(gaps) >= max(8.0 * central_gap, 8.0)


def display_positions(xs: Sequence[float], *, even_spacing: bool) -> tuple[float, ...]:
    """Return display X positions only; never modify or interpolate observations."""
    return (tuple(float(i) for i in range(len(xs))) if even_spacing
            else tuple(float(x) for x in xs))


def source_id_ticks(xs: Sequence[float], low: float, high: float, plot_width: int,
                    *, min_pixels: int = 78) -> tuple[tuple[float, str], ...]:
    """Label categorical ticks with REAL frame IDs of actual, visible samples.

    Ticks are source sample positions, not fabricated intermediate frame IDs.
    Preserve visible endpoints when label spacing allows.
    """
    if not xs or high <= low:
        return ()
    visible = [i for i in range(len(xs)) if low <= float(i) <= high]
    if not visible:
        return ()
    if len(visible) == 1:
        i = visible[0]
        return ((float(i), _frame_label(xs[i])),)
    width = max(1, int(plot_width))
    max_ticks = max(2, (width // max(1, min_pixels)) + 1)
    if len(visible) <= max_ticks:
        indices = visible
    else:
        stride = max(1, ceil((len(visible) - 1) / (max_ticks - 1)))
        indices = visible[::stride]
        if indices[-1] != visible[-1]:
            indices.append(visible[-1])
        # Near-edge label collisions: keep the endpoint and discard nearby tick.
        while len(indices) > 2 and (indices[-1] - indices[-2]) / (high-low) * width < min_pixels:
            indices.pop(-2)
    return tuple((float(i), _frame_label(xs[i])) for i in indices)


def _frame_label(raw: float) -> str:
    value = float(raw)
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def viewport_counts(xs: Sequence[float], low: float, high: float) -> tuple[int, int, int]:
    """(left of view, in view, right of view), with no data modifications."""
    left = sum(float(x) < low for x in xs)
    visible = sum(low <= float(x) <= high for x in xs)
    right = len(xs) - left - visible
    return left, visible, right
