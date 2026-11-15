"""Readable, source-faithful horizontal axes for mobile ALPR latency plots.

All tick labels belong to the source coordinate system. Dense-pixel summaries
retain each display column's extrema without modifying or fabricating measurements.
"""
from __future__ import annotations

import math
from typing import Sequence


def _nice_step(required: float) -> float:
    """Return a human-readable grid spacing of 1/2/5 * power of ten."""
    if not math.isfinite(required) or required <= 0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(required))
    for factor in (1.0, 2.0, 5.0, 10.0):
        step = factor * magnitude
        if step >= required - 1e-12 * max(1.0, required):
            return step
    return 10.0 * magnitude


def axis_ticks(xs: Sequence[float], plot_width: int, axis_label: str) -> tuple[tuple[float, str], ...]:
    """Give legible numeric X ticks for a plot that grows/shrinks with the UI.

    A few real frame IDs are shown individually when there is space. Longer
    timelines receive a 1/2/5-based scale plus actual first and last values.
    Intermediate ticks label coordinates, not fabricated measured frames.
    """
    if not xs:
        return ()
    numeric = [float(x) for x in xs]
    if not all(math.isfinite(x) for x in numeric):
        return ()
    start, end = min(numeric), max(numeric)
    integer_axis = axis_label == "ID klatki" or axis_label.startswith("Numer pomiaru")

    def render(value: float) -> str:
        if integer_axis or float(value).is_integer():
            return str(int(round(value)))
        return f"{value:.3f}".rstrip("0").rstrip(".")

    if start == end:
        return ((start, render(start)),)

    width = max(1, int(plot_width))
    max_count = max(2, min(12, width // 80 + 1))
    # Small series: show *exact measured frame IDs* if labels are not crowded.
    if len(numeric) <= max_count and all(
        (right - left) / (end - start) * width >= 62
        for left, right in zip(numeric, numeric[1:])
    ):
        return tuple((value, render(value)) for value in numeric)

    step = _nice_step((end - start) / max(1, max_count - 1))
    first_multiple = math.ceil((start - 1e-10 * step) / step)
    last_multiple = math.floor((end + 1e-10 * step) / step)
    # Prevent pathological work for very large coordinate ranges.
    available = max(0, last_multiple - first_multiple + 1)
    if available > 10000:
        return ((start, render(start)), (end, render(end)))
    candidates = [start] + [i * step for i in range(first_multiple, last_multiple + 1)] + [end]
    unique: list[float] = []
    for value in candidates:
        if not unique or abs(value - unique[-1]) > 1e-9 * max(1, abs(value)):
            unique.append(value)
    min_spacing = 56.0
    kept = [unique[0]]
    for value in unique[1:-1]:
        if (value - kept[-1]) / (end - start) * width >= min_spacing and \
                (end - value) / (end - start) * width >= min_spacing:
            kept.append(value)
    if (end - kept[-1]) / (end - start) * width >= min_spacing:
        kept.append(end)
    else:
        # Endpoint takes priority over a nearby intermediate tick.
        kept[-1] = end
    return tuple((value, render(value)) for value in kept)


def pixel_column_extrema(xs: Sequence[float], values: Sequence[float | None],
                         min_x: float, max_x: float, plot_width: int) -> tuple[tuple[int, float, float, int], ...]:
    """Aggregate *display pixels only*, preserving every per-column min/max.

    Does not return averaged or interpolated results. Each item is
    (x_pixel_offset, min_value, max_value, count_of_measured_points).
    """
    if len(xs) != len(values):
        raise ValueError("X_AND_STAGE_LENGTH_MISMATCH")
    width = max(1, int(plot_width))
    span = max_x - min_x
    bucket: dict[int, tuple[float, float, int]] = {}
    for source_x, source_y in zip(xs, values):
        if source_y is None:
            continue
        x, y = float(source_x), float(source_y)
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        offset = (x - min_x) / span if span > 0 else 0.5
        column = max(0, min(width, int(round(offset * width))))
        current = bucket.get(column)
        if current is None:
            bucket[column] = (y, y, 1)
        else:
            lo, hi, count = current
            bucket[column] = (min(lo, y), max(hi, y), count + 1)
    return tuple((column, *bucket[column]) for column in sorted(bucket))
