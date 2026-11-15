"""Readable, accessible visual styles for measured ALPR latency series.

Pure presentation helpers. They never transform timestamps, frame IDs, data,
measurements, denominators, or other research evidence.
"""
from __future__ import annotations

from typing import NamedTuple


class ChartStyle(NamedTuple):
    label: str
    color: str
    marker: str
    line_width: float


# Fixed scientific palette independent of theme accent, success and warning.
# Distinguishable by both hue and point shape, including monochrome prints.
LATENCY_SERIES_STYLES: dict[str, ChartStyle] = {
    "total_ms": ChartStyle("Cały proces", "#374151", "diamond", 2.6),
    "plate_inference_ms": ChartStyle("MT", "#1565C0", "circle", 2.0),
    "character_inference_ms": ChartStyle("MZ", "#C2410C", "square", 2.0),
    "vehicle_inference_ms": ChartStyle("MP", "#7C3AED", "triangle", 2.0),
}


def draw_chart_marker(canvas, x: float, y: float, style: ChartStyle, *,
                      radius: float = 3.3, background: str | None = None) -> None:
    """Draw a consistent non-color-only marker on a Tk-like canvas.

    Can be used by the chart and its legend. The caller supplies measured
    coordinates; this function deliberately never estimates missing values.
    """
    r = max(1.0, float(radius))
    stroke = background if background is not None else style.color
    common = {"fill": style.color, "outline": stroke, "width": 0.75}
    if style.marker == "circle":
        canvas.create_oval(x - r, y - r, x + r, y + r, **common)
    elif style.marker == "square":
        canvas.create_rectangle(x - r, y - r, x + r, y + r, **common)
    elif style.marker == "diamond":
        canvas.create_polygon(x, y-r, x+r, y, x, y+r, x-r, y, **common)
    elif style.marker == "triangle":
        canvas.create_polygon(x, y-r, x+r, y+r, x-r, y+r, **common)
    else:
        raise ValueError("UNKNOWN_CHART_MARKER: " + style.marker)
