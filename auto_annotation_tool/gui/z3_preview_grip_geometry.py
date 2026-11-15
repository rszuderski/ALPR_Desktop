from __future__ import annotations

import math

CORNER_HANDLE_ORDER = ("nw", "ne", "sw", "se")
EDGE_HANDLE_ORDER = ("n", "e", "s", "w")


def get_edge_handle_rects(x1, y1, x2, y2, radius):
    """Canvas-space rectangles outside each side, including their stroke."""
    radius = max(0.0, float(radius))
    thickness = max(4.0, min(7.0, radius * 0.6))
    horizontal_length = min(max(2.0, (x2 - x1) * 0.5), max(6.0, min(16.0, radius * 1.4)))
    vertical_length = min(max(2.0, (y2 - y1) * 0.5), max(6.0, min(16.0, radius * 1.4)))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    margin = 1.5  # Half of the widest (hover) outline: paint stays outside.
    return {
        "n": (cx - horizontal_length / 2, y1 - margin - thickness, cx + horizontal_length / 2, y1 - margin),
        "e": (x2 + margin, cy - vertical_length / 2, x2 + margin + thickness, cy + vertical_length / 2),
        "s": (cx - horizontal_length / 2, y2 + margin, cx + horizontal_length / 2, y2 + margin + thickness),
        "w": (x1 - margin - thickness, cy - vertical_length / 2, x1 - margin, cy + vertical_length / 2),
    }


def get_box_handle_point(handle_name, x1, y1, x2, y2):
    if handle_name in CORNER_HANDLE_ORDER:
        return get_box_corner_point(handle_name, x1, y1, x2, y2)
    return {"n": ((x1 + x2) / 2, y1), "e": (x2, (y1 + y2) / 2),
            "s": ((x1 + x2) / 2, y2), "w": (x1, (y1 + y2) / 2)}[handle_name]


def resize_box_edge(bbox, handle_name, x, y, *, min_size=4.0):
    x1, y1, x2, y2 = map(float, bbox)
    if handle_name == "w":
        x1 = min(float(x), x2 - min_size)
    elif handle_name == "e":
        x2 = max(float(x), x1 + min_size)
    elif handle_name == "n":
        y1 = min(float(y), y2 - min_size)
    elif handle_name == "s":
        y2 = max(float(y), y1 + min_size)
    else:
        raise ValueError(f"Nieznany bok: {handle_name!r}")
    return [x1, y1, x2, y2]


def get_corner_handle_centers(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
) -> dict[str, tuple[float, float]]:
    r = max(0.0, float(radius))
    offset = r / math.sqrt(2.0)
    return {
        "nw": (float(x1) - offset, float(y1) - offset),
        "ne": (float(x2) + offset, float(y1) - offset),
        "sw": (float(x1) - offset, float(y2) + offset),
        "se": (float(x2) + offset, float(y2) + offset),
    }


def get_box_corner_point(
    handle_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[float, float]:
    name = str(handle_name or "").strip().lower()
    if name == "nw":
        return float(x1), float(y1)
    if name == "ne":
        return float(x2), float(y1)
    if name == "sw":
        return float(x1), float(y2)
    if name == "se":
        return float(x2), float(y2)
    raise ValueError(f"Nieznany narożnik uchwytu: {handle_name!r}")
