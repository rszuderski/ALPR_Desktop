"""Read-only interpretations of Android ALPR evidence for the Desktop report.

Do not rewrite model predictions, telemetry, historical sessions, or source GT.
MZ timings and other optional stages retain their original frame alignment.
"""
from __future__ import annotations

from datetime import datetime, tzinfo
import math
from typing import Any, Mapping, Sequence


def _nonnegative_finite(value: Any) -> float | None:
    """Distinguish a measured zero from a missing or invalid measurement."""
    if value is None or isinstance(value, bool):
        return None
    try:
        text = str(value).strip()
        result = float(text) if text else float("nan")
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _strictly_increasing_numbers(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[float, ...] | None:
    numbers = tuple(_nonnegative_finite(row.get(key)) for row in rows)
    if any(item is None for item in numbers):
        return None
    if key == "frame_id" and any(not float(item).is_integer() for item in numbers if item is not None):
        return None
    values = tuple(float(item) for item in numbers if item is not None)
    if not values or any(b <= a for a, b in zip(values, values[1:])):
        return None
    return values


def timeline_axis(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, ...], str]:
    """One common, evidence-backed x-axis across *all* stages in a trace.

    Prefer frame_id, then capture timestamp_ms. Only when neither is usable use
    positional sample numbers, clearly marked as such. Do not sort or silently
    renumber frames and never discard rows from this shared axis.
    """
    if not rows:
        return (), "Numer pomiaru"
    frames = _strictly_increasing_numbers(rows, "frame_id")
    if frames is not None:
        return frames, "ID klatki"
    timestamps = _strictly_increasing_numbers(rows, "timestamp_ms")
    if timestamps is not None:
        start = timestamps[0]
        return tuple((value - start) / 1000.0 for value in timestamps), "Czas od pierwszej klatki [s]"
    return tuple(float(i) for i in range(1, len(rows) + 1)), "Numer pomiaru (bez ID klatki)"


def timeline_stages(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    """Preserve gaps and source indices; no independent re-spacing of MZ values."""
    xs, label = timeline_axis(rows)
    stages: dict[str, tuple[float | None, ...]] = {}
    for field in fields:
        stages[field] = tuple(_nonnegative_finite(row.get(field)) for row in rows)
    return {"x": xs, "axis_label": label, "stages": stages, "count": len(rows)}


def continuous_segments(series: Sequence[float | None], x: Sequence[float] | None = None,
                        *, require_consecutive: bool = False) -> tuple[tuple[int, int], ...]:
    """Connect only neighboring measured rows; never bridge absent source frames."""
    if require_consecutive and (x is None or len(series) != len(x)):
        return ()
    return tuple((i - 1, i) for i in range(1, len(series))
                 if series[i - 1] is not None and series[i] is not None
                 and (not require_consecutive or abs(x[i] - x[i - 1] - 1.0) < 1e-9))


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ("true", "1", "yes"):
        return True
    if str(value).strip().lower() in ("false", "0", "no"):
        return False
    return None


def stage_enabled(report: Any, role: str) -> bool | None:
    """Return an explicit stage activation only; absence is UNKNOWN, not off.

    Research execution configuration has priority over legacy compact fields.
    """
    raw = getattr(report, "raw", None)
    if not isinstance(raw, dict):
        return None
    role = str(role or "").lower()
    canonical = {"mp": "vehicle", "mt": "plate", "mz": "character"}.get(role, role)
    paths = (
        ("research_execution_config", "stages", role),
        ("experiment_session", "execution", "stages", role),
        ("execution", canonical),
        ("models", canonical),
        ("research_collection", "execution", "stages", role),
    )
    for path in paths:
        item: Any = raw
        for key in path:
            if not isinstance(item, dict) or key not in item:
                break
            item = item[key]
        else:
            if isinstance(item, dict) and "enabled" in item:
                status = _boolean(item["enabled"])
                if status is not None:
                    return status
    return None


def model_input_sizes(report: Any) -> str | None:
    """Describe *model* input dimensions; never present them as camera resolution."""
    raw = getattr(report, "raw", None)
    if not isinstance(raw, dict):
        return None
    execution = raw.get("research_execution_config")
    stages = execution.get("stages") if isinstance(execution, dict) else None
    if not isinstance(stages, dict):
        return None
    result = []
    for role in ("mt", "mz"):
        value = stages.get(role)
        source = value.get("input") if isinstance(value, dict) else None
        if not isinstance(source, dict) or _boolean(value.get("enabled")) is False:
            continue
        width = _nonnegative_finite(source.get("width"))
        height = _nonnegative_finite(source.get("height"))
        if width is None or height is None or width <= 0 or height <= 0 or not width.is_integer() or not height.is_integer():
            continue
        result.append(f"{role.upper()} {int(width)}×{int(height)}")
    return "; ".join(result) if result else None


def local_measurement_time(value: Any, *, compact: bool = False, zone: tzinfo | None = None, show_offset: bool = True) -> str:
    """Render aware source timestamps in *machine* local time, with an offset.

    Never silently assume a timezone for a naive historical timestamp. Source
    UTC remains unchanged in source JSON and detailed provenance.
    """
    string = str(value or "").strip()
    if not string:
        return "—"
    try:
        dt = datetime.fromisoformat(string.replace("Z", "+00:00"))
    except ValueError:
        return string + " (strefa nieznana)"
    pattern = "%d.%m %H:%M" if compact else "%d.%m.%Y %H:%M:%S"
    if dt.tzinfo is None:
        return dt.strftime(pattern) + " (brak strefy źródłowej)"
    dt = dt.astimezone(zone)
    offset = dt.strftime("%z")
    offset_label = f"UTC{offset[:3]}:{offset[3:]}" if offset else "strefa lokalna"
    return dt.strftime(pattern) + (f" ({offset_label})" if show_offset else "")


def crash_description(errors: Mapping[str, Any]) -> tuple[str, str] | None:
    """Historical recovery counter != failures of the selected measurement."""
    if not isinstance(errors, dict):
        return None
    if errors.get("crash_semantics") == "previous_uncontrolled_measurement_sessions_recovered_on_next_process_start":
        count = _nonnegative_finite(errors.get("crash_count"))
        total = str(int(count)) if count is not None and count.is_integer() else "nieznana liczba"
        return ("Odzyskane wcześniejsze sesje", total)
    return None
