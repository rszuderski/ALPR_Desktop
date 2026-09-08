#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe train-only YOLO dataset augmentation.

This module intentionally avoids crops, flips and strong perspective warps.
It only creates additional training images from the train split and transforms
YOLO labels together with the image.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..config import CONFIG, CV2_AVAILABLE, YAML_AVAILABLE, cv2, logger, np, yaml
from ..utils import safe_load_yaml


MAX_DIRT_FLOW_POINTS = 2000
MAX_RAIN_DROPS = 60000
MAX_RAIN_LENS_PIXELS = 1_600_000
MAX_RAIN_EDGE_PIXELS = 2_000_000
MAX_AUGMENTATION_PREVIEW_PIXELS = 1_400_000
MAX_CONTOUR_WIREFRAME_PIXELS = 360_000
MAX_CONTOUR_WIREFRAME_SEGMENTS = 2_400
DEFAULT_BASE_LIGHT_STRENGTH = 0.08
DEFAULT_RELIEF_PROFILE_PRESET = "rounded"
DEFAULT_RELIEF_PROFILE_CURVE = (
    (0.0, 0.0),
    (0.018, 0.02),
    (0.046, 0.96),
    (0.145, 0.98),
    (0.330, 1.0),
    (0.670, 1.0),
    (0.855, 0.98),
    (0.954, 0.96),
    (0.982, 0.02),
    (1.0, 0.0),
)
_RELIEF_PROFILE_PRESETS = {
    "rounded": {
        "label": "Zaokrąglony",
        "curve": DEFAULT_RELIEF_PROFILE_CURVE,
    },
    "flat": {
        "label": "Płaski grzbiet",
        "curve": (
            (0.0, 0.0),
            (0.012, 0.0),
            (0.036, 0.98),
            (0.120, 1.0),
            (0.360, 1.0),
            (0.640, 1.0),
            (0.880, 1.0),
            (0.964, 0.98),
            (0.988, 0.0),
            (1.0, 0.0),
        ),
    },
    "soft": {
        "label": "Miękki",
        "curve": (
            (0.0, 0.0),
            (0.060, 0.08),
            (0.160, 0.58),
            (0.300, 0.72),
            (0.420, 0.76),
            (0.580, 0.76),
            (0.700, 0.72),
            (0.840, 0.58),
            (0.940, 0.08),
            (1.0, 0.0),
        ),
    },
    "sharp": {
        "label": "Ostry",
        "curve": (
            (0.0, 0.0),
            (0.010, 0.0),
            (0.024, 1.0),
            (0.080, 1.0),
            (0.460, 1.0),
            (0.540, 1.0),
            (0.920, 1.0),
            (0.976, 1.0),
            (0.990, 0.0),
            (1.0, 0.0),
        ),
    },
    "worn": {
        "label": "Zużyty",
        "curve": (
            (0.0, 0.0),
            (0.030, 0.04),
            (0.075, 0.72),
            (0.180, 0.78),
            (0.370, 0.86),
            (0.630, 0.82),
            (0.820, 0.78),
            (0.925, 0.72),
            (0.970, 0.04),
            (1.0, 0.0),
        ),
    },
}

AUGMENTATION_RANDOMNESS_MODES = {"fixed", "soft", "realistic", "wide", "manual"}
AUGMENTATION_RANDOMNESS_LABELS = {
    "fixed": "Stały",
    "soft": "Łagodny",
    "realistic": "Realistyczny",
    "wide": "Agresywny",
    "manual": "Ręczny",
}
_AUGMENTATION_RANDOMNESS_ALIASES = {
    "": "realistic",
    "none": "fixed",
    "off": "fixed",
    "0": "fixed",
    "staly": "fixed",
    "stały": "fixed",
    "fixed": "fixed",
    "soft": "soft",
    "lagodny": "soft",
    "łagodny": "soft",
    "real": "realistic",
    "realistic": "realistic",
    "realistyczny": "realistic",
    "wide": "wide",
    "aggressive": "wide",
    "agresywny": "wide",
    "manual": "manual",
    "reczny": "manual",
    "ręczny": "manual",
    "ręczny": "manual",
}


def normalize_augmentation_randomness_mode(value: object) -> str:
    """Normalize persisted/UI randomness mode names."""
    raw = ("" if value is None else str(value)).strip().lower()
    raw = raw.replace("ą", "a").replace("ę", "e").replace("ó", "o").replace("ś", "s")
    raw = raw.replace("ł", "l").replace("ż", "z").replace("ź", "z").replace("ć", "c").replace("ń", "n")
    raw = raw.replace("ą", "a").replace("ę", "e").replace("ó", "o").replace("ś", "s")
    raw = raw.replace("ł", "l").replace("ż", "z").replace("ź", "z").replace("ć", "c").replace("ń", "n")
    return _AUGMENTATION_RANDOMNESS_ALIASES.get(raw, "realistic")


def describe_augmentation_randomness_mode(value: object) -> str:
    mode = normalize_augmentation_randomness_mode(value)
    return AUGMENTATION_RANDOMNESS_LABELS.get(mode, AUGMENTATION_RANDOMNESS_LABELS["realistic"])


def _legacy_relief_cubic_to_piecewise(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Convert the old one-curve profile into a stable three-curve profile."""
    if len(points) < 4:
        return list(DEFAULT_RELIEF_PROFILE_CURVE)
    peak = max(0.30, min(1.0, float(_sample_cubic_bezier_xy(points[:4], 0.5)[1])))
    left_x = max(0.10, min(0.34, float(points[1][0]) * 0.72))
    right_x = min(0.90, max(0.66, 1.0 - left_x))
    return [
        (0.0, 0.0),
        (max(0.008, left_x * 0.12), 0.02),
        (max(0.020, left_x * 0.32), peak * 0.96),
        (left_x, peak),
        (min(0.50, left_x + (0.50 - left_x) * 0.62), peak),
        (max(0.50, right_x - (right_x - 0.50) * 0.62), peak),
        (right_x, peak),
        (min(0.980, 1.0 - max(0.020, left_x * 0.32)), peak * 0.96),
        (min(0.992, 1.0 - max(0.008, left_x * 0.12)), 0.02),
        (1.0, 0.0),
    ]


def normalize_relief_profile_curve(value: object = None) -> tuple:
    """Normalize a three-piece cubic Bezier cross-section curve for raised glyph relief."""
    raw_points = value if isinstance(value, (list, tuple)) else DEFAULT_RELIEF_PROFILE_CURVE
    points: list[tuple[float, float]] = []
    try:
        for item in raw_points:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            x = max(0.0, min(1.0, float(item[0])))
            y = max(0.0, min(1.0, float(item[1])))
            points.append((x, y))
            if len(points) >= 10:
                break
    except Exception:
        points = []
    if len(points) == 4:
        points = _legacy_relief_cubic_to_piecewise(points)
    if len(points) < 10:
        points = list(DEFAULT_RELIEF_PROFILE_CURVE)
    points = list(points[:10])
    points[0] = (0.0, 0.0)
    points[9] = (1.0, 0.0)
    min_step = 0.006
    previous_x = 0.0
    normalized: list[tuple[float, float]] = [points[0]]
    for index in range(1, 9):
        max_x = 1.0 - min_step * (9 - index)
        x = max(previous_x + min_step, min(max_x, float(points[index][0])))
        y = max(0.0, min(1.0, float(points[index][1])))
        normalized.append((x, y))
        previous_x = x
    normalized.append(points[9])
    return tuple(normalized)


def normalize_relief_profile_preset(value: object) -> str:
    raw = str(value or DEFAULT_RELIEF_PROFILE_PRESET).strip().lower()
    if raw in _RELIEF_PROFILE_PRESETS or raw == "custom":
        return raw
    return DEFAULT_RELIEF_PROFILE_PRESET


def relief_profile_presets() -> list[dict]:
    return [
        {"key": key, "label": spec["label"], "curve": spec["curve"]}
        for key, spec in _RELIEF_PROFILE_PRESETS.items()
    ]


def relief_profile_curve_for_preset(preset: object, curve: object = None) -> tuple:
    key = normalize_relief_profile_preset(preset)
    if key != "custom":
        return normalize_relief_profile_curve(_RELIEF_PROFILE_PRESETS[key]["curve"])
    return normalize_relief_profile_curve(curve)


def relief_profile_preset_label(preset: object) -> str:
    key = normalize_relief_profile_preset(preset)
    if key == "custom":
        return "Własny"
    return str(_RELIEF_PROFILE_PRESETS.get(key, {}).get("label") or _RELIEF_PROFILE_PRESETS[DEFAULT_RELIEF_PROFILE_PRESET]["label"])


def _sample_cubic_bezier_xy(points: list[tuple[float, float]] | tuple, t: float) -> tuple[float, float]:
    local_t = max(0.0, min(1.0, float(t)))
    inv = 1.0 - local_t
    p0, p1, p2, p3 = points[:4]
    x = (
        inv * inv * inv * float(p0[0])
        + 3.0 * inv * inv * local_t * float(p1[0])
        + 3.0 * inv * local_t * local_t * float(p2[0])
        + local_t * local_t * local_t * float(p3[0])
    )
    y = (
        inv * inv * inv * float(p0[1])
        + 3.0 * inv * inv * local_t * float(p1[1])
        + 3.0 * inv * local_t * local_t * float(p2[1])
        + local_t * local_t * local_t * float(p3[1])
    )
    return x, y


def _sample_relief_profile_y_scalar(curve: tuple, x_value: float) -> float:
    x_value = max(0.0, min(1.0, float(x_value)))
    segments = ((0, 1, 2, 3), (3, 4, 5, 6), (6, 7, 8, 9))
    for segment_indexes in segments:
        segment = [curve[index] for index in segment_indexes]
        start_x = float(segment[0][0])
        end_x = float(segment[3][0])
        if x_value < start_x - 1e-9 or x_value > end_x + 1e-9:
            continue
        low, high = 0.0, 1.0
        for _iteration in range(18):
            mid = (low + high) / 2.0
            current_x, _current_y = _sample_cubic_bezier_xy(segment, mid)
            if current_x < x_value:
                low = mid
            else:
                high = mid
        _sample_x, sample_y = _sample_cubic_bezier_xy(segment, (low + high) / 2.0)
        return max(0.0, min(1.0, float(sample_y)))
    return 0.0


def _sample_relief_profile_y(curve: object, t):
    points = normalize_relief_profile_curve(curve)
    if np is not None and hasattr(t, "__array__"):
        local_t = np.clip(np.asarray(t, dtype="float32"), 0.0, 1.0)
        result = np.zeros_like(local_t, dtype="float32")
        for segment_indexes in ((0, 1, 2, 3), (3, 4, 5, 6), (6, 7, 8, 9)):
            segment = [points[index] for index in segment_indexes]
            start_x = float(segment[0][0])
            end_x = float(segment[3][0])
            mask = (local_t >= start_x - 1e-6) & (local_t <= end_x + 1e-6)
            if not np.any(mask):
                continue
            samples = np.linspace(0.0, 1.0, 96, dtype="float32")
            inv = 1.0 - samples
            sx = (
                inv * inv * inv * float(segment[0][0])
                + 3.0 * inv * inv * samples * float(segment[1][0])
                + 3.0 * inv * samples * samples * float(segment[2][0])
                + samples * samples * samples * float(segment[3][0])
            )
            sy = (
                inv * inv * inv * float(segment[0][1])
                + 3.0 * inv * inv * samples * float(segment[1][1])
                + 3.0 * inv * samples * samples * float(segment[2][1])
                + samples * samples * samples * float(segment[3][1])
            )
            order = np.argsort(sx)
            sx = sx[order]
            sy = sy[order]
            unique_x, unique_index = np.unique(sx, return_index=True)
            if unique_x.size < 2:
                result[mask] = float(sy[0] if sy.size else 0.0)
            else:
                result[mask] = np.interp(local_t[mask], unique_x, sy[unique_index])
        return np.clip(result, 0.0, 1.0)
    return _sample_relief_profile_y_scalar(points, float(t))


def _relief_profile_height_from_center_ratio(center_ratio, curve: object):
    """Map distance from nearest stroke edge to the full Bezier profile."""
    left_t = np.clip(np.asarray(center_ratio, dtype="float32") * 0.5, 0.0, 0.5)
    right_t = 1.0 - left_t
    return np.maximum(_sample_relief_profile_y(curve, left_t), _sample_relief_profile_y(curve, right_t))


_MANUAL_RANDOMNESS_GROUPS: tuple[dict, ...] = (
    {
        "key": "geometry",
        "label": "Geometria",
        "targets": ("plate",),
        "note": "Dostępna tylko w torze tablic. Tor znaków chroni rektyfikację.",
        "fields": (
            ("rotation_limit", "Obrót"),
            ("translate_limit", "Przesunięcie"),
            ("scale_limit", "Skala"),
        ),
    },
    {
        "key": "color",
        "label": "Kolor i tonalność",
        "targets": ("plate", "char"),
        "note": "Jasność, kontrast i nasycenie bez zmiany geometrii.",
        "fields": (
            ("brightness_limit", "Jasność"),
            ("contrast_limit", "Kontrast"),
            ("saturation_limit", "Nasycenie"),
        ),
    },
    {
        "key": "weather",
        "label": "Pogoda",
        "targets": ("plate", "char"),
        "note": "Opad, krople i mgiełka przy konturach.",
        "fields": (
            ("rain_strength", "Deszcz"),
            ("rain_drop_size_min", "Kropla min"),
            ("rain_drop_size_max", "Kropla max"),
            ("rain_alpha", "Alfa kropli"),
            ("rain_vector_field_strength", "Pole deszczu"),
            ("rain_vortex_strength", "Wiry deszczu"),
            ("rain_lens_strength", "Soczewka kropli"),
            ("rain_edge_mist_strength", "Mgiełka przy konturach"),
            ("rain_edge_mist_radius", "Promień mgiełki"),
            ("tyndall_strength", "Efekt Tyndalla"),
        ),
    },
    {
        "key": "illumination",
        "label": "Światło",
        "targets": ("plate", "char"),
        "note": "Noc, światło bazowe, reflektory, odbicia wtórne i przesłonięcia światła.",
        "fields": (
            ("night_strength", "Noc"),
            ("night_light_strength", "Światło bazowe"),
            ("night_light_warmth", "Temperatura światła"),
            ("scene_camera_z", "Kamera z"),
            ("relief_bounce_depth", "Głębia odbić"),
            ("relief_bounce_strength", "Siła odbić wtórnych"),
            ("traffic_headlight_strength", "Reflektor R1 moc"),
            ("traffic_headlight_source_world_x", "R1 x"),
            ("traffic_headlight_source_world_y", "R1 y"),
            ("traffic_headlight_source_world_z", "R1 z"),
            ("traffic_headlight_1_cone", "R1 stożek"),
            ("traffic_headlight_1_source_radius", "R1 promień"),
            ("traffic_headlight_2_strength", "Reflektor R2 moc"),
            ("traffic_headlight_2_source_world_x", "R2 x"),
            ("traffic_headlight_2_source_world_y", "R2 y"),
            ("traffic_headlight_2_source_world_z", "R2 z"),
            ("traffic_headlight_2_cone", "R2 stożek"),
            ("traffic_headlight_3_strength", "Reflektor R3 moc"),
            ("traffic_headlight_3_source_world_x", "R3 x"),
            ("traffic_headlight_3_source_world_y", "R3 y"),
            ("traffic_headlight_3_source_world_z", "R3 z"),
            ("traffic_headlight_3_cone", "R3 stożek"),
            ("overhang_shadow_strength", "Cień daszka"),
            ("overhang_shadow_depth", "Zasięg cienia"),
            ("overhang_shadow_skew", "Skos cienia"),
        ),
    },
    {
        "key": "material",
        "label": "Materiał i powierzchnia",
        "targets": ("plate", "char"),
        "note": "Powierzchnia tablicy, film wodny i odbicia materiałowe.",
        "fields": (
            ("water_film_strength", "Film wodny"),
            ("water_film_unevenness", "Nierówność filmu"),
            ("water_film_lens_strength", "Soczewka filmu"),
            ("water_film_contour_response", "Reakcja filmu na kontury"),
            ("water_film_gloss_strength", "Połysk filmu"),
            ("plate_reflect_gradient_strength", "Kierunkowe cieniowanie"),
            ("plate_reflect_glare_strength", "Odblask tablicy"),
            ("plate_reflect_curve_strength", "Krzywizna tablicy"),
        ),
    },
    {
        "key": "relief",
        "label": "Relief i kontury",
        "targets": ("char",),
        "note": "Wypukłość tuszu, kontury znaków i profil przekroju.",
        "fields": (
            ("dark_relief_strength", "Wypukłość konturów"),
        ),
    },
    {
        "key": "dirt",
        "label": "Błoto i zabrudzenia",
        "targets": ("char",),
        "note": "Grudki, smugi, lepkość, krycie i ruch zabrudzeń po powierzchni.",
        "fields": (
            ("wet_mud_gloss_strength", "Połysk błota"),
            ("dirt_flow_points", "Liczba smug błota"),
            ("dirt_flow_mass_min", "Masa błota min"),
            ("dirt_flow_mass_max", "Masa błota max"),
            ("dirt_flow_trail_length", "Długość smug"),
            ("dirt_flow_humidity", "Wilgotność błota"),
            ("dirt_flow_stickiness_min", "Lepkość min"),
            ("dirt_flow_stickiness_max", "Lepkość max"),
            ("dirt_flow_opacity_min", "Krycie min"),
            ("dirt_flow_opacity_max", "Krycie max"),
            ("dirt_flow_air_angle", "Kierunek wiatru"),
            ("dirt_flow_wind_strength", "Siła wiatru"),
            ("vehicle_speed", "Ruch pojazdu"),
        ),
    },
    {
        "key": "sensor",
        "label": "Kamera i optyka",
        "targets": ("plate", "char"),
        "note": "Blur, szum i optyczna utrata informacji.",
        "fields": (
            ("blur_strength", "Rozmycie"),
            ("noise_strength", "Szum"),
            ("noise_grain_size", "Ziarno szumu"),
            ("night_iso_noise_strength", "Szum ISO"),
            ("night_bloom_strength", "Poświata"),
            ("flare_strength", "Glare kamery"),
            ("overexposure_strength", "Prześwietlenie"),
        ),
    },
)


def _normalize_task_target_value(value: object, *, fallback: str = "char") -> str:
    try:
        normalized = CONFIG.normalize_task_target(str(value or ""))
        if normalized in {"plate", "char"}:
            return normalized
    except Exception:
        pass
    raw = str(value or "").strip().lower()
    if raw in {"plate", "plates", "tablice", "tablica"}:
        return "plate"
    if raw in {"char", "chars", "znaki", "znak"}:
        return "char"
    return "plate" if str(fallback or "").lower() == "plate" else "char"


def _profile_task_target(profile: object) -> str:
    raw = getattr(profile, "task_target", None)
    if raw:
        return _normalize_task_target_value(raw)
    class_name = str(getattr(profile, "class_name", "") or "").strip().lower()
    if class_name == "plate":
        return "plate"
    return "char"


def manual_randomness_group_specs(target: str | None = None) -> list[dict]:
    normalized_target = _normalize_task_target_value(target, fallback="char") if target else ""
    specs: list[dict] = []
    for group in _MANUAL_RANDOMNESS_GROUPS:
        targets = tuple(group.get("targets") or ("plate", "char"))
        applicable = not normalized_target or normalized_target in targets
        specs.append(
            {
                "key": str(group.get("key") or ""),
                "label": str(group.get("label") or group.get("key") or ""),
                "note": str(group.get("note") or ""),
                "targets": targets,
                "applicable": bool(applicable),
                "fields": [
                    {"key": str(field_key), "label": str(field_label)}
                    for field_key, field_label in tuple(group.get("fields") or ())
                ],
            }
        )
    return specs


def _percent(value: object, default: float = 100.0) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except Exception:
        return max(0.0, min(100.0, float(default)))


def default_manual_randomness_config(target: str | None = None) -> dict:
    normalized_target = _normalize_task_target_value(target, fallback="char") if target else "char"
    groups: dict[str, dict] = {}
    for spec in manual_randomness_group_specs(normalized_target):
        applicable = bool(spec.get("applicable", True))
        groups[spec["key"]] = {
            "enabled": bool(applicable),
            "amount": 100.0 if applicable else 0.0,
            "fields": {
                field_spec["key"]: {"enabled": bool(applicable), "amount": 100.0 if applicable else 0.0}
                for field_spec in spec.get("fields", [])
            },
        }
    return {"schema": "augmentation_manual_randomness_v1", "target": normalized_target, "groups": groups}


def normalize_manual_randomness_config(value: object, target: str | None = None) -> dict:
    normalized_target = _normalize_task_target_value(target, fallback="char") if target else "char"
    base = default_manual_randomness_config(normalized_target)
    incoming = value if isinstance(value, dict) else {}
    incoming_groups = incoming.get("groups") if isinstance(incoming.get("groups"), dict) else incoming
    if not isinstance(incoming_groups, dict):
        incoming_groups = {}
    for spec in manual_randomness_group_specs(normalized_target):
        group_key = spec["key"]
        applicable = bool(spec.get("applicable", True))
        current = base["groups"][group_key]
        raw_group = incoming_groups.get(group_key, {})
        if isinstance(raw_group, dict):
            current["enabled"] = bool(raw_group.get("enabled", current["enabled"])) and applicable
            current["amount"] = _percent(raw_group.get("amount", current["amount"]), current["amount"]) if applicable else 0.0
            raw_fields = raw_group.get("fields", {})
            if not isinstance(raw_fields, dict):
                raw_fields = {}
            for field_spec in spec.get("fields", []):
                field_key = field_spec["key"]
                field_state = current["fields"][field_key]
                raw_field = raw_fields.get(field_key, {})
                if (
                    not raw_field
                    and group_key == "illumination"
                    and field_key in {"relief_bounce_depth", "relief_bounce_strength"}
                ):
                    legacy_group = incoming_groups.get("relief", {})
                    legacy_fields = legacy_group.get("fields", {}) if isinstance(legacy_group, dict) else {}
                    if isinstance(legacy_fields, dict):
                        raw_field = legacy_fields.get(field_key, {})
                if isinstance(raw_field, dict):
                    field_state["enabled"] = bool(raw_field.get("enabled", field_state["enabled"])) and applicable
                    field_state["amount"] = _percent(raw_field.get("amount", field_state["amount"]), field_state["amount"]) if applicable else 0.0
        if not applicable:
            current["enabled"] = False
            current["amount"] = 0.0
            for field_state in current["fields"].values():
                field_state["enabled"] = False
                field_state["amount"] = 0.0
    return base


def describe_manual_randomness_config(value: object, target: str | None = None) -> dict:
    normalized = normalize_manual_randomness_config(value, target)
    summary: list[dict] = []
    for spec in manual_randomness_group_specs(normalized.get("target") or target):
        group = normalized["groups"].get(spec["key"], {})
        fields = []
        for field_spec in spec.get("fields", []):
            state = (group.get("fields") or {}).get(field_spec["key"], {})
            fields.append(
                {
                    "key": field_spec["key"],
                    "label": field_spec["label"],
                    "enabled": bool(state.get("enabled")),
                    "amount": _percent(state.get("amount", 0.0), 0.0),
                }
            )
        summary.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "enabled": bool(group.get("enabled")),
                "amount": _percent(group.get("amount", 0.0), 0.0),
                "applicable": bool(spec.get("applicable", True)),
                "fields": fields,
            }
        )
    return {"schema": "augmentation_manual_randomness_summary_v1", "target": normalized.get("target"), "groups": summary}


def _profile_float(profile, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(profile, name, default) if getattr(profile, name, None) is not None else default)
    except Exception:
        return float(default)


def _profile_int(profile, name: str, default: int = 0) -> int:
    try:
        return int(float(getattr(profile, name, default) if getattr(profile, name, None) is not None else default))
    except Exception:
        return int(default)


def _profile_headlight_rgb(profile, index: int, warmth: float | None = None) -> tuple[float, float, float]:
    def clamp01(value: object, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return max(0.0, min(1.0, float(default)))

    try:
        index = max(1, min(3, int(index or 1)))
    except Exception:
        index = 1
    if warmth is None:
        default_warmth = _profile_float(profile, "night_light_warmth", 0.35)
        warmth_name = "traffic_headlight_1_warmth" if index == 1 else f"traffic_headlight_{index}_warmth"
        warmth = _profile_float(profile, warmth_name, -1.0 if index == 1 else 0.35)
        if warmth < 0.0:
            warmth = default_warmth
    warmth = clamp01(warmth, 0.35)
    prefix = "traffic_headlight_1" if index == 1 else f"traffic_headlight_{index}"
    red = _profile_float(profile, f"{prefix}_r", -1.0)
    green = _profile_float(profile, f"{prefix}_g", -1.0)
    blue = _profile_float(profile, f"{prefix}_b", -1.0)
    if red >= 0.0 and green >= 0.0 and blue >= 0.0:
        return clamp01(red, 1.0), clamp01(green, 0.88), clamp01(blue, 0.54)
    cold_rgb = (0.84, 0.94, 1.0)
    warm_rgb = (1.0, 0.88, 0.54)
    return tuple(cold_rgb[i] * (1.0 - warmth) + warm_rgb[i] * warmth for i in range(3))


def _profile_headlight_bgr(profile, index: int, warmth: float | None = None, scale: float = 255.0) -> tuple[float, float, float]:
    red, green, blue = _profile_headlight_rgb(profile, index, warmth)
    return blue * scale, green * scale, red * scale


def _truthy_control_names(profile, controls: tuple[tuple[str, str, str], ...]) -> list[str]:
    active: list[str] = []
    for label, name, kind in controls:
        if kind == "delta":
            if abs(_profile_float(profile, name, 0.0)) > 0.001:
                active.append(label)
        elif kind == "saturation":
            if abs(_profile_float(profile, name, 1.0) - 1.0) > 0.001:
                active.append(label)
        elif kind == "int":
            if _profile_int(profile, name, 0) > 0:
                active.append(label)
        elif kind == "grain":
            if _profile_int(profile, name, 1) > 1:
                active.append(label)
        else:
            if _profile_float(profile, name, 0.0) > 0.001:
                active.append(label)
    return active


def describe_augmentation_components(profile: "AugmentationProfile") -> dict:
    """Describe orthogonal augmentation components used by preview/export.

    The rendered image is intentionally irreversible, but this contract keeps
    the process reproducible and auditable through profile + seed + component
    state.
    """
    profile = (profile or AugmentationProfile()).normalized()
    def component(order: int, component_id: str, label: str, intent: str, controls: list[str]) -> dict:
        return {
            "id": component_id,
            "order": order,
            "label": label,
            "intent": intent,
            "active": bool(controls),
            "active_controls": list(controls),
        }

    surface_rain_active = _profile_float(profile, "rain_strength") > 0.001
    rain_active = surface_rain_active
    fog_active = _profile_float(profile, "rain_edge_mist_strength") > 0.001
    dirt_active = _profile_int(profile, "dirt_flow_points") > 0
    headlight_active = any(
        _profile_float(profile, name) > 0.001
        for name in ("traffic_headlight_strength", "traffic_headlight_2_strength", "traffic_headlight_3_strength")
    )
    illumination_base_active = any(
        _profile_float(profile, name) > 0.001
        for name in ("night_strength", "night_light_strength", "overexposure_strength")
    ) or headlight_active

    scene_controls: list[str] = []
    scene_defaults = (
        ("dystans obserwatora", "scene_camera_z", 1.65),
    )
    for label, name, default in scene_defaults:
        if abs(_profile_float(profile, name, default) - default) > 0.001 and label not in scene_controls:
            scene_controls.append(label)
    if headlight_active:
        scene_controls.append("reflektory x/y/z")
    geometry_controls = _truthy_control_names(
        profile,
        (
            ("obrót", "rotation_limit", "delta"),
            ("przesunięcie", "translate_limit", "float"),
            ("skala", "scale_limit", "float"),
        ),
    )
    color_controls = _truthy_control_names(
        profile,
        (
            ("jasność", "brightness_limit", "float"),
            ("kontrast", "contrast_limit", "float"),
            ("nasycenie", "saturation_limit", "saturation"),
        ),
    )
    weather_controls: list[str] = []
    if rain_active or fog_active:
        if surface_rain_active:
            weather_controls.extend(
                _truthy_control_names(
                    profile,
                    (
                        ("deszcz", "rain_strength", "float"),
                        ("pole wektorowe deszczu", "rain_vector_field_strength", "float"),
                        ("wiry deszczu", "rain_vortex_strength", "float"),
                        ("soczewka kropli", "rain_lens_strength", "float"),
                        ("mgła przy krawędziach", "rain_edge_mist_strength", "float"),
                    ),
                )
            )
    if (rain_active or fog_active) and _profile_float(profile, "tyndall_strength", 0.0) > 0.001:
        weather_controls.append("efekt Tyndalla")

    material_controls: list[str] = []
    if _profile_float(profile, "wet_reflection_strength") > 0.001:
        material_controls.append("mokre odbicie")
    if _profile_float(profile, "water_film_strength") > 0.001:
        material_controls.extend(
            _truthy_control_names(
                profile,
                (
                    ("film wodny", "water_film_strength", "float"),
                    ("nierówność filmu", "water_film_unevenness", "float"),
                    ("soczewkowanie filmu", "water_film_lens_strength", "float"),
                    ("reakcja filmu na kontury", "water_film_contour_response", "float"),
                    ("połysk filmu wodnego", "water_film_gloss_strength", "float"),
                ),
            )
        )
    material_controls.extend(
        _truthy_control_names(
            profile,
            (
                ("kierunkowe cieniowanie", "plate_reflect_gradient_strength", "float"),
                ("odblask tablicy", "plate_reflect_glare_strength", "float"),
                ("wygięcie powierzchni", "plate_reflect_curve_strength", "float"),
            ),
        )
    )

    relief_controls = _truthy_control_names(
        profile,
        (
            ("wypukłość konturów", "dark_relief_strength", "float"),
        ),
    )
    if abs(_profile_float(profile, "contour_detection_sensitivity", 0.55) - 0.55) > 0.001:
        relief_controls.append("czułość wykrywania konturów")
    if normalize_relief_profile_preset(getattr(profile, "relief_profile_preset", DEFAULT_RELIEF_PROFILE_PRESET)) != DEFAULT_RELIEF_PROFILE_PRESET:
        relief_controls.append(f"profil: {relief_profile_preset_label(getattr(profile, 'relief_profile_preset', DEFAULT_RELIEF_PROFILE_PRESET))}")

    dirt_controls: list[str] = []
    if dirt_active:
        dirt_controls.append("grudki błota")
        dirt_controls.extend(
            _truthy_control_names(
                profile,
                (
                    ("smugi błota", "dirt_flow_trail_length", "float"),
                    ("wilgoć błota", "dirt_flow_humidity", "float"),
                    ("połysk mokrego błota", "wet_mud_gloss_strength", "float"),
                    ("ruch pojazdu", "vehicle_speed", "float"),
                ),
            )
        )

    illumination_controls: list[str] = []
    if illumination_base_active:
        illumination_controls.extend(
            _truthy_control_names(
                profile,
                (
                    ("noc", "night_strength", "float"),
                    ("światło bazowe", "night_light_strength", "float"),
                    ("reflektor R1", "traffic_headlight_strength", "float"),
                    ("reflektor R2", "traffic_headlight_2_strength", "float"),
                    ("reflektor R3", "traffic_headlight_3_strength", "float"),
                    ("cień daszka", "overhang_shadow_strength", "float"),
                ),
            )
        )
        if abs(_profile_float(profile, "night_light_warmth", 0.35) - 0.35) > 0.001:
            illumination_controls.append("barwa światła")
        if (
            _profile_float(profile, "dark_relief_strength") > 0.001
            and int(float(getattr(profile, "relief_bounce_depth", 1) or 1)) > 1
            and _profile_float(profile, "relief_bounce_strength") > 0.001
        ):
            illumination_controls.append("wtórne odbicia światła")

    sensor_controls = _truthy_control_names(
        profile,
        (
            ("rozmycie", "blur_strength", "float"),
            ("szum", "noise_strength", "float"),
            ("ziarno", "noise_grain_size", "grain"),
            ("szum ISO", "night_iso_noise_strength", "float"),
            ("poświata", "night_bloom_strength", "float"),
            ("glare kamery", "flare_strength", "float"),
            ("prześwietlenie", "overexposure_strength", "float"),
        ),
    )
    components = [
        component(1, "scene", "Scena 3D", "Wspolny uklad kamery, tablicy i reflektorow.", scene_controls),
        component(1, "geometry", "Geometria", "Pozycja obrazu i resampling etykiet.", geometry_controls),
        component(2, "color", "Kolor i tonalność", "Jasność, kontrast i nasycenie bez zmiany geometrii.", color_controls),
        component(3, "weather", "Pogoda", "Deszcz, krople i mgiełka przy konturach.", weather_controls),
        component(4, "material", "Materiał tablicy", "Film wodny, połysk, soczewkowanie i odbicia powierzchni tablicy.", material_controls),
        component(5, "relief", "Relief i kontury", "Wypukłość znaków, cień lokalny i profil konturu.", relief_controls),
        component(6, "dirt", "Błoto i zabrudzenia", "Grudki, smugi, lepkość, krycie i ruch zabrudzeń po powierzchni.", dirt_controls),
        component(7, "illumination", "Światło", "Noc, światło bazowe, reflektory, odbicia wtórne i przesłonięcia światła.", illumination_controls),
        component(8, "sensor", "Kamera i optyka", "Utrata informacji: blur, szum, poświata, glare i artefakty optyczne.", sensor_controls),
    ]
    for order, item in enumerate(components, start=1):
        item["order"] = order
    active_count = sum(1 for item in components if item.get("active"))
    randomness_mode = normalize_augmentation_randomness_mode(getattr(profile, "randomness_mode", "realistic"))
    return {
        "model": "orthogonal_components_v1",
        "irreversible_result": True,
        "reproducible_process": True,
        "seed": int(getattr(profile, "seed", 42) or 42),
        "randomness_mode": randomness_mode,
        "randomness_label": describe_augmentation_randomness_mode(randomness_mode),
        "active_count": active_count,
        "components": components,
    }


@dataclass(frozen=True)
class AugmentationProfile:
    """User-facing augmentation settings for one dataset build."""

    enabled: bool = False
    sample_size: int = 32
    extra_count: int = 0
    rotation_limit: float = 0.0
    translate_limit: float = 0.0
    scale_limit: float = 0.0
    brightness_limit: float = 0.0
    contrast_limit: float = 0.0
    saturation_limit: float = 1.0
    noise_strength: float = 0.0
    noise_grain_size: int = 1
    rain_strength: float = 0.0
    rain_drop_size: float = 0.07
    rain_drop_size_min: float = 0.01
    rain_drop_size_max: float = 0.13
    rain_vector_field_strength: float = 0.0
    rain_vortex_strength: float = 0.0
    rain_alpha: float = 0.22
    rain_lens_strength: float = 0.0
    rain_edge_mist_strength: float = 0.0
    rain_edge_mist_radius: float = 0.45
    tyndall_strength: float = 0.55
    wet_reflection_strength: float = 0.0
    vehicle_speed: float = 0.0
    night_strength: float = 0.0
    night_luma_min: float = 0.56
    night_luma_max: float = 0.92
    night_light_strength: float = DEFAULT_BASE_LIGHT_STRENGTH
    night_bloom_strength: float = 0.0
    night_iso_noise_strength: float = 0.0
    night_light_warmth: float = 0.35
    scene_plate_width: float = 1.0
    scene_plate_height: float = 0.24
    scene_camera_x: float = 0.0
    scene_camera_y: float = 0.0
    scene_camera_z: float = 1.65
    scene_camera_target_x: float = 0.0
    scene_camera_target_y: float = 0.0
    scene_camera_target_z: float = 0.0
    scene_view_yaw: float = 35.0
    scene_view_pitch: float = -24.0
    scene_view_roll: float = 0.0
    scene_plate_texture_enabled: bool = True
    traffic_headlight_strength: float = 0.0
    traffic_headlight_count: int = 1
    traffic_headlight_source_x: float = -1.0
    traffic_headlight_source_y: float = -1.0
    traffic_headlight_target_x: float = -1.0
    traffic_headlight_target_y: float = -1.0
    traffic_headlight_source_world_x: float = -999.0
    traffic_headlight_source_world_y: float = -999.0
    traffic_headlight_source_world_z: float = 0.95
    traffic_headlight_target_world_x: float = -999.0
    traffic_headlight_target_world_y: float = -999.0
    traffic_headlight_target_world_z: float = 0.0
    traffic_headlight_1_warmth: float = -1.0
    traffic_headlight_1_r: float = -1.0
    traffic_headlight_1_g: float = -1.0
    traffic_headlight_1_b: float = -1.0
    traffic_headlight_1_cone: float = 0.45
    traffic_headlight_1_source_radius: float = 0.08
    traffic_headlight_2_strength: float = 0.0
    traffic_headlight_2_warmth: float = 0.35
    traffic_headlight_2_r: float = -1.0
    traffic_headlight_2_g: float = -1.0
    traffic_headlight_2_b: float = -1.0
    traffic_headlight_2_cone: float = 0.45
    traffic_headlight_2_source_radius: float = 0.08
    traffic_headlight_2_source_x: float = -1.0
    traffic_headlight_2_source_y: float = -1.0
    traffic_headlight_2_target_x: float = -1.0
    traffic_headlight_2_target_y: float = -1.0
    traffic_headlight_2_source_world_x: float = -999.0
    traffic_headlight_2_source_world_y: float = -999.0
    traffic_headlight_2_source_world_z: float = 0.95
    traffic_headlight_2_target_world_x: float = -999.0
    traffic_headlight_2_target_world_y: float = -999.0
    traffic_headlight_2_target_world_z: float = 0.0
    traffic_headlight_3_strength: float = 0.0
    traffic_headlight_3_warmth: float = 0.35
    traffic_headlight_3_r: float = -1.0
    traffic_headlight_3_g: float = -1.0
    traffic_headlight_3_b: float = -1.0
    traffic_headlight_3_cone: float = 0.45
    traffic_headlight_3_source_radius: float = 0.08
    traffic_headlight_3_source_x: float = -1.0
    traffic_headlight_3_source_y: float = -1.0
    traffic_headlight_3_target_x: float = -1.0
    traffic_headlight_3_target_y: float = -1.0
    traffic_headlight_3_source_world_x: float = -999.0
    traffic_headlight_3_source_world_y: float = -999.0
    traffic_headlight_3_source_world_z: float = 0.95
    traffic_headlight_3_target_world_x: float = -999.0
    traffic_headlight_3_target_world_y: float = -999.0
    traffic_headlight_3_target_world_z: float = 0.0
    wet_mud_gloss_strength: float = 0.0
    water_film_strength: float = 0.0
    water_film_unevenness: float = 0.35
    water_film_lens_strength: float = 0.0
    water_film_contour_response: float = 0.55
    water_film_gloss_strength: float = 0.45
    flare_strength: float = 0.0
    overexposure_strength: float = 0.0
    dirt_streak_strength: float = 0.0
    dirt_flow_strength: float = 0.0
    dirt_flow_points: int = 0
    dirt_flow_mass_min: float = 0.18
    dirt_flow_mass_max: float = 1.0
    dirt_flow_splash_scale: float = 0.45
    dirt_flow_trail_length: float = 0.55
    dirt_flow_humidity: float = 0.45
    dirt_flow_stickiness: float = 0.45
    dirt_flow_stickiness_min: float = 0.30
    dirt_flow_stickiness_max: float = 0.70
    dirt_flow_air_angle: float = 0.0
    dirt_flow_wind_strength: float = 0.45
    dirt_flow_gravity_angle: float = 90.0
    dirt_flow_gravity_strength: float = 1.0
    dirt_flow_opacity_min: float = 0.25
    dirt_flow_opacity_max: float = 0.80
    dirt_flow_stop_on_dark_contour: bool = False
    contour_detection_sensitivity: float = 0.55
    dark_relief_strength: float = 0.30
    dark_relief_light_angle: float = 135.0
    relief_profile_preset: str = DEFAULT_RELIEF_PROFILE_PRESET
    relief_profile_curve: tuple = DEFAULT_RELIEF_PROFILE_CURVE
    relief_bounce_depth: int = 1
    relief_bounce_strength: float = 0.28
    light_normal_strength: float = 0.0
    overhang_shadow_strength: float = 0.0
    overhang_shadow_depth: float = 0.60
    overhang_shadow_skew: float = 0.0
    plate_reflect_gradient_strength: float = 0.0
    plate_reflect_glare_strength: float = 0.0
    plate_reflect_curve_strength: float = 0.0
    blur_strength: float = 0.0
    blur_enabled: bool = False
    randomness_mode: str = "realistic"
    seed: int = 42
    class_name: str = ""
    task_target: str = ""
    manual_randomness: dict = field(default_factory=dict)

    def normalized(self) -> "AugmentationProfile":
        task_target = _profile_task_target(self)
        manual_randomness = normalize_manual_randomness_config(
            getattr(self, "manual_randomness", {}),
            task_target,
        )
        dirt_flow_opacity_min = max(0.0, min(1.0, float(self.dirt_flow_opacity_min or 0.0)))
        dirt_flow_opacity_max = max(0.0, min(1.0, float(self.dirt_flow_opacity_max or 0.0)))
        if dirt_flow_opacity_min > dirt_flow_opacity_max:
            dirt_flow_opacity_min, dirt_flow_opacity_max = dirt_flow_opacity_max, dirt_flow_opacity_min
        dirt_flow_mass_min = max(0.05, min(2.8, float(getattr(self, "dirt_flow_mass_min", 0.18) or 0.18)))
        dirt_flow_mass_max = max(0.05, min(2.8, float(getattr(self, "dirt_flow_mass_max", 1.0) or 1.0)))
        if dirt_flow_mass_min > dirt_flow_mass_max:
            dirt_flow_mass_min, dirt_flow_mass_max = dirt_flow_mass_max, dirt_flow_mass_min
        legacy_stickiness = max(0.0, min(1.0, float(getattr(self, "dirt_flow_stickiness", 0.45) if getattr(self, "dirt_flow_stickiness", None) is not None else 0.45)))
        dirt_flow_stickiness_min = max(0.0, min(1.0, float(getattr(self, "dirt_flow_stickiness_min", legacy_stickiness) if getattr(self, "dirt_flow_stickiness_min", None) is not None else legacy_stickiness)))
        dirt_flow_stickiness_max = max(0.0, min(1.0, float(getattr(self, "dirt_flow_stickiness_max", legacy_stickiness) if getattr(self, "dirt_flow_stickiness_max", None) is not None else legacy_stickiness)))
        if dirt_flow_stickiness_min > dirt_flow_stickiness_max:
            dirt_flow_stickiness_min, dirt_flow_stickiness_max = dirt_flow_stickiness_max, dirt_flow_stickiness_min
        dirt_flow_stickiness = (dirt_flow_stickiness_min + dirt_flow_stickiness_max) / 2.0
        legacy_drop_size = max(0.0, min(1.0, float(self.rain_drop_size if self.rain_drop_size is not None else 0.35)))
        rain_drop_size_min = max(0.0, min(1.0, float(getattr(self, "rain_drop_size_min", legacy_drop_size) if getattr(self, "rain_drop_size_min", None) is not None else legacy_drop_size)))
        rain_drop_size_max = max(0.0, min(1.0, float(getattr(self, "rain_drop_size_max", legacy_drop_size) if getattr(self, "rain_drop_size_max", None) is not None else legacy_drop_size)))
        if rain_drop_size_min > rain_drop_size_max:
            rain_drop_size_min, rain_drop_size_max = rain_drop_size_max, rain_drop_size_min
        rain_drop_size = (rain_drop_size_min + rain_drop_size_max) / 2.0
        blur_strength = max(0.0, min(1.0, float(getattr(self, "blur_strength", 0.0) or 0.0)))
        if blur_strength <= 0.001 and bool(getattr(self, "blur_enabled", False)):
            blur_strength = 0.18
        scene_camera_z = max(0.35, min(4.0, float(getattr(self, "scene_camera_z", 1.65) if getattr(self, "scene_camera_z", None) is not None else 1.65)))
        scene_camera_target_x = 0.0
        scene_camera_target_y = 0.0
        camera_target_offset = math.hypot(scene_camera_target_x, scene_camera_target_y)
        camera_target_distance = math.sqrt(scene_camera_z * scene_camera_z + camera_target_offset * camera_target_offset)
        camera_target_cos = scene_camera_z / max(0.001, camera_target_distance)
        camera_axis_strength = max(0.0, min(1.0, (1.0 / (1.0 + scene_camera_z * 0.42)) * (0.74 + 0.26 * camera_target_cos)))
        rotation_limit = max(-15.0, min(15.0, float(self.rotation_limit or 0.0)))
        translate_limit = max(0.0, min(0.03, float(self.translate_limit or 0.0)))
        scale_limit = max(0.0, min(0.08, float(self.scale_limit or 0.0)))
        if task_target == "char":
            rotation_limit = 0.0
            translate_limit = 0.0
            scale_limit = 0.0
        relief_profile_preset = normalize_relief_profile_preset(getattr(self, "relief_profile_preset", DEFAULT_RELIEF_PROFILE_PRESET))
        relief_profile_curve = relief_profile_curve_for_preset(
            relief_profile_preset,
            getattr(self, "relief_profile_curve", DEFAULT_RELIEF_PROFILE_CURVE),
        )
        return AugmentationProfile(
            enabled=bool(self.enabled),
            sample_size=max(1, int(self.sample_size or 1)),
            extra_count=max(0, int(self.extra_count or 0)),
            rotation_limit=rotation_limit,
            translate_limit=translate_limit,
            scale_limit=scale_limit,
            brightness_limit=max(0.0, min(0.25, float(self.brightness_limit or 0.0))),
            contrast_limit=max(0.0, min(0.25, float(self.contrast_limit or 0.0))),
            saturation_limit=max(0.0, min(3.0, float(getattr(self, "saturation_limit", 1.0) if getattr(self, "saturation_limit", None) is not None else 1.0))),
            noise_strength=max(0.0, min(0.08, float(self.noise_strength or 0.0))),
            noise_grain_size=max(1, min(12, int(float(self.noise_grain_size or 1)))),
            rain_strength=max(0.0, min(1.0, float(self.rain_strength or 0.0))),
            rain_drop_size=rain_drop_size,
            rain_drop_size_min=rain_drop_size_min,
            rain_drop_size_max=rain_drop_size_max,
            rain_vector_field_strength=max(0.0, min(1.0, float(getattr(self, "rain_vector_field_strength", 0.0) or 0.0))),
            rain_vortex_strength=max(0.0, min(1.0, float(getattr(self, "rain_vortex_strength", 0.0) or 0.0))),
            rain_alpha=max(0.0, min(1.0, float(getattr(self, "rain_alpha", 0.22) if getattr(self, "rain_alpha", None) is not None else 0.22))),
            rain_lens_strength=max(0.0, min(1.0, float(getattr(self, "rain_lens_strength", 0.0) or 0.0))),
            rain_edge_mist_strength=max(0.0, min(1.0, float(getattr(self, "rain_edge_mist_strength", 0.0) or 0.0))),
            rain_edge_mist_radius=max(0.0, min(1.0, float(getattr(self, "rain_edge_mist_radius", 0.45) if getattr(self, "rain_edge_mist_radius", None) is not None else 0.45))),
            tyndall_strength=max(0.0, min(1.0, float(getattr(self, "tyndall_strength", 0.55) if getattr(self, "tyndall_strength", None) is not None else 0.55))),
            wet_reflection_strength=max(0.0, min(1.0, float(self.wet_reflection_strength if self.wet_reflection_strength is not None else 0.0))),
            vehicle_speed=max(0.0, min(1.0, float(self.vehicle_speed if self.vehicle_speed is not None else 0.0))),
            night_strength=max(0.0, min(1.0, float(self.night_strength or 0.0))),
            night_luma_min=0.56,
            night_luma_max=0.92,
            night_light_strength=max(0.0, min(1.0, float(getattr(self, "night_light_strength", 0.0) or 0.0))),
            night_bloom_strength=max(0.0, min(1.0, float(getattr(self, "night_bloom_strength", 0.0) or 0.0))),
            night_iso_noise_strength=max(0.0, min(1.0, float(getattr(self, "night_iso_noise_strength", 0.0) or 0.0))),
            night_light_warmth=max(0.0, min(1.0, float(getattr(self, "night_light_warmth", 0.35) if getattr(self, "night_light_warmth", None) is not None else 0.35))),
            scene_plate_width=1.0,
            scene_plate_height=0.24,
            scene_camera_x=0.0,
            scene_camera_y=0.0,
            scene_camera_z=scene_camera_z,
            scene_camera_target_x=scene_camera_target_x,
            scene_camera_target_y=scene_camera_target_y,
            scene_camera_target_z=0.0,
            scene_view_yaw=max(-180.0, min(180.0, float(getattr(self, "scene_view_yaw", 35.0) if getattr(self, "scene_view_yaw", None) is not None else 35.0))),
            scene_view_pitch=max(-89.0, min(89.0, float(getattr(self, "scene_view_pitch", -24.0) if getattr(self, "scene_view_pitch", None) is not None else -24.0))),
            scene_view_roll=max(-180.0, min(180.0, float(getattr(self, "scene_view_roll", 0.0) if getattr(self, "scene_view_roll", None) is not None else 0.0))),
            scene_plate_texture_enabled=bool(getattr(self, "scene_plate_texture_enabled", True)),
            traffic_headlight_strength=max(0.0, min(1.0, float(getattr(self, "traffic_headlight_strength", 0.0) or 0.0))),
            traffic_headlight_count=max(0, min(6, int(float(getattr(self, "traffic_headlight_count", 1) if getattr(self, "traffic_headlight_count", None) is not None else 1)))),
            traffic_headlight_source_x=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_source_x", -1.0) if getattr(self, "traffic_headlight_source_x", None) is not None else -1.0))),
            traffic_headlight_source_y=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_source_y", -1.0) if getattr(self, "traffic_headlight_source_y", None) is not None else -1.0))),
            traffic_headlight_target_x=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_target_x", -1.0) if getattr(self, "traffic_headlight_target_x", None) is not None else -1.0))),
            traffic_headlight_target_y=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_target_y", -1.0) if getattr(self, "traffic_headlight_target_y", None) is not None else -1.0))),
            traffic_headlight_source_world_x=max(-999.0, min(5.0, float(getattr(self, "traffic_headlight_source_world_x", -999.0) if getattr(self, "traffic_headlight_source_world_x", None) is not None else -999.0))),
            traffic_headlight_source_world_y=max(-999.0, min(5.0, float(getattr(self, "traffic_headlight_source_world_y", -999.0) if getattr(self, "traffic_headlight_source_world_y", None) is not None else -999.0))),
            traffic_headlight_source_world_z=max(0.0, min(8.0, float(getattr(self, "traffic_headlight_source_world_z", 0.95) if getattr(self, "traffic_headlight_source_world_z", None) is not None else 0.95))),
            traffic_headlight_target_world_x=max(-999.0, min(5.0, float(getattr(self, "traffic_headlight_target_world_x", -999.0) if getattr(self, "traffic_headlight_target_world_x", None) is not None else -999.0))),
            traffic_headlight_target_world_y=max(-999.0, min(5.0, float(getattr(self, "traffic_headlight_target_world_y", -999.0) if getattr(self, "traffic_headlight_target_world_y", None) is not None else -999.0))),
            traffic_headlight_target_world_z=0.0,
            traffic_headlight_1_warmth=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_1_warmth", -1.0) if getattr(self, "traffic_headlight_1_warmth", None) is not None else -1.0))),
            traffic_headlight_1_r=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_1_r", -1.0) if getattr(self, "traffic_headlight_1_r", None) is not None else -1.0))),
            traffic_headlight_1_g=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_1_g", -1.0) if getattr(self, "traffic_headlight_1_g", None) is not None else -1.0))),
            traffic_headlight_1_b=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_1_b", -1.0) if getattr(self, "traffic_headlight_1_b", None) is not None else -1.0))),
            traffic_headlight_1_cone=max(0.02, min(2.5, float(getattr(self, "traffic_headlight_1_cone", 0.45) if getattr(self, "traffic_headlight_1_cone", None) is not None else 0.45))),
            traffic_headlight_1_source_radius=max(0.0, min(2.5, float(getattr(self, "traffic_headlight_1_source_radius", 0.08) if getattr(self, "traffic_headlight_1_source_radius", None) is not None else 0.08))),
            traffic_headlight_2_strength=max(0.0, min(1.0, float(getattr(self, "traffic_headlight_2_strength", 0.0) or 0.0))),
            traffic_headlight_2_warmth=max(0.0, min(1.0, float(getattr(self, "traffic_headlight_2_warmth", 0.35) if getattr(self, "traffic_headlight_2_warmth", None) is not None else 0.35))),
            traffic_headlight_2_r=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_2_r", -1.0) if getattr(self, "traffic_headlight_2_r", None) is not None else -1.0))),
            traffic_headlight_2_g=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_2_g", -1.0) if getattr(self, "traffic_headlight_2_g", None) is not None else -1.0))),
            traffic_headlight_2_b=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_2_b", -1.0) if getattr(self, "traffic_headlight_2_b", None) is not None else -1.0))),
            traffic_headlight_2_cone=max(0.02, min(2.5, float(getattr(self, "traffic_headlight_2_cone", 0.45) if getattr(self, "traffic_headlight_2_cone", None) is not None else 0.45))),
            traffic_headlight_2_source_radius=max(0.0, min(2.5, float(getattr(self, "traffic_headlight_2_source_radius", 0.08) if getattr(self, "traffic_headlight_2_source_radius", None) is not None else 0.08))),
            traffic_headlight_2_source_x=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_2_source_x", -1.0) if getattr(self, "traffic_headlight_2_source_x", None) is not None else -1.0))),
            traffic_headlight_2_source_y=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_2_source_y", -1.0) if getattr(self, "traffic_headlight_2_source_y", None) is not None else -1.0))),
            traffic_headlight_2_target_x=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_2_target_x", -1.0) if getattr(self, "traffic_headlight_2_target_x", None) is not None else -1.0))),
            traffic_headlight_2_target_y=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_2_target_y", -1.0) if getattr(self, "traffic_headlight_2_target_y", None) is not None else -1.0))),
            traffic_headlight_2_source_world_x=max(-999.0, min(5.0, float(getattr(self, "traffic_headlight_2_source_world_x", -999.0) if getattr(self, "traffic_headlight_2_source_world_x", None) is not None else -999.0))),
            traffic_headlight_2_source_world_y=max(-999.0, min(5.0, float(getattr(self, "traffic_headlight_2_source_world_y", -999.0) if getattr(self, "traffic_headlight_2_source_world_y", None) is not None else -999.0))),
            traffic_headlight_2_source_world_z=max(0.0, min(8.0, float(getattr(self, "traffic_headlight_2_source_world_z", 0.95) if getattr(self, "traffic_headlight_2_source_world_z", None) is not None else 0.95))),
            traffic_headlight_2_target_world_x=max(-999.0, min(5.0, float(getattr(self, "traffic_headlight_2_target_world_x", -999.0) if getattr(self, "traffic_headlight_2_target_world_x", None) is not None else -999.0))),
            traffic_headlight_2_target_world_y=max(-999.0, min(5.0, float(getattr(self, "traffic_headlight_2_target_world_y", -999.0) if getattr(self, "traffic_headlight_2_target_world_y", None) is not None else -999.0))),
            traffic_headlight_2_target_world_z=0.0,
            traffic_headlight_3_strength=max(0.0, min(1.0, float(getattr(self, "traffic_headlight_3_strength", 0.0) or 0.0))),
            traffic_headlight_3_warmth=max(0.0, min(1.0, float(getattr(self, "traffic_headlight_3_warmth", 0.35) if getattr(self, "traffic_headlight_3_warmth", None) is not None else 0.35))),
            traffic_headlight_3_r=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_3_r", -1.0) if getattr(self, "traffic_headlight_3_r", None) is not None else -1.0))),
            traffic_headlight_3_g=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_3_g", -1.0) if getattr(self, "traffic_headlight_3_g", None) is not None else -1.0))),
            traffic_headlight_3_b=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_3_b", -1.0) if getattr(self, "traffic_headlight_3_b", None) is not None else -1.0))),
            traffic_headlight_3_cone=max(0.02, min(2.5, float(getattr(self, "traffic_headlight_3_cone", 0.45) if getattr(self, "traffic_headlight_3_cone", None) is not None else 0.45))),
            traffic_headlight_3_source_radius=max(0.0, min(2.5, float(getattr(self, "traffic_headlight_3_source_radius", 0.08) if getattr(self, "traffic_headlight_3_source_radius", None) is not None else 0.08))),
            traffic_headlight_3_source_x=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_3_source_x", -1.0) if getattr(self, "traffic_headlight_3_source_x", None) is not None else -1.0))),
            traffic_headlight_3_source_y=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_3_source_y", -1.0) if getattr(self, "traffic_headlight_3_source_y", None) is not None else -1.0))),
            traffic_headlight_3_target_x=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_3_target_x", -1.0) if getattr(self, "traffic_headlight_3_target_x", None) is not None else -1.0))),
            traffic_headlight_3_target_y=max(-1.0, min(1.0, float(getattr(self, "traffic_headlight_3_target_y", -1.0) if getattr(self, "traffic_headlight_3_target_y", None) is not None else -1.0))),
            traffic_headlight_3_source_world_x=max(-999.0, min(5.0, float(getattr(self, "traffic_headlight_3_source_world_x", -999.0) if getattr(self, "traffic_headlight_3_source_world_x", None) is not None else -999.0))),
            traffic_headlight_3_source_world_y=max(-999.0, min(5.0, float(getattr(self, "traffic_headlight_3_source_world_y", -999.0) if getattr(self, "traffic_headlight_3_source_world_y", None) is not None else -999.0))),
            traffic_headlight_3_source_world_z=max(0.0, min(8.0, float(getattr(self, "traffic_headlight_3_source_world_z", 0.95) if getattr(self, "traffic_headlight_3_source_world_z", None) is not None else 0.95))),
            traffic_headlight_3_target_world_x=max(-999.0, min(5.0, float(getattr(self, "traffic_headlight_3_target_world_x", -999.0) if getattr(self, "traffic_headlight_3_target_world_x", None) is not None else -999.0))),
            traffic_headlight_3_target_world_y=max(-999.0, min(5.0, float(getattr(self, "traffic_headlight_3_target_world_y", -999.0) if getattr(self, "traffic_headlight_3_target_world_y", None) is not None else -999.0))),
            traffic_headlight_3_target_world_z=0.0,
            wet_mud_gloss_strength=max(0.0, min(1.0, float(getattr(self, "wet_mud_gloss_strength", 0.0) or 0.0))),
            water_film_strength=max(0.0, min(1.0, float(getattr(self, "water_film_strength", 0.0) or 0.0))),
            water_film_unevenness=max(0.0, min(1.0, float(getattr(self, "water_film_unevenness", 0.35) if getattr(self, "water_film_unevenness", None) is not None else 0.35))),
            water_film_lens_strength=max(0.0, min(1.0, float(getattr(self, "water_film_lens_strength", 0.0) or 0.0))),
            water_film_contour_response=max(0.0, min(1.0, float(getattr(self, "water_film_contour_response", 0.55) if getattr(self, "water_film_contour_response", None) is not None else 0.55))),
            water_film_gloss_strength=max(0.0, min(1.0, float(getattr(self, "water_film_gloss_strength", 0.45) if getattr(self, "water_film_gloss_strength", None) is not None else 0.45))),
            flare_strength=max(0.0, min(1.0, float(self.flare_strength or 0.0))),
            overexposure_strength=max(0.0, min(1.0, float(self.overexposure_strength or 0.0))),
            dirt_streak_strength=max(0.0, min(1.0, float(self.dirt_streak_strength or 0.0))),
            dirt_flow_strength=max(0.0, min(1.0, float(self.dirt_flow_strength or 0.0))),
            dirt_flow_points=max(0, min(MAX_DIRT_FLOW_POINTS, int(float(self.dirt_flow_points or 0)))),
            dirt_flow_mass_min=dirt_flow_mass_min,
            dirt_flow_mass_max=dirt_flow_mass_max,
            dirt_flow_splash_scale=max(0.0, min(1.0, float(self.dirt_flow_splash_scale if self.dirt_flow_splash_scale is not None else 0.45))),
            dirt_flow_trail_length=max(0.0, min(1.0, float(self.dirt_flow_trail_length if self.dirt_flow_trail_length is not None else 0.55))),
            dirt_flow_humidity=max(0.0, min(1.0, float(self.dirt_flow_humidity if self.dirt_flow_humidity is not None else 0.45))),
            dirt_flow_stickiness=dirt_flow_stickiness,
            dirt_flow_stickiness_min=dirt_flow_stickiness_min,
            dirt_flow_stickiness_max=dirt_flow_stickiness_max,
            dirt_flow_air_angle=((float(self.dirt_flow_air_angle or 0.0) + 180.0) % 360.0) - 180.0,
            dirt_flow_wind_strength=max(0.0, min(1.0, float(self.dirt_flow_wind_strength if self.dirt_flow_wind_strength is not None else 0.45))),
            dirt_flow_gravity_angle=float(self.dirt_flow_gravity_angle if self.dirt_flow_gravity_angle is not None else 90.0) % 360.0,
            dirt_flow_gravity_strength=max(0.0, min(1.0, float(self.dirt_flow_gravity_strength if self.dirt_flow_gravity_strength is not None else 1.0))),
            dirt_flow_opacity_min=dirt_flow_opacity_min,
            dirt_flow_opacity_max=dirt_flow_opacity_max,
            dirt_flow_stop_on_dark_contour=bool(getattr(self, "dirt_flow_stop_on_dark_contour", False)),
            contour_detection_sensitivity=max(0.0, min(1.0, float(getattr(self, "contour_detection_sensitivity", 0.55) if getattr(self, "contour_detection_sensitivity", None) is not None else 0.55))),
            dark_relief_strength=max(0.0, min(3.0, float(self.dark_relief_strength or 0.0))),
            dark_relief_light_angle=float(self.dark_relief_light_angle or 0.0) % 360.0,
            relief_profile_preset=relief_profile_preset,
            relief_profile_curve=relief_profile_curve,
            relief_bounce_depth=max(1, min(4, int(float(getattr(self, "relief_bounce_depth", 1) if getattr(self, "relief_bounce_depth", None) is not None else 1)))),
            relief_bounce_strength=max(0.0, min(1.0, float(getattr(self, "relief_bounce_strength", 0.28) if getattr(self, "relief_bounce_strength", None) is not None else 0.28))),
            light_normal_strength=camera_axis_strength,
            overhang_shadow_strength=max(0.0, min(1.0, float(getattr(self, "overhang_shadow_strength", 0.0) or 0.0))),
            overhang_shadow_depth=max(0.0, min(0.60, float(getattr(self, "overhang_shadow_depth", 0.60) if getattr(self, "overhang_shadow_depth", None) is not None else 0.60))),
            overhang_shadow_skew=max(-1.0, min(1.0, float(getattr(self, "overhang_shadow_skew", 0.0) or 0.0))),
            plate_reflect_gradient_strength=max(0.0, min(2.0, float(getattr(self, "plate_reflect_gradient_strength", 0.0) or 0.0))),
            plate_reflect_glare_strength=max(0.0, min(2.0, float(getattr(self, "plate_reflect_glare_strength", 0.0) or 0.0))),
            plate_reflect_curve_strength=max(0.0, min(2.0, float(getattr(self, "plate_reflect_curve_strength", 0.0) or 0.0))),
            blur_strength=blur_strength,
            blur_enabled=bool(self.blur_enabled) or blur_strength > 0.001,
            randomness_mode=normalize_augmentation_randomness_mode(getattr(self, "randomness_mode", "realistic")),
            seed=int(self.seed or 42),
            class_name=str(self.class_name or "").strip(),
            task_target=task_target,
            manual_randomness=manual_randomness,
        )


@dataclass
class YoloObject:
    class_id: int
    bbox: list[float]
    keypoints: list[tuple[float, float, float | None]]


def augmentation_profile_from_mapping(
    payload: object,
    *,
    base: AugmentationProfile | None = None,
    target: str | None = None,
) -> AugmentationProfile:
    """Load a persisted augmentation profile without trusting stale schemas."""
    if not isinstance(payload, dict):
        return (base or AugmentationProfile(task_target=target or "")).normalized()
    raw_profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else payload
    source_profile = (base or AugmentationProfile(task_target=target or "")).normalized()
    data = asdict(source_profile)
    for key in AugmentationProfile.__dataclass_fields__:
        if key in raw_profile:
            data[key] = raw_profile[key]
    if target:
        data["task_target"] = _normalize_task_target_value(target)
    if "manual_randomness" not in data and isinstance(payload.get("manual_randomness"), dict):
        data["manual_randomness"] = payload["manual_randomness"]
    return AugmentationProfile(**data).normalized()


def is_albumentations_available() -> bool:
    return importlib.util.find_spec("albumentations") is not None


def get_albumentations_status() -> dict:
    if not is_albumentations_available():
        return {
            "available": False,
            "message": "Biblioteka Albumentations nie jest zainstalowana.",
        }
    return {
        "available": True,
        "message": "Albumentations jest dostępne.",
    }


def build_albumentations_install_command() -> list[str]:
    return [sys.executable, "-m", "pip", "install", "albumentations"]


def install_albumentations(
    line_callback: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Install Albumentations on explicit user request."""
    if is_albumentations_available():
        return True, "Albumentations jest już zainstalowane."

    cmd = build_albumentations_install_command()
    if callable(line_callback):
        line_callback("> " + subprocess.list2cmdline(cmd))

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                if callable(line_callback):
                    line_callback(str(line or "").rstrip())
        code = process.wait()
    except Exception as exc:
        return False, f"Nie udało się uruchomić instalacji Albumentations: {exc}"

    importlib.invalidate_caches()
    if code == 0 and is_albumentations_available():
        return True, "Albumentations zostało zainstalowane i jest gotowe do augmentacji."
    if code == 0:
        return False, "Instalacja pip zakończyła się, ale moduł Albumentations nadal nie jest widoczny."
    return False, f"Instalacja Albumentations zakończyła się błędem pip (kod {code})."


def _load_albumentations():
    # Albumentations checks PyPI on import unless this flag is set. The app
    # should not touch the network while preparing a local dataset.
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    return importlib.import_module("albumentations")


def _stable_preview_seed(image_path: Path, profile: AugmentationProfile) -> int:
    normalized = (profile or AugmentationProfile()).normalized()
    # Preview geometry must stay stable while the user tweaks visual effects
    # such as noise, night mode or overexposure. Dataset knobs also must not
    # visually reshuffle the preview. The seed intentionally does not include
    # image_path: switching the preview sample should carry the already tuned
    # effect layout to the next plate/photo instead of silently rerolling it.
    payload = {
        "geometry_profile": {
            "rotation_limit": normalized.rotation_limit,
            "translate_limit": normalized.translate_limit,
            "scale_limit": normalized.scale_limit,
            "seed": normalized.seed,
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def _seed_transform(transform, seed: int) -> None:
    random.seed(seed)
    try:
        if np is not None:
            np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:
        transform.set_random_seed(seed)
    except Exception:
        pass


def _apply_preview_seed(transform, image_path: Path, profile: AugmentationProfile) -> None:
    _seed_transform(transform, _stable_preview_seed(image_path, profile))


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(float(minimum), min(float(maximum), float(value)))


def _jitter_float(
    value: float,
    rng: random.Random,
    *,
    minimum: float,
    maximum: float,
    sigma_abs: float = 0.0,
    sigma_ratio: float = 0.08,
    max_delta_abs: float = 0.0,
    max_delta_ratio: float = 0.20,
    zero_stays_zero: bool = True,
) -> float:
    base = _clamp_float(value, minimum, maximum)
    if zero_stays_zero and abs(base) <= 0.000001:
        return base
    sigma = max(float(sigma_abs), abs(base) * float(sigma_ratio))
    if sigma <= 0.000001:
        return base
    max_delta = max(float(max_delta_abs), abs(base) * float(max_delta_ratio), sigma * 1.5)
    sampled = rng.gauss(base, sigma)
    sampled = _clamp_float(sampled, base - max_delta, base + max_delta)
    return _clamp_float(sampled, minimum, maximum)


def _jitter_int(
    value: int,
    rng: random.Random,
    *,
    minimum: int,
    maximum: int,
    sigma_abs: float = 1.0,
    sigma_ratio: float = 0.08,
    max_delta_ratio: float = 0.18,
    zero_stays_zero: bool = True,
) -> int:
    base = max(int(minimum), min(int(maximum), int(value or 0)))
    if zero_stays_zero and base <= 0:
        return base
    sigma = max(float(sigma_abs), abs(float(base)) * float(sigma_ratio))
    max_delta = max(1.0, abs(float(base)) * float(max_delta_ratio), sigma * 1.5)
    sampled = rng.gauss(float(base), sigma)
    sampled = _clamp_float(sampled, float(base) - max_delta, float(base) + max_delta)
    return max(int(minimum), min(int(maximum), int(round(sampled))))


def _jitter_optional_unit(value: float, rng: random.Random, *, allow_negative_sentinel: bool = True) -> float:
    base = float(value if value is not None else -1.0)
    if allow_negative_sentinel and base < 0.0:
        return base
    return _jitter_float(
        base,
        rng,
        minimum=0.0,
        maximum=1.0,
        sigma_abs=0.015,
        sigma_ratio=0.06,
        max_delta_abs=0.04,
        max_delta_ratio=0.12,
        zero_stays_zero=False,
    )


def _jitter_coordinate(value: float, rng: random.Random) -> float:
    base = float(value if value is not None else -1.0)
    if base < 0.0:
        return base
    return _jitter_float(
        base,
        rng,
        minimum=0.0,
        maximum=1.0,
        sigma_abs=0.012,
        sigma_ratio=0.025,
        max_delta_abs=0.035,
        max_delta_ratio=0.06,
        zero_stays_zero=False,
    )


def _jitter_world_coordinate(value: float, rng: random.Random, *, minimum: float = -5.0, maximum: float = 5.0) -> float:
    base = float(value if value is not None else -999.0)
    if base <= -998.0:
        return base
    return _jitter_float(
        base,
        rng,
        minimum=minimum,
        maximum=maximum,
        sigma_abs=0.030,
        sigma_ratio=0.035,
        max_delta_abs=0.090,
        max_delta_ratio=0.08,
        zero_stays_zero=False,
    )


def _jitter_range_pair(
    min_value: float,
    max_value: float,
    rng: random.Random,
    *,
    minimum: float,
    maximum: float,
    sigma_abs: float = 0.015,
    sigma_ratio: float = 0.06,
) -> tuple[float, float]:
    lower = _clamp_float(min_value, minimum, maximum)
    upper = _clamp_float(max_value, minimum, maximum)
    if lower > upper:
        lower, upper = upper, lower
    if abs(lower) <= 0.000001 and abs(upper) <= 0.000001:
        return lower, upper

    center = (lower + upper) / 2.0
    width = max(0.0, upper - lower)
    center = _jitter_float(
        center,
        rng,
        minimum=minimum,
        maximum=maximum,
        sigma_abs=sigma_abs,
        sigma_ratio=sigma_ratio,
        max_delta_abs=sigma_abs * 2.5,
        max_delta_ratio=0.10,
        zero_stays_zero=False,
    )
    width = _jitter_float(
        width,
        rng,
        minimum=0.0,
        maximum=maximum - minimum,
        sigma_abs=sigma_abs * 0.75,
        sigma_ratio=0.05,
        max_delta_abs=sigma_abs * 2.0,
        max_delta_ratio=0.12,
        zero_stays_zero=False,
    )
    lower = _clamp_float(center - width / 2.0, minimum, maximum)
    upper = _clamp_float(center + width / 2.0, minimum, maximum)
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper


def _augmentation_randomness_extra_scale(mode: str) -> float:
    mode = normalize_augmentation_randomness_mode(mode)
    if mode == "realistic":
        return 1.0
    if mode == "wide":
        return 1.85
    return 0.0


def _active_headlight_strength(profile: AugmentationProfile, index: int) -> float:
    if index == 1:
        return float(getattr(profile, "traffic_headlight_strength", 0.0) or 0.0)
    return float(getattr(profile, f"traffic_headlight_{index}_strength", 0.0) or 0.0)


def _headlight_default_geometry(index: int) -> tuple[float, float, float, float]:
    if index == 2:
        return 0.86, 0.90, 0.58, 0.48
    if index == 3:
        return 0.50, 0.98, 0.50, 0.36
    return 0.14, 0.91, 0.42, 0.44


def _headlight_field_name(index: int, suffix: str) -> str:
    if index == 1:
        mapping = {
            "strength": "traffic_headlight_strength",
            "source_x": "traffic_headlight_source_x",
            "source_y": "traffic_headlight_source_y",
            "target_x": "traffic_headlight_target_x",
            "target_y": "traffic_headlight_target_y",
        }
        return mapping.get(suffix, f"traffic_headlight_1_{suffix}")
    return f"traffic_headlight_{index}_{suffix}"


def _scene_plate_dimensions(profile: AugmentationProfile, height: int, width: int) -> tuple[float, float]:
    plate_width = 1.0
    default_height = plate_width * max(0.08, min(1.2, float(height) / max(1.0, float(width))))
    plate_height = max(0.08, min(1.2, default_height))
    return plate_width, plate_height


def _scene_norm_to_plate_xy(profile: AugmentationProfile, norm_x: float, norm_y: float, height: int, width: int, *, scale: float = 1.0) -> tuple[float, float]:
    plate_width, plate_height = _scene_plate_dimensions(profile, height, width)
    return (float(norm_x) - 0.5) * plate_width * scale, (0.5 - float(norm_y)) * plate_height * scale


def _plate_curvature_gain(profile: AugmentationProfile | None) -> float:
    try:
        strength = max(0.0, min(2.0, float(getattr(profile, "plate_reflect_curve_strength", 0.0) or 0.0)))
        return min(1.0, strength / 2.0)
    except Exception:
        return 0.0


def _plate_curvature_amplitude(profile: AugmentationProfile | None, plate_width: float) -> float:
    gain = _plate_curvature_gain(profile)
    if gain <= 0.001:
        return 0.0
    return max(0.0, float(plate_width)) * (0.025 + 0.085 * gain) * gain


def _plate_curvature_z_at_x(profile: AugmentationProfile | None, height: int, width: int, world_x: float) -> float:
    try:
        plate_width, _plate_height = _scene_plate_dimensions(profile or AugmentationProfile(), height, width)
        amplitude = _plate_curvature_amplitude(profile, plate_width)
        if amplitude <= 0.0001:
            return 0.0
        half_width = max(0.0001, plate_width * 0.5)
        x_norm = max(-1.0, min(1.0, float(world_x) / half_width))
        return float(amplitude * max(0.0, 1.0 - x_norm * x_norm))
    except Exception:
        return 0.0


def _plate_curvature_surface(profile: AugmentationProfile | None, height: int, width: int) -> dict:
    """Return the cylindrical base surface of a horizontally bent plate."""
    if np is None or height <= 1 or width <= 1:
        return {}
    try:
        plate_width, _plate_height = _scene_plate_dimensions(profile or AugmentationProfile(), height, width)
        gain = _plate_curvature_gain(profile)
        amplitude = _plate_curvature_amplitude(profile, plate_width)
        if gain <= 0.001 or amplitude <= 0.0001:
            return {}
        _yy, xx = np.mgrid[0:height, 0:width].astype("float32")
        world_x = (xx / max(1.0, float(width - 1)) - 0.5) * float(plate_width)
        half_width = max(0.0001, float(plate_width) * 0.5)
        x_norm = np.clip(world_x / half_width, -1.0, 1.0)
        z = np.clip(amplitude * (1.0 - x_norm * x_norm), 0.0, None).astype("float32")
        dzdx = (-2.0 * amplitude * x_norm / half_width).astype("float32")
        normal_len = np.sqrt(dzdx * dzdx + 1.0).astype("float32")
        normal_x = np.clip(-dzdx / np.maximum(normal_len, 0.0001), -1.0, 1.0).astype("float32")
        normal_y = np.zeros_like(normal_x, dtype="float32")
        normal_z = np.clip(1.0 / np.maximum(normal_len, 0.0001), 0.0, 1.0).astype("float32")
        return {
            "z": z,
            "normal_x": normal_x,
            "normal_y": normal_y,
            "normal_z": normal_z,
            "normal_energy": np.clip(np.abs(normal_x), 0.0, 1.0).astype("float32"),
            "gradient_x": np.clip(dzdx, -1.0, 1.0).astype("float32"),
            "gradient_y": np.zeros_like(normal_x, dtype="float32"),
            "gain": float(gain),
            "amplitude": float(amplitude),
        }
    except Exception:
        return {}


def _scene_camera_position(profile: AugmentationProfile) -> np.ndarray:
    return np.array(
        [
            0.0,
            0.0,
            max(0.35, min(4.0, float(getattr(profile, "scene_camera_z", 1.65) if getattr(profile, "scene_camera_z", None) is not None else 1.65))),
        ],
        dtype="float32",
    )


def _scene_camera_target(profile: AugmentationProfile, height: int, width: int) -> np.ndarray:
    return np.array([0.0, 0.0, 0.0], dtype="float32")


def _scene_headlight_world_name(index: int, endpoint: str, axis: str) -> str:
    if index == 1:
        return f"traffic_headlight_{endpoint}_world_{axis}"
    return f"traffic_headlight_{index}_{endpoint}_world_{axis}"


def _scene_headlight_point(
    profile: AugmentationProfile,
    index: int,
    endpoint: str,
    height: int,
    width: int,
    *,
    default_norm: tuple[float, float],
    default_z: float,
    scale: float,
) -> np.ndarray:
    explicit_x = float(getattr(profile, _scene_headlight_world_name(index, endpoint, "x"), -999.0) if getattr(profile, _scene_headlight_world_name(index, endpoint, "x"), None) is not None else -999.0)
    explicit_y = float(getattr(profile, _scene_headlight_world_name(index, endpoint, "y"), -999.0) if getattr(profile, _scene_headlight_world_name(index, endpoint, "y"), None) is not None else -999.0)
    explicit_z = float(getattr(profile, _scene_headlight_world_name(index, endpoint, "z"), default_z) if getattr(profile, _scene_headlight_world_name(index, endpoint, "z"), None) is not None else default_z)
    if str(endpoint or "").lower() == "target":
        explicit_z = 0.0
    if explicit_x > -998.0 and explicit_y > -998.0:
        if str(endpoint or "").lower() == "target":
            plate_width, plate_height = _scene_plate_dimensions(profile, height, width)
            explicit_x = max(-plate_width / 2.0, min(plate_width / 2.0, explicit_x))
            explicit_y = max(-plate_height / 2.0, min(plate_height / 2.0, explicit_y))
        return np.array([explicit_x, explicit_y, explicit_z], dtype="float32")
    x, y = _scene_norm_to_plate_xy(profile, default_norm[0], default_norm[1], height, width, scale=scale)
    return np.array([x, y, explicit_z], dtype="float32")


def _scene_headlight_field(profile: AugmentationProfile, index: int, height: int, width: int) -> dict | None:
    if np is None or height <= 1 or width <= 1:
        return None
    try:
        strength = max(0.0, min(1.0, _active_headlight_strength(profile, index)))
        if strength <= 0.001:
            return None
        source_x, source_y, target_x, target_y = _headlight_default_geometry(index)
        try:
            sx = float(getattr(profile, _headlight_field_name(index, "source_x"), source_x))
            sy = float(getattr(profile, _headlight_field_name(index, "source_y"), source_y))
            tx = float(getattr(profile, _headlight_field_name(index, "target_x"), target_x))
            ty = float(getattr(profile, _headlight_field_name(index, "target_y"), target_y))
        except Exception:
            sx, sy, tx, ty = source_x, source_y, target_x, target_y
        if not (0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0):
            sx, sy = source_x, source_y
        if not (0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0):
            tx, ty = target_x, target_y

        source = _scene_headlight_point(profile, index, "source", height, width, default_norm=(sx, sy), default_z=0.95, scale=1.55)
        target = _scene_headlight_point(profile, index, "target", height, width, default_norm=(tx, ty), default_z=0.0, scale=1.0)
        target = target.copy()
        target[2] = _plate_curvature_z_at_x(profile, height, width, float(target[0]))
        plate_width, plate_height = _scene_plate_dimensions(profile, height, width)
        yy, xx = np.mgrid[0:height, 0:width].astype("float32")
        px = (xx / max(1.0, float(width - 1)) - 0.5) * plate_width
        py = (0.5 - yy / max(1.0, float(height - 1))) * plate_height
        plate_surface = _plate_curvature_surface(profile, height, width)
        if plate_surface:
            pz = np.asarray(plate_surface.get("z"), dtype="float32")
            normal_x = np.asarray(plate_surface.get("normal_x"), dtype="float32")
            normal_y = np.asarray(plate_surface.get("normal_y"), dtype="float32")
            normal_z = np.asarray(plate_surface.get("normal_z"), dtype="float32")
            if pz.shape != (height, width) or normal_x.shape != (height, width) or normal_y.shape != (height, width) or normal_z.shape != (height, width):
                pz = np.zeros_like(px, dtype="float32")
                normal_x = np.zeros_like(px, dtype="float32")
                normal_y = np.zeros_like(px, dtype="float32")
                normal_z = np.ones_like(px, dtype="float32")
        else:
            pz = np.zeros_like(px, dtype="float32")
            normal_x = np.zeros_like(px, dtype="float32")
            normal_y = np.zeros_like(px, dtype="float32")
            normal_z = np.ones_like(px, dtype="float32")
        source_to_point_x = px - float(source[0])
        source_to_point_y = py - float(source[1])
        source_to_point_z = pz - float(source[2])
        distance = np.sqrt(source_to_point_x * source_to_point_x + source_to_point_y * source_to_point_y + source_to_point_z * source_to_point_z)
        distance = np.maximum(distance, 0.001)
        to_point_x = source_to_point_x / distance
        to_point_y = source_to_point_y / distance
        to_point_z = source_to_point_z / distance
        axis = target - source
        axis_norm = float(np.linalg.norm(axis) or 1.0)
        axis = axis / axis_norm
        spot_cos = np.clip(to_point_x * float(axis[0]) + to_point_y * float(axis[1]) + to_point_z * float(axis[2]), -1.0, 1.0)
        cone = max(0.02, min(2.5, float(getattr(profile, f"traffic_headlight_{index}_cone", 0.45) if index > 1 else getattr(profile, "traffic_headlight_1_cone", 0.45))))
        source_radius = max(
            0.0,
            min(
                2.5,
                float(
                    getattr(profile, f"traffic_headlight_{index}_source_radius", 0.08)
                    if index > 1
                    else getattr(profile, "traffic_headlight_1_source_radius", 0.08)
                ),
            ),
        )
        cone_angle = math.radians(8.0 + 34.0 * min(1.0, cone / 2.5))
        inner_cos = math.cos(cone_angle * 0.52)
        outer_cos = math.cos(cone_angle)
        lateral_axis = math.hypot(float(axis[0]), float(axis[1]))
        source_height = max(0.001, abs(float(source[2]) - float(target[2])))
        near_normal_axis = lateral_axis <= max(0.010, source_height * 0.035)
        if near_normal_axis:
            radial = np.sqrt(
                (px - float(target[0])) * (px - float(target[0]))
                + (py - float(target[1])) * (py - float(target[1]))
                + ((pz - float(target[2])) * 0.85) * ((pz - float(target[2])) * 0.85)
            )
            outer_radius = max(0.0025, source_height * math.tan(cone_angle))
            inner_radius = max(0.0010, source_height * math.tan(cone_angle * 0.52))
            spot = np.clip((outer_radius - radial) / max(0.0001, outer_radius - inner_radius), 0.0, 1.0)
        else:
            spot = np.clip((spot_cos - outer_cos) / max(0.0001, inner_cos - outer_cos), 0.0, 1.0)
        spot = spot * spot * (3.0 - 2.0 * spot)
        source_radius_gain = min(1.0, source_radius / 0.45) if source_radius > 0.001 else 0.0
        near_source_light = None
        if source_radius_gain > 0.001:
            # The apparent diameter of a light source matters only in the near
            # field. A distant reflector should be controlled mostly by the
            # target cone, not by the physical size of the lamp face.
            close_reference = max(0.045, 0.18 + 0.62 * min(1.0, source_radius / 2.5))
            near_source_gate = 1.0 / (1.0 + (source_height / close_reference) ** 3.2)
            if near_source_gate > 0.002:
                center_blend = min(0.78, 0.72 * near_source_gate)
                source_center_x = float(target[0]) * (1.0 - center_blend) + float(source[0]) * center_blend
                source_center_y = float(target[1]) * (1.0 - center_blend) + float(source[1]) * center_blend
                source_center_z = float(target[2]) * (1.0 - center_blend) + float(source[2]) * center_blend
                source_radial = np.sqrt(
                    (px - source_center_x) * (px - source_center_x)
                    + (py - source_center_y) * (py - source_center_y)
                    + ((pz - source_center_z) * 0.18) * ((pz - source_center_z) * 0.18)
                )
                source_sigma = max(0.012, source_radius * (0.20 + 0.58 * near_source_gate) + 0.014)
                source_disk = np.exp(-((source_radial / source_sigma) ** 2) * 0.5).astype("float32")
                source_disk = cv2.GaussianBlur(
                    source_disk,
                    (0, 0),
                    sigmaX=0.45 + 1.20 * near_source_gate,
                    sigmaY=0.45 + 1.20 * near_source_gate,
                )
                near_source_light = np.clip(
                    source_disk * near_source_gate * source_radius_gain * strength,
                    0.0,
                    1.0,
                )
        surface_to_light_x = -to_point_x
        surface_to_light_y = -to_point_y
        surface_to_light_z = -to_point_z
        incidence = np.clip(
            surface_to_light_x * normal_x
            + surface_to_light_y * normal_y
            + surface_to_light_z * normal_z,
            0.0,
            1.0,
        )
        camera = _scene_camera_position(profile)
        camera_x = float(camera[0]) - px
        camera_y = float(camera[1]) - py
        camera_z = float(camera[2]) - pz
        camera_distance = np.sqrt(camera_x * camera_x + camera_y * camera_y + camera_z * camera_z)
        camera_distance = np.maximum(camera_distance, 0.001)
        view_x = camera_x / camera_distance
        view_y = camera_y / camera_distance
        view_z = camera_z / camera_distance
        camera_axis = _scene_camera_target(profile, height, width) - camera
        camera_axis_norm = float(np.linalg.norm(camera_axis) or 1.0)
        camera_axis = camera_axis / camera_axis_norm
        camera_to_point_x = -view_x
        camera_to_point_y = -view_y
        camera_to_point_z = -view_z
        optical_cos = np.clip(
            camera_to_point_x * float(camera_axis[0])
            + camera_to_point_y * float(camera_axis[1])
            + camera_to_point_z * float(camera_axis[2]),
            0.0,
            1.0,
        )
        optical_gate = np.clip((optical_cos - 0.55) / 0.45, 0.0, 1.0)
        retro_alignment = np.clip(surface_to_light_x * view_x + surface_to_light_y * view_y + surface_to_light_z * view_z, 0.0, 1.0)
        attenuation = 1.0 / (1.0 + distance * distance * 0.86)
        view_gate = 1.0 if near_normal_axis else (0.58 + 0.42 * retro_alignment) * (0.72 + 0.28 * optical_gate)
        field = np.clip(
            strength
            * spot
            * attenuation
            * (0.18 + 0.82 * incidence)
            * view_gate,
            0.0,
            1.0,
        )
        if near_source_light is not None:
            field = np.clip(
                field + near_source_light * (0.20 + 0.80 * incidence) * (1.0 - field * 0.35),
                0.0,
                1.0,
            )
        hotspot_gate = 1.0 if near_normal_axis else (0.78 + 0.22 * optical_gate)
        hotspot = np.clip(field * (0.35 + 0.85 * spot * incidence) * hotspot_gate, 0.0, 1.0)
        if near_source_light is not None:
            hotspot = np.clip(
                np.maximum(hotspot, near_source_light * (0.28 + 0.52 * incidence) * (0.70 + 0.30 * optical_gate)),
                0.0,
                1.0,
            )
        half_x = surface_to_light_x + view_x
        half_y = surface_to_light_y + view_y
        half_z = surface_to_light_z + view_z
        half_norm = np.sqrt(half_x * half_x + half_y * half_y + half_z * half_z)
        half_norm = np.maximum(half_norm, 0.001)
        half_x = half_x / half_norm
        half_y = half_y / half_norm
        half_z = half_z / half_norm
        roughness = max(0.06, min(1.0, 0.18 + 0.42 * min(1.0, source_radius / 2.5) + 0.18 * min(1.0, cone / 2.5)))
        specular_power = 10.0 + 58.0 * (1.0 - roughness)
        specular_alignment = np.clip(half_x * normal_x + half_y * normal_y + half_z * normal_z, 0.0, 1.0)
        specular = np.clip(
            (specular_alignment ** specular_power)
            * spot
            * attenuation
            * strength
            * (0.30 + 0.70 * incidence)
            * (0.42 + 0.58 * optical_gate),
            0.0,
            1.0,
        )
        if near_source_light is not None:
            specular = np.clip(
                specular + near_source_light * (specular_alignment ** max(6.0, specular_power * 0.42)) * (0.16 + 0.44 * incidence),
                0.0,
                1.0,
            )
        color_bgr = np.array(_profile_headlight_bgr(profile, index), dtype="float32")
        return {
            "field": field.astype("float32"),
            "hotspot": hotspot.astype("float32"),
            "specular": specular.astype("float32"),
            "color_bgr": color_bgr,
            "source": source,
            "target": target,
            "axis": axis.astype("float32"),
            # Pixel effects work in image space: X grows right, Y grows down.
            # The scene model stores Y upward, so convert the travel vector here
            # and keep the rest of the renderer from silently flipping lights.
            "ray_x": to_point_x.astype("float32"),
            "ray_y": (-to_point_y).astype("float32"),
            "direction_x": surface_to_light_x.astype("float32"),
            "direction_y": to_point_y.astype("float32"),
            "incidence": incidence.astype("float32"),
            "view_z": view_z.astype("float32"),
        }
    except Exception:
        return None


def _combined_scene_headlight_fields(profile: AugmentationProfile, height: int, width: int) -> dict | None:
    if np is None or height <= 1 or width <= 1:
        return None
    try:
        field = np.zeros((height, width), dtype="float32")
        hotspot = np.zeros((height, width), dtype="float32")
        specular = np.zeros((height, width), dtype="float32")
        weight = np.zeros((height, width), dtype="float32")
        color_accum = np.zeros((height, width, 3), dtype="float32")
        color_weight = np.zeros((height, width), dtype="float32")
        ray_x = np.zeros((height, width), dtype="float32")
        ray_y = np.zeros((height, width), dtype="float32")
        incidence = np.zeros((height, width), dtype="float32")
        for index in (1, 2, 3):
            scene = _scene_headlight_field(profile, index, height, width)
            if not scene:
                continue
            local_field = np.clip(np.asarray(scene.get("field"), dtype="float32"), 0.0, 1.0)
            local_hotspot = np.clip(np.asarray(scene.get("hotspot"), dtype="float32"), 0.0, 1.0)
            if local_field.shape != (height, width):
                continue
            local_weight = np.maximum(local_field, local_hotspot * 0.55)
            field = np.maximum(field, local_field)
            hotspot = np.maximum(hotspot, local_hotspot)
            local_specular = np.clip(np.asarray(scene.get("specular"), dtype="float32"), 0.0, 1.0)
            if local_specular.shape == (height, width):
                specular = np.maximum(specular, local_specular)
            else:
                local_specular = np.zeros((height, width), dtype="float32")
            local_color = np.asarray(scene.get("color_bgr"), dtype="float32")
            if local_color.shape == (3,):
                local_color_source = np.clip(local_field * 0.72 + local_hotspot * 0.50 + local_specular * 0.82, 0.0, 1.0)
                color_accum += local_color_source[:, :, None] * local_color
                color_weight += local_color_source
            weight += local_weight
            ray_x += np.asarray(scene.get("ray_x"), dtype="float32") * local_weight
            ray_y += np.asarray(scene.get("ray_y"), dtype="float32") * local_weight
            incidence += np.asarray(scene.get("incidence"), dtype="float32") * local_weight
        if float(np.max(field) or 0.0) <= 0.001:
            return None
        safe_weight = np.maximum(weight, 0.0001)
        fallback_color = np.array(_profile_headlight_bgr(profile, 1), dtype="float32")
        color = color_accum / np.maximum(color_weight[:, :, None], 0.0001)
        color = np.where(color_weight[:, :, None] > 0.001, color, fallback_color)
        return {
            "field": np.clip(field, 0.0, 1.0).astype("float32"),
            "hotspot": np.clip(hotspot, 0.0, 1.0).astype("float32"),
            "specular": np.clip(specular, 0.0, 1.0).astype("float32"),
            "color_accum": color_accum.astype("float32"),
            "color": np.clip(color, 0.0, 255.0).astype("float32"),
            "color_weight": np.clip(color_weight, 0.0, 1.0).astype("float32"),
            "color_weight_raw": color_weight.astype("float32"),
            "ray_x": (ray_x / safe_weight).astype("float32"),
            "ray_y": (ray_y / safe_weight).astype("float32"),
            "incidence": np.clip(incidence / safe_weight, 0.0, 1.0).astype("float32"),
        }
    except Exception:
        return None


def _average_scene_light_direction(scene: dict | None) -> tuple[float, float, float] | None:
    """Return weighted XY light travel direction from a scene light field."""
    if np is None or not scene:
        return None
    try:
        field = np.clip(np.asarray(scene.get("field"), dtype="float32"), 0.0, 1.0)
        hotspot = np.clip(np.asarray(scene.get("hotspot"), dtype="float32"), 0.0, 1.0)
        ray_x = np.asarray(scene.get("ray_x"), dtype="float32")
        ray_y = np.asarray(scene.get("ray_y"), dtype="float32")
        if field.shape != ray_x.shape or field.shape != ray_y.shape:
            return None
        weight = np.clip(field + hotspot * 0.65, 0.0, 1.0)
        weight_sum = float(np.sum(weight) or 0.0)
        if weight_sum <= 0.0001:
            return None
        direction_x = float(np.sum(ray_x * weight) / weight_sum)
        direction_y = float(np.sum(ray_y * weight) / weight_sum)
        length = math.hypot(direction_x, direction_y)
        if length <= 0.0001:
            return None
        strength = max(float(np.max(field) or 0.0), float(np.max(hotspot) or 0.0))
        return direction_x / length, direction_y / length, max(0.0, min(1.0, strength))
    except Exception:
        return None


def _scene_or_fallback_light_direction(profile: AugmentationProfile, height: int, width: int) -> tuple[float, float, float]:
    scene = _combined_scene_headlight_fields(profile, height, width)
    averaged = _average_scene_light_direction(scene)
    if averaged is not None:
        return averaged
    angle = math.radians(_effective_light_angle_degrees(profile))
    strength = max(
        float(getattr(profile, "night_light_strength", 0.0) or 0.0),
        float(getattr(profile, "traffic_headlight_strength", 0.0) or 0.0),
        float(getattr(profile, "traffic_headlight_2_strength", 0.0) or 0.0),
        float(getattr(profile, "traffic_headlight_3_strength", 0.0) or 0.0),
        0.0,
    )
    return math.cos(angle), math.sin(angle), max(0.0, min(1.0, strength))


def _active_relief_light_context(profile: AugmentationProfile, height: int, width: int) -> dict | None:
    """Return explicit lighting for material relief, or None when relief is only geometry."""
    if np is None:
        return None
    try:
        base_strength = max(0.0, min(1.0, float(getattr(profile, "night_light_strength", 0.0) or 0.0)))
        headlight_samples: list[dict[str, float]] = []
        for index in (1, 2, 3):
            strength = max(0.0, min(1.0, _active_headlight_strength(profile, index)))
            if strength <= 0.001:
                continue
            source_x, source_y, target_x, target_y = _headlight_default_geometry(index)
            try:
                sx = float(getattr(profile, _headlight_field_name(index, "source_x"), source_x))
                sy = float(getattr(profile, _headlight_field_name(index, "source_y"), source_y))
                tx = float(getattr(profile, _headlight_field_name(index, "target_x"), target_x))
                ty = float(getattr(profile, _headlight_field_name(index, "target_y"), target_y))
            except Exception:
                sx, sy, tx, ty = source_x, source_y, target_x, target_y
            if not (0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0):
                sx, sy = source_x, source_y
            if not (0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0):
                tx, ty = target_x, target_y
            source = _scene_headlight_point(
                profile,
                index,
                "source",
                height,
                width,
                default_norm=(sx, sy),
                default_z=0.95,
                scale=1.55,
            )
            target = _scene_headlight_point(
                profile,
                index,
                "target",
                height,
                width,
                default_norm=(tx, ty),
                default_z=0.0,
                scale=1.0,
            )
            target = target.copy()
            target[2] = _plate_curvature_z_at_x(profile, height, width, float(target[0]))
            try:
                cone = max(
                    0.02,
                    min(
                        2.5,
                        float(
                            getattr(profile, f"traffic_headlight_{index}_cone", 0.45)
                            if index > 1
                            else getattr(profile, "traffic_headlight_1_cone", 0.45)
                        ),
                    ),
                )
            except Exception:
                cone = 0.45
            try:
                source_radius = max(
                    0.0,
                    min(
                        2.5,
                        float(
                            getattr(profile, f"traffic_headlight_{index}_source_radius", 0.08)
                            if index > 1
                            else getattr(profile, "traffic_headlight_1_source_radius", 0.08)
                        ),
                    ),
                )
            except Exception:
                source_radius = 0.08
            source_z = max(0.035, float(source[2]) - float(target[2]))
            lateral = math.hypot(float(source[0] - target[0]), float(source[1] - target[1]))
            headlight_samples.append(
                {
                    "strength": strength,
                    "source_z": source_z,
                    "lateral": lateral,
                    "cone": cone,
                    "source_radius": source_radius,
                    "color_bgr": _profile_headlight_bgr(profile, index),
                }
            )

        total_headlight_strength = sum(sample["strength"] for sample in headlight_samples)
        light_strength = max(base_strength, min(1.0, total_headlight_strength))
        if light_strength <= 0.001:
            return None

        scene = _combined_scene_headlight_fields(profile, height, width)
        averaged_scene = _average_scene_light_direction(scene)
        if averaged_scene is not None:
            light_x, light_y, scene_strength = averaged_scene
        else:
            light_x, light_y, scene_strength = _scene_or_fallback_light_direction(profile, height, width)
        direction_len = math.hypot(float(light_x), float(light_y))
        if direction_len <= 0.0001:
            angle = math.radians(_effective_light_angle_degrees(profile))
            light_x, light_y = math.cos(angle), math.sin(angle)
        else:
            light_x, light_y = float(light_x) / direction_len, float(light_y) / direction_len
        light_strength = max(light_strength, float(scene_strength or 0.0))

        if headlight_samples:
            weighted_z = sum(sample["strength"] * sample["source_z"] for sample in headlight_samples) / max(0.0001, total_headlight_strength)
            weighted_lateral = sum(sample["strength"] * sample["lateral"] for sample in headlight_samples) / max(0.0001, total_headlight_strength)
            weighted_cone = sum(sample["strength"] * sample["cone"] for sample in headlight_samples) / max(0.0001, total_headlight_strength)
            weighted_source_radius = sum(sample["strength"] * sample["source_radius"] for sample in headlight_samples) / max(0.0001, total_headlight_strength)
            weighted_color = tuple(
                sum(sample["strength"] * float(sample["color_bgr"][channel]) for sample in headlight_samples)
                / max(0.0001, total_headlight_strength)
                for channel in range(3)
            )
            cone_span = max(0.02, weighted_cone - weighted_source_radius)
            perspective_gain = (weighted_lateral + 0.10 + cone_span * 0.16) / max(0.055, weighted_z)
            shadow_z_gain = max(0.42, min(7.2, 0.42 + perspective_gain * (1.0 + 0.16 * min(1.0, cone_span))))
            shadow_blur_gain = max(0.55, min(2.45, 0.70 + weighted_source_radius * 0.36 + weighted_cone * 0.13 + weighted_z * 0.07))
            shadow_crispness = max(0.18, min(0.88, 0.76 - weighted_source_radius * 0.16 - weighted_cone * 0.055 + min(1.2, perspective_gain) * 0.10))
        else:
            # Base light behaves like a distant, softer source.
            weighted_z = 1.45
            weighted_color = _profile_headlight_bgr(profile, 1, _profile_float(profile, "night_light_warmth", 0.35))
            shadow_z_gain = 0.72
            shadow_blur_gain = 1.28
            shadow_crispness = 0.28
        light_gate = None
        light_color_field = None
        if headlight_samples and scene and cv2 is not None:
            try:
                field = np.clip(np.asarray(scene.get("field"), dtype="float32"), 0.0, 1.0)
                hotspot = np.clip(np.asarray(scene.get("hotspot"), dtype="float32"), 0.0, 1.0)
                if field.shape == (height, width) and hotspot.shape == (height, width):
                    raw_gate = np.clip(field * 0.78 + hotspot * 0.58, 0.0, 1.0)
                    peak = max(float(np.percentile(raw_gate, 98.0) or 0.0), float(raw_gate.max() or 0.0) * 0.55)
                    if peak > 0.001:
                        normalized_gate = np.clip(raw_gate / max(0.035, peak), 0.0, 1.0)
                        light_gate = np.clip((normalized_gate - 0.055) / 0.64, 0.0, 1.0)
                        light_gate = cv2.GaussianBlur(light_gate, (0, 0), sigmaX=0.85, sigmaY=0.85)
                        light_gate = np.clip(light_gate, 0.0, 1.0).astype("float32")
                        raw_color = np.asarray(scene.get("color"), dtype="float32")
                        if raw_color.shape == (height, width, 3):
                            light_color_field = np.clip(raw_color, 0.0, 255.0).astype("float32")
            except Exception:
                light_gate = None
                light_color_field = None
        return {
            "x": light_x,
            "y": light_y,
            "strength": max(0.0, min(1.0, light_strength)),
            "source_z": weighted_z,
            "color_bgr": tuple(float(value) for value in weighted_color),
            "color_field": light_color_field,
            "shadow_z_gain": shadow_z_gain,
            "shadow_blur_gain": shadow_blur_gain,
            "shadow_crispness": shadow_crispness,
            "light_gate": light_gate,
        }
    except Exception:
        return None


def _effective_light_angle_degrees(profile: AugmentationProfile, fallback: float | None = None) -> float:
    """Return the scene light direction, preferring active R1/R2/R3 geometry."""
    try:
        fallback_angle = (
            float(getattr(profile, "dark_relief_light_angle", 135.0) if fallback is None else fallback) % 360.0
        )
    except Exception:
        fallback_angle = 135.0

    vector_x = 0.0
    vector_y = 0.0
    strongest_angle = fallback_angle
    strongest_weight = 0.0
    for index in (1, 2, 3):
        try:
            weight = max(0.0, min(1.0, _active_headlight_strength(profile, index)))
        except Exception:
            weight = 0.0
        if weight <= 0.001:
            continue

        default_source_x, default_source_y, default_target_x, default_target_y = _headlight_default_geometry(index)
        try:
            source_x = float(getattr(profile, _headlight_field_name(index, "source_x"), default_source_x))
            source_y = float(getattr(profile, _headlight_field_name(index, "source_y"), default_source_y))
            target_x = float(getattr(profile, _headlight_field_name(index, "target_x"), default_target_x))
            target_y = float(getattr(profile, _headlight_field_name(index, "target_y"), default_target_y))
        except Exception:
            source_x, source_y, target_x, target_y = (
                default_source_x,
                default_source_y,
                default_target_x,
                default_target_y,
            )
        if not (0.0 <= source_x <= 1.0 and 0.0 <= source_y <= 1.0):
            source_x, source_y = default_source_x, default_source_y
        if not (0.0 <= target_x <= 1.0 and 0.0 <= target_y <= 1.0):
            target_x, target_y = default_target_x, default_target_y

        try:
            source = _scene_headlight_point(
                profile,
                index,
                "source",
                100,
                420,
                default_norm=(source_x, source_y),
                default_z=0.95,
                scale=1.55,
            )
            target = _scene_headlight_point(
                profile,
                index,
                "target",
                100,
                420,
                default_norm=(target_x, target_y),
                default_z=0.0,
                scale=1.0,
            )
            direction_x = float(target[0]) - float(source[0])
            direction_y = float(target[1]) - float(source[1])
        except Exception:
            direction_x = target_x - source_x
            direction_y = target_y - source_y
        length = math.hypot(direction_x, direction_y)
        if length <= 0.001:
            continue
        direction_x /= length
        direction_y /= length
        angle = math.degrees(math.atan2(-direction_y, direction_x)) % 360.0
        vector_x += direction_x * weight
        vector_y += direction_y * weight
        if weight > strongest_weight:
            strongest_weight = weight
            strongest_angle = angle

    combined_length = math.hypot(vector_x, vector_y)
    if combined_length <= 0.001:
        return strongest_angle
    return math.degrees(math.atan2(-vector_y, vector_x)) % 360.0


def _contour_detection_sensitivity(profile: AugmentationProfile | None = None) -> float:
    try:
        value = getattr(profile, "contour_detection_sensitivity", 0.55) if profile is not None else 0.55
        if value is None:
            value = 0.55
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.55


def _build_symbol_contour_height_map(image, sensitivity: float = 0.55) -> object:
    """Return a stable 0..1 response for plate symbols and their contours."""
    if np is None or cv2 is None:
        return None
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_f = gray.astype("float32")
        gray_norm = np.clip(gray_f / 255.0, 0.0, 1.0)
        contour_sensitivity = max(0.0, min(1.0, float(sensitivity if sensitivity is not None else 0.55)))
        image_h, image_w = gray_norm.shape[:2]
        polarity_roi = np.ones_like(gray_norm, dtype=bool)
        try:
            pad_x = max(2, int(round(float(image_w) * 0.055)))
            pad_y = max(1, int(round(float(image_h) * 0.075)))
            if image_w - 2 * pad_x >= max(12, int(round(float(image_w) * 0.56))) and image_h - 2 * pad_y >= max(6, int(round(float(image_h) * 0.56))):
                polarity_roi[:, :] = False
                polarity_roi[pad_y:image_h - pad_y, pad_x:image_w - pad_x] = True
        except Exception:
            polarity_roi[:, :] = True

        local_bg = cv2.GaussianBlur(gray_norm, (0, 0), sigmaX=3.0, sigmaY=3.0)
        local_bg_wide = cv2.GaussianBlur(gray_norm, (0, 0), sigmaX=6.0, sigmaY=5.0)
        global_dark = np.clip((0.72 - gray_norm) / 0.48, 0.0, 1.0)
        global_light = np.clip((gray_norm - 0.28) / 0.48, 0.0, 1.0)
        local_dark = np.clip((local_bg - gray_norm - 0.012) / 0.18, 0.0, 1.0)
        local_light = np.clip((gray_norm - local_bg - 0.012) / 0.18, 0.0, 1.0)
        faint_dark = np.clip((local_bg_wide - gray_norm - 0.004) / 0.075, 0.0, 1.0)
        faint_light = np.clip((gray_norm - local_bg_wide - 0.004) / 0.075, 0.0, 1.0)
        try:
            gray_u8 = np.clip(gray_f, 0, 255).astype("uint8")
            clahe = cv2.createCLAHE(clipLimit=1.65, tileGridSize=(4, 4))
            local_equalized = clahe.apply(gray_u8).astype("float32") / 255.0
            eq_bg = cv2.GaussianBlur(local_equalized, (0, 0), sigmaX=4.2, sigmaY=4.2)
            equalized_dark = np.clip((eq_bg - local_equalized - 0.006) / 0.105, 0.0, 1.0)
            equalized_light = np.clip((local_equalized - eq_bg - 0.006) / 0.105, 0.0, 1.0)
            blackhat_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (max(3, int(round(min(gray.shape[:2]) / 14.0)) | 1), max(3, int(round(min(gray.shape[:2]) / 22.0)) | 1)),
            )
            blackhat = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, blackhat_kernel).astype("float32") / 255.0
            blackhat_scale = float(np.percentile(blackhat, 99.0) or 0.0)
            if blackhat_scale > 0.0001:
                blackhat = np.clip(blackhat / blackhat_scale, 0.0, 1.0)
            else:
                blackhat = np.clip(blackhat, 0.0, 1.0)
            tophat = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, blackhat_kernel).astype("float32") / 255.0
            tophat_scale = float(np.percentile(tophat, 99.0) or 0.0)
            if tophat_scale > 0.0001:
                tophat = np.clip(tophat / tophat_scale, 0.0, 1.0)
            else:
                tophat = np.clip(tophat, 0.0, 1.0)
        except Exception:
            equalized_dark = np.zeros_like(gray_norm, dtype="float32")
            equalized_light = np.zeros_like(gray_norm, dtype="float32")
            blackhat = np.zeros_like(gray_norm, dtype="float32")
            tophat = np.zeros_like(gray_norm, dtype="float32")

        grad_x = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)
        grad_abs = np.abs(grad_x) + np.abs(grad_y)
        grad_scale = float(np.percentile(grad_abs, 98.5) or 0.0)
        if grad_scale > 0.0001:
            edge_response = np.clip(grad_abs / grad_scale, 0.0, 1.0)
        else:
            edge_response = np.clip(grad_abs, 0.0, 1.0)

        def _normalize_response(response) -> object:
            response = np.clip(np.asarray(response, dtype="float32"), 0.0, 1.0)
            baseline = float(np.percentile(response, 42.0) or 0.0)
            if baseline > 0.018:
                response = np.clip((response - baseline * 0.82) / max(0.08, 1.0 - baseline * 0.82), 0.0, 1.0)
            peak = float(np.percentile(response, 99.0) or 0.0)
            if 0.018 < peak < 0.38:
                response = np.clip(response / peak * 0.70, 0.0, 1.0)
            return response.astype("float32", copy=False)

        dark_evidence = np.maximum(local_dark, faint_dark * 0.92)
        dark_evidence = np.maximum(dark_evidence, equalized_dark * 0.88)
        dark_evidence = np.maximum(dark_evidence, blackhat * 0.70)
        dark_core = np.maximum(global_dark * (0.12 + 0.88 * dark_evidence), local_dark)
        dark_core = np.maximum(dark_core, faint_dark * 0.92)
        dark_core = np.maximum(dark_core, equalized_dark * 0.88)
        dark_core = np.maximum(dark_core, blackhat * 0.62)
        dark_response = _normalize_response(
            (dark_core * 0.78)
            + (edge_response * np.clip(dark_core + 0.20, 0.20, 1.0) * 0.30)
            + (blackhat * np.clip(edge_response + 0.20, 0.20, 1.0) * 0.18)
        )

        light_evidence = np.maximum(local_light, faint_light * 0.92)
        light_evidence = np.maximum(light_evidence, equalized_light * 0.88)
        light_evidence = np.maximum(light_evidence, tophat * 0.70)
        light_core = np.maximum(global_light * (0.12 + 0.88 * light_evidence), local_light)
        light_core = np.maximum(light_core, faint_light * 0.92)
        light_core = np.maximum(light_core, equalized_light * 0.88)
        light_core = np.maximum(light_core, tophat * 0.62)
        light_response = _normalize_response(
            (light_core * 0.78)
            + (edge_response * np.clip(light_core + 0.20, 0.20, 1.0) * 0.30)
            + (tophat * np.clip(edge_response + 0.20, 0.20, 1.0) * 0.18)
        )
        absolute_evidence = np.maximum(local_dark, local_light)
        absolute_evidence = np.maximum(absolute_evidence, faint_dark * 0.92)
        absolute_evidence = np.maximum(absolute_evidence, faint_light * 0.92)
        absolute_evidence = np.maximum(absolute_evidence, equalized_dark * 0.88)
        absolute_evidence = np.maximum(absolute_evidence, equalized_light * 0.88)
        absolute_evidence = np.maximum(absolute_evidence, blackhat * 0.58)
        absolute_evidence = np.maximum(absolute_evidence, tophat * 0.58)
        absolute_response = _normalize_response(
            (absolute_evidence * 0.80)
            + (edge_response * np.clip(absolute_evidence + 0.18, 0.18, 1.0) * 0.30)
        )

        def _response_score(response) -> float:
            response = np.clip(np.asarray(response, dtype="float32"), 0.0, 1.0)
            roi_values = response[polarity_roi] if response.shape[:2] == polarity_roi.shape[:2] and np.any(polarity_roi) else response
            peak = float(np.percentile(roi_values, 99.0) or 0.0)
            if peak <= 0.025:
                return 0.0
            threshold = max(0.14, min(0.62, float(np.percentile(roi_values, 72.0) or 0.0)))
            foreground = np.logical_and(response > threshold, polarity_roi)
            ratio = float(np.count_nonzero(foreground)) / max(1.0, float(np.count_nonzero(polarity_roi)))
            if ratio <= 0.001:
                return 0.0
            if ratio > 0.62:
                ratio_score = 0.06
            else:
                ratio_score = max(0.12, 1.0 - abs(ratio - 0.20) / 0.34)
            try:
                labels, _components = cv2.connectedComponents(foreground.astype("uint8"))
                component_score = min(1.0, max(0.35, float(labels - 1) / 6.0))
            except Exception:
                component_score = 0.65
            strong_mean = float(np.mean(response[foreground])) if np.any(foreground) else 0.0
            return (peak * 0.62 + strong_mean * 0.38) * ratio_score * component_score

        def _candidate_contour_mask(response) -> object:
            response = np.clip(np.asarray(response, dtype="float32"), 0.0, 1.0)
            roi_values = response[polarity_roi] if response.shape[:2] == polarity_roi.shape[:2] and np.any(polarity_roi) else response
            response_threshold = max(0.15, min(0.58, float(np.percentile(roi_values, 74.0) or 0.0)))
            response_mask = np.logical_and(response > response_threshold, polarity_roi)
            edge_values = edge_response[polarity_roi] if edge_response.shape[:2] == polarity_roi.shape[:2] and np.any(polarity_roi) else edge_response
            edge_threshold = max(0.055, min(0.42, float(np.percentile(edge_values, 68.0) or 0.0)))
            contour_mask = np.logical_and(response_mask, edge_response >= edge_threshold)
            roi_area = max(1, int(np.count_nonzero(polarity_roi)))
            if int(np.count_nonzero(contour_mask)) < max(12, int(roi_area * 0.0015)):
                contour_mask = response_mask
            if int(np.count_nonzero(contour_mask)) <= 0:
                return contour_mask
            try:
                count, labels, stats, _centroids = cv2.connectedComponentsWithStats(contour_mask.astype("uint8"), 8)
                filtered = np.zeros_like(contour_mask, dtype=bool)
                area_total = max(1.0, float(np.count_nonzero(polarity_roi)))
                min_area = max(3, int(round(area_total * 0.00008)))
                max_area = max(min_area + 1, int(round(area_total * 0.36)))
                for idx in range(1, count):
                    area = int(stats[idx, cv2.CC_STAT_AREA])
                    if min_area <= area <= max_area:
                        filtered[labels == idx] = True
                if int(np.count_nonzero(filtered)) > 0:
                    return filtered
            except Exception:
                pass
            return contour_mask

        def _polarity_validation_score(response, *, expected_dark: bool) -> float:
            # Validate polarity on the detected contour candidate, not on the
            # whole plate. This avoids treating a dark plate background as ink.
            response = np.clip(np.asarray(response, dtype="float32"), 0.0, 1.0)
            candidate = _candidate_contour_mask(response)
            candidate_count = int(np.count_nonzero(candidate))
            if candidate_count <= 0:
                return 0.0
            area_ratio = float(candidate_count) / max(1.0, float(np.count_nonzero(polarity_roi)))
            if area_ratio > 0.58:
                return _response_score(response) * 0.05
            local_delta = local_bg_wide - gray_norm if expected_dark else gray_norm - local_bg_wide
            delta_values = np.asarray(local_delta[candidate], dtype="float32")
            weight_values = np.asarray(response[candidate], dtype="float32")
            if delta_values.size <= 0:
                return 0.0
            weights = np.clip(weight_values, 0.05, 1.0)
            try:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                candidate_u8 = candidate.astype("uint8")
                ring = cv2.dilate(candidate_u8, kernel, iterations=2).astype(bool)
                ring = np.logical_and(ring, np.logical_not(candidate))
                ring = np.logical_and(ring, polarity_roi)
                if int(np.count_nonzero(ring)) > 0:
                    inside_mean = float(np.average(gray_norm[candidate], weights=weights))
                    outside_mean = float(np.mean(gray_norm[ring]))
                    ring_delta = outside_mean - inside_mean if expected_dark else inside_mean - outside_mean
                else:
                    ring_delta = 0.0
            except Exception:
                ring_delta = 0.0
            positive = np.clip(delta_values, 0.0, 0.28) / 0.28
            negative = np.clip(-delta_values, 0.0, 0.22) / 0.22
            ring_positive = max(0.0, min(1.0, ring_delta / 0.22))
            ring_negative = max(0.0, min(1.0, -ring_delta / 0.18))
            contrast_score = float(np.average(positive, weights=weights)) * 0.70 + ring_positive * 0.30
            wrong_score = float(np.average(negative, weights=weights)) * 0.70 + ring_negative * 0.30
            support_ratio = float(np.count_nonzero(delta_values > 0.010)) / max(1.0, float(delta_values.size))
            wrong_ratio = float(np.count_nonzero(delta_values < -0.006)) / max(1.0, float(delta_values.size))
            shape_score = _response_score(response)
            coverage_score = max(0.15, min(1.0, 1.0 - abs(area_ratio - 0.055) / 0.34))
            return (
                shape_score * 0.30
                + contrast_score * 0.42
                + support_ratio * 0.28
                - wrong_score * 0.34
                - wrong_ratio * 0.20
            ) * coverage_score

        shared_candidate = _candidate_contour_mask(absolute_response)

        def _shared_candidate_polarity_score(*, expected_dark: bool) -> float:
            candidate = shared_candidate
            candidate_count = int(np.count_nonzero(candidate))
            if candidate_count <= 0:
                return 0.0
            area_ratio = float(candidate_count) / max(1.0, float(np.count_nonzero(polarity_roi)))
            if area_ratio > 0.58:
                return 0.0
            signed_delta = local_bg_wide - gray_norm if expected_dark else gray_norm - local_bg_wide
            delta_values = np.asarray(signed_delta[candidate], dtype="float32")
            weights = np.clip(np.asarray(absolute_response[candidate], dtype="float32"), 0.05, 1.0)
            if delta_values.size <= 0:
                return 0.0
            support = float(np.average(np.clip(delta_values, 0.0, 0.28) / 0.28, weights=weights))
            contradiction = float(np.average(np.clip(-delta_values, 0.0, 0.22) / 0.22, weights=weights))
            support_ratio = float(np.count_nonzero(delta_values > 0.010)) / max(1.0, float(delta_values.size))
            contradiction_ratio = float(np.count_nonzero(delta_values < -0.006)) / max(1.0, float(delta_values.size))
            coverage_score = max(0.15, min(1.0, 1.0 - abs(area_ratio - 0.055) / 0.34))
            return (
                support * 0.52
                + support_ratio * 0.34
                - contradiction * 0.44
                - contradiction_ratio * 0.30
            ) * coverage_score

        def _raw_glyph_component_score(*, expected_dark: bool) -> float:
            # Independent sanity check on the source pixels: the chosen
            # polarity should form compact glyph-like components, not a large
            # plate-background blob or a halo around the opposite polarity.
            signed_delta = local_bg_wide - gray_norm if expected_dark else gray_norm - local_bg_wide
            roi_delta = signed_delta[polarity_roi] if signed_delta.shape[:2] == polarity_roi.shape[:2] and np.any(polarity_roi) else signed_delta
            positive_values = roi_delta[roi_delta > 0.0]
            if positive_values.size <= 0:
                return 0.0
            threshold = max(0.010, min(0.075, float(np.percentile(positive_values, 54.0) or 0.0)))
            mask = np.logical_and(signed_delta > threshold, polarity_roi)
            area_total = max(1.0, float(np.count_nonzero(polarity_roi)))
            area_ratio = float(np.count_nonzero(mask)) / area_total
            if area_ratio <= 0.001:
                return 0.0
            if area_ratio > 0.58:
                return 0.02
            try:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_u8 = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_OPEN, kernel, iterations=1)
                mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
                count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
                good_area = 0.0
                good_components = 0
                good_component_area_ratios: list[float] = []
                height, width = mask.shape[:2]
                for idx in range(1, count):
                    area = float(stats[idx, cv2.CC_STAT_AREA])
                    if area <= 0.0:
                        continue
                    comp_w = float(stats[idx, cv2.CC_STAT_WIDTH])
                    comp_h = float(stats[idx, cv2.CC_STAT_HEIGHT])
                    comp_area_ratio = area / area_total
                    if comp_area_ratio > 0.34:
                        continue
                    if comp_h < max(3.0, height * 0.10) or comp_w < max(2.0, width * 0.006):
                        continue
                    if comp_h > height * 0.98 or comp_w > width * 0.92:
                        continue
                    good_area += area
                    good_components += 1
                    good_component_area_ratios.append(comp_area_ratio)
                if good_area <= 0.0:
                    return 0.0
                good_area_ratios = sorted(good_component_area_ratios, reverse=True)
                top_area = good_area_ratios[0] if good_area_ratios else 0.0
                top_six_area = sum(good_area_ratios[:6])
                component_score = min(1.0, max(0.22, top_six_area / 0.14))
                cohesion_score = min(1.0, max(0.10, top_area / 0.035))
                fragmentation_score = max(0.30, min(1.0, 1.0 - max(0, good_components - 10) * 0.045))
                fill_score = max(0.12, min(1.0, 1.0 - abs((good_area / area_total) - 0.060) / 0.22))
                contrast_score = float(np.mean(np.clip(positive_values, 0.0, 0.22) / 0.22))
                return (
                    contrast_score * 0.30
                    + component_score * 0.30
                    + cohesion_score * 0.26
                    + fill_score * 0.14
                ) * fragmentation_score
            except Exception:
                contrast_score = float(np.mean(np.clip(positive_values, 0.0, 0.22) / 0.22))
                return contrast_score * max(0.10, min(1.0, 1.0 - abs(area_ratio - 0.060) / 0.28))

        # Pick the glyph polarity once. The decision is validated on pixels
        # belonging to contour candidates; the sensitivity slider below only
        # exposes more/less of the chosen response, otherwise it feels random.
        dark_support = _polarity_validation_score(dark_response, expected_dark=True)
        dark_inverse = _polarity_validation_score(dark_response, expected_dark=False)
        light_support = _polarity_validation_score(light_response, expected_dark=False)
        light_inverse = _polarity_validation_score(light_response, expected_dark=True)
        shared_dark = _shared_candidate_polarity_score(expected_dark=True)
        shared_light = _shared_candidate_polarity_score(expected_dark=False)
        dark_score = (dark_support - dark_inverse * 0.78) * 0.42 + (shared_dark - shared_light * 0.86) * 0.58
        light_score = (light_support - light_inverse * 0.78) * 0.42 + (shared_light - shared_dark * 0.86) * 0.58
        raw_dark = _raw_glyph_component_score(expected_dark=True)
        raw_light = _raw_glyph_component_score(expected_dark=False)
        dark_score += (raw_dark - raw_light * 0.72) * 0.95
        light_score += (raw_light - raw_dark * 0.72) * 0.95
        try:
            dark_mask = _candidate_contour_mask(dark_response)
            light_mask = _candidate_contour_mask(light_response)
            dark_area = float(np.count_nonzero(dark_mask)) / max(1.0, float(dark_mask.size))
            light_area = float(np.count_nonzero(light_mask)) / max(1.0, float(light_mask.size))
            if dark_area > 0.0 and light_area > 0.0:
                # The wrong polarity often describes a bright/dark halo around
                # the glyph. The ink itself is usually the more compact side
                # of the same high-contrast edge.
                dark_score += max(0.0, min(0.20, light_area - dark_area)) * 0.95
                light_score += max(0.0, min(0.20, dark_area - light_area)) * 0.95
        except Exception:
            pass
        score_gap = abs(dark_score - light_score)
        if score_gap <= 0.035:
            dark_score += _response_score(dark_response) * 0.16
            light_score += _response_score(light_response) * 0.16
            score_gap = abs(dark_score - light_score)
        confidence = max(0.0, min(1.0, score_gap / 0.18))
        if confidence < 0.28:
            polarity_gate = np.clip(np.abs(gray_norm - local_bg_wide) / 0.12, 0.0, 1.0)
            conservative_response = np.minimum(dark_response, light_response)
            balanced_response = np.clip(
                absolute_response * (0.34 + 0.66 * polarity_gate)
                + conservative_response * 0.18,
                0.0,
                1.0,
            )
            symbol_response = np.clip(
                balanced_response * (0.58 + 0.42 * confidence),
                0.0,
                1.0,
            )
        elif dark_score >= light_score:
            polarity_gate = np.clip((local_bg_wide - gray_norm) / 0.12, 0.0, 1.0)
            symbol_response = np.clip(
                dark_response * (0.30 + 0.70 * polarity_gate)
                + absolute_response * polarity_gate * 0.22,
                0.0,
                1.0,
            )
        else:
            polarity_gate = np.clip((gray_norm - local_bg_wide) / 0.12, 0.0, 1.0)
            symbol_response = np.clip(
                light_response * (0.30 + 0.70 * polarity_gate)
                + absolute_response * polarity_gate * 0.22,
                0.0,
                1.0,
            )

        threshold = 0.24 - 0.18 * contour_sensitivity
        symbol_response = np.clip((symbol_response - threshold) / max(0.08, 1.0 - threshold), 0.0, 1.0)
        gamma = max(0.55, 1.18 - 0.55 * contour_sensitivity)
        symbol_response = np.power(symbol_response, gamma).astype("float32", copy=False)
        baseline = float(np.percentile(symbol_response, 18.0) or 0.0)
        if baseline > 0.018:
            symbol_response = np.clip((symbol_response - baseline * 0.35) / max(0.08, 1.0 - baseline * 0.35), 0.0, 1.0)
        symbol_response = cv2.GaussianBlur(symbol_response.astype("float32"), (0, 0), sigmaX=0.18, sigmaY=0.18)
        if float(symbol_response.max() or 0.0) < 0.035 and float(edge_response.max() or 0.0) > 0.08:
            symbol_response = np.clip(edge_response * (0.22 + 0.28 * contour_sensitivity), 0.0, 1.0)
        return symbol_response.astype("float32", copy=False)
    except Exception:
        return None


def _build_smooth_symbol_contour_edge_mask(height_map) -> object:
    """Return a curve-smoothed contour mask for raised symbol shadows."""
    if np is None or cv2 is None or height_map is None:
        return None
    try:
        local = np.asarray(height_map, dtype="float32")
        if local.ndim != 2:
            return None
        height, width = local.shape[:2]
        if height < 4 or width < 4:
            return None
        peak = float(local.max() or 0.0)
        if peak <= 0.025:
            return None
        local = np.clip(local / max(0.001, peak), 0.0, 1.0)
        u8 = np.clip(local * 255.0, 0, 255).astype("uint8")
        u8 = cv2.GaussianBlur(u8, (0, 0), sigmaX=0.55, sigmaY=0.55)
        binary_candidates = []
        try:
            _threshold, otsu_binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_candidates.append(otsu_binary)
        except Exception:
            threshold = max(18, min(128, int(round(float(np.percentile(u8, 70.0) or 54.0)))))
            _threshold, threshold_binary = cv2.threshold(u8, threshold, 255, cv2.THRESH_BINARY)
            binary_candidates.append(threshold_binary)
        try:
            adaptive_block = max(9, int(round(min(height, width) / 5.5)))
            if adaptive_block % 2 == 0:
                adaptive_block += 1
            adaptive_block = min(45, adaptive_block)
            adaptive_binary = cv2.adaptiveThreshold(
                u8,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                max(3, adaptive_block),
                -2,
            )
            binary_candidates.append(adaptive_binary)
        except Exception:
            pass
        try:
            soft_threshold = max(10, min(96, int(round(float(np.percentile(u8, 62.0) or 36.0)))))
            _threshold, soft_binary = cv2.threshold(u8, soft_threshold, 255, cv2.THRESH_BINARY)
            binary_candidates.append(soft_binary)
        except Exception:
            pass
        if binary_candidates:
            binary = np.zeros_like(u8, dtype="uint8")
            for candidate in binary_candidates:
                binary = np.maximum(binary, candidate.astype("uint8", copy=False))
        else:
            binary = np.zeros_like(u8, dtype="uint8")

        foreground_ratio = float(np.count_nonzero(binary)) / max(1.0, float(height * width))
        if foreground_ratio < 0.002 or foreground_ratio > 0.62:
            threshold = max(16, min(138, int(round(float(np.percentile(u8, 72.0) or 58.0)))))
            _threshold, binary = cv2.threshold(u8, threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        min_dim = max(1, min(height, width))
        min_area = max(2.0, float(height * width) * 0.000012)
        max_area = float(height * width) * 0.42
        thickness = max(1, int(round(min_dim / 118.0)))
        canvas = np.zeros((height, width), dtype="uint8")

        for contour in contours:
            if contour is None or len(contour) < 4:
                continue
            area = abs(float(cv2.contourArea(contour)))
            perimeter = float(cv2.arcLength(contour, True) or 0.0)
            if area < min_area or area > max_area or perimeter < 4.0:
                continue
            points = contour.reshape(-1, 2).astype("float32")
            if len(points) >= 7:
                # A few closed-curve averaging passes remove pixel stairs but
                # preserve the glyph contour better than a heavy polygon fit.
                passes = 2 if len(points) < 64 else 3
                for _pass in range(passes):
                    points = (
                        points * 0.46
                        + np.roll(points, 1, axis=0) * 0.24
                        + np.roll(points, -1, axis=0) * 0.24
                        + np.roll(points, 2, axis=0) * 0.03
                        + np.roll(points, -2, axis=0) * 0.03
                    )
            points = np.round(points).astype("int32").reshape(-1, 1, 2)
            cv2.polylines(canvas, [points], True, 255, thickness=thickness, lineType=cv2.LINE_AA)

        if not np.any(canvas):
            return None
        edge = canvas.astype("float32") / 255.0
        edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=0.28, sigmaY=0.28)
        return np.clip(edge * 1.18, 0.0, 1.0).astype("float32", copy=False)
    except Exception:
        return None


def _build_solid_symbol_surface_mask(height_map) -> object:
    """Return a filled glyph mask so parallel stroke edges form one surface."""
    if np is None or cv2 is None or height_map is None:
        return None
    try:
        local = np.asarray(height_map, dtype="float32")
        if local.ndim != 2:
            return None
        height, width = local.shape[:2]
        if height < 4 or width < 4:
            return None
        peak = float(local.max() or 0.0)
        if peak <= 0.025:
            return None
        local = np.clip(local / max(0.001, peak), 0.0, 1.0)
        u8 = np.clip(local * 255.0, 0, 255).astype("uint8")
        u8 = cv2.GaussianBlur(u8, (0, 0), sigmaX=0.48, sigmaY=0.48)

        binary = None
        try:
            _threshold, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception:
            binary = None
        if binary is None:
            threshold = max(18, min(132, int(round(float(np.percentile(u8, 72.0) or 58.0)))))
            _threshold, binary = cv2.threshold(u8, threshold, 255, cv2.THRESH_BINARY)

        foreground_ratio = float(np.count_nonzero(binary)) / max(1.0, float(height * width))
        if foreground_ratio < 0.002 or foreground_ratio > 0.58:
            threshold = max(14, min(126, int(round(float(np.percentile(u8, 76.0) or 64.0)))))
            _threshold, binary = cv2.threshold(u8, threshold, 255, cv2.THRESH_BINARY)
        raw_binary = binary.copy()

        min_dim = max(1, min(height, width))
        bridge = max(2, min(8, int(round(min_dim * 0.045))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge * 2 + 1, bridge * 2 + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

        try:
            components, labels, stats, _centroids = cv2.connectedComponentsWithStats((binary > 0).astype("uint8"), 8)
            filtered = np.zeros_like(binary, dtype="uint8")
            min_area = max(2, int(round(float(height * width) * 0.000014)))
            aspect = float(width) / max(1.0, float(height))
            max_area_ratio = 0.62 if aspect >= 2.2 else 0.48
            max_area = max(min_area + 1, int(round(float(height * width) * max_area_ratio)))
            for label in range(1, int(components)):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if min_area <= area <= max_area:
                    filtered[labels == label] = 255
            if np.any(filtered):
                binary = filtered
        except Exception:
            pass

        # Closing is useful for broken ink, but it must not fill real hollow
        # glyph interiors such as 0, 4, 8 or 9.
        hole_mask = _build_enclosed_symbol_hole_mask(raw_binary)
        if hole_mask is not None:
            binary = np.where(hole_mask > 0, 0, binary).astype("uint8", copy=False)

        if int(np.count_nonzero(binary)) <= 0:
            return None
        soft = binary.astype("float32") / 255.0
        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=0.34, sigmaY=0.34)
        return np.clip(soft, 0.0, 1.0).astype("float32", copy=False)
    except Exception:
        return None


def _build_enclosed_symbol_hole_mask(binary_mask) -> object:
    """Return enclosed background islands inside glyph strokes."""
    if np is None or cv2 is None or binary_mask is None:
        return None
    try:
        foreground = (np.asarray(binary_mask) > 0).astype("uint8")
        height, width = foreground.shape[:2]
        if height < 4 or width < 4:
            return None
        background = (1 - foreground).astype("uint8")
        flood = background.copy()
        mask = np.zeros((height + 2, width + 2), dtype="uint8")
        border_points = []
        for x in range(width):
            border_points.append((x, 0))
            border_points.append((x, height - 1))
        for y in range(height):
            border_points.append((0, y))
            border_points.append((width - 1, y))
        for x, y in border_points:
            if flood[y, x] == 1:
                cv2.floodFill(flood, mask, (int(x), int(y)), 2)
        holes = ((background == 1) & (flood != 2)).astype("uint8")
        if int(np.count_nonzero(holes)) <= 0:
            return None
        components, labels, stats, _centroids = cv2.connectedComponentsWithStats(holes, 8)
        filtered = np.zeros_like(holes, dtype="uint8")
        min_area = max(3, int(round(float(height * width) * 0.000045)))
        max_area = max(min_area + 1, int(round(float(height * width) * 0.16)))
        for label in range(1, int(components)):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if min_area <= area <= max_area:
                filtered[labels == label] = 1
        return filtered if int(np.count_nonzero(filtered)) > 0 else None
    except Exception:
        return None


def _build_contour_profile_surface(height_map, profile: AugmentationProfile | None = None) -> dict:
    """Build a rounded relief profile orthogonal to the smoothed glyph contour."""
    if np is None or cv2 is None or height_map is None:
        return {}
    try:
        local = np.asarray(height_map, dtype="float32")
        if local.ndim != 2:
            return {}
        height, width = local.shape[:2]
        if height < 4 or width < 4:
            return {}

        solid_symbol = _build_solid_symbol_surface_mask(local)
        contour_curve = _build_smooth_symbol_contour_edge_mask(local)
        if contour_curve is None:
            return {}
        if solid_symbol is None:
            solid_symbol = cv2.GaussianBlur(np.clip(contour_curve, 0.0, 1.0), (0, 0), sigmaX=0.55, sigmaY=0.55)
        solid_u8 = (np.clip(solid_symbol, 0.0, 1.0) > 0.16).astype("uint8")
        if int(np.count_nonzero(solid_u8)) <= 0:
            return {}

        min_dim = max(1.0, float(min(height, width)))
        relief_strength = max(
            0.0,
            min(
                3.0,
                float(getattr(profile, "dark_relief_strength", 0.0) or 0.0) if profile is not None else 0.0,
            ),
        )
        base_profile_radius = max(1.45, min(8.0, min_dim * (0.016 + 0.014 * min(1.0, relief_strength / 3.0))))
        inside_distance = cv2.distanceTransform(solid_u8.astype("uint8"), cv2.DIST_L2, 5).astype("float32")
        outside_distance = cv2.distanceTransform((1 - solid_u8).astype("uint8"), cv2.DIST_L2, 5).astype("float32")
        stroke_samples = inside_distance[inside_distance > 0.01]
        if getattr(stroke_samples, "size", 0):
            stroke_half_width = float(np.percentile(stroke_samples, 72.0) or base_profile_radius)
        else:
            stroke_half_width = base_profile_radius
        mesh_square_cap = max(1.35, min(5.0, min_dim * 0.030))
        profile_radius = max(
            1.20,
            min(
                base_profile_radius,
                max(1.20, stroke_half_width * 0.92),
                mesh_square_cap,
            ),
        )
        inside_ratio = np.clip(inside_distance / max(0.001, profile_radius), 0.0, 1.0)
        outside_ratio = np.clip(outside_distance / max(0.001, profile_radius * 0.72), 0.0, 1.0)
        relief_curve = relief_profile_curve_for_preset(
            getattr(profile, "relief_profile_preset", DEFAULT_RELIEF_PROFILE_PRESET) if profile is not None else DEFAULT_RELIEF_PROFILE_PRESET,
            getattr(profile, "relief_profile_curve", DEFAULT_RELIEF_PROFILE_CURVE) if profile is not None else DEFAULT_RELIEF_PROFILE_CURVE,
        )

        # Build one solid raised glyph, not two ridges following both stroke
        # edges. This removes the "double tunnel" cross-section that made
        # letters look concave under directional light.
        inside_cap = _relief_profile_height_from_center_ratio(inside_ratio, relief_curve)
        inside_cap = np.where(solid_u8 > 0, inside_cap, 0.0).astype("float32")
        outside_bevel = (0.5 + 0.5 * np.cos(outside_ratio * math.pi)).astype("float32")
        outside_bevel = np.where((solid_u8 <= 0) & (outside_distance <= profile_radius * 0.72), outside_bevel * 0.12, 0.0)
        rounded_cap = np.clip(np.maximum(inside_cap, outside_bevel), 0.0, 1.0)
        rounded_cap = np.maximum(rounded_cap, np.clip(solid_symbol, 0.0, 1.0) * 0.06)
        rounded_cap = cv2.GaussianBlur(
            rounded_cap,
            (0, 0),
            sigmaX=0.24 + 0.08 * profile_radius,
            sigmaY=0.24 + 0.08 * profile_radius,
        )
        peak = float(rounded_cap.max() or 0.0)
        if peak <= 0.001:
            return {}
        rounded_cap = np.clip(rounded_cap / peak, 0.0, 1.0)

        grad_x = cv2.Sobel(rounded_cap, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(rounded_cap, cv2.CV_32F, 0, 1, ksize=3)
        normal_energy = np.abs(grad_x) + np.abs(grad_y)
        normal_scale = float(np.percentile(normal_energy, 99.0) or 0.0)
        if normal_scale > 0.0001:
            normal_x = np.clip(grad_x / normal_scale, -1.0, 1.0)
            normal_y = np.clip(grad_y / normal_scale, -1.0, 1.0)
            normal_energy = np.clip(normal_energy / normal_scale, 0.0, 1.0)
        else:
            normal_x = np.zeros_like(rounded_cap, dtype="float32")
            normal_y = np.zeros_like(rounded_cap, dtype="float32")
            normal_energy = np.zeros_like(rounded_cap, dtype="float32")

        return {
            "height": rounded_cap.astype("float32", copy=False),
            "contour_curve": np.clip(contour_curve, 0.0, 1.0).astype("float32", copy=False),
            "wire_curve": np.clip(solid_symbol, 0.0, 1.0).astype("float32", copy=False),
            "normal_x": normal_x.astype("float32", copy=False),
            "normal_y": normal_y.astype("float32", copy=False),
            "normal_energy": normal_energy.astype("float32", copy=False),
            "profile_radius": float(profile_radius),
            "stroke_half_width": float(stroke_half_width),
        }
    except Exception:
        return {}


def _build_directional_symbol_shadow_edge_mask(height_map, light_x: float, light_y: float, fallback_edge=None) -> object:
    """Return only the downstream glyph silhouette that can cast a shadow."""
    if np is None or cv2 is None or height_map is None:
        return fallback_edge
    try:
        local = np.asarray(height_map, dtype="float32")
        if local.ndim != 2:
            return fallback_edge
        height, width = local.shape[:2]
        if height < 4 or width < 4:
            return fallback_edge
        direction_len = math.hypot(float(light_x), float(light_y))
        if direction_len <= 0.0001:
            return fallback_edge
        lx = float(light_x) / direction_len
        ly = float(light_y) / direction_len

        peak = float(local.max() or 0.0)
        if peak <= 0.025:
            return fallback_edge
        local = np.clip(local / max(0.001, peak), 0.0, 1.0)
        u8 = np.clip(local * 255.0, 0, 255).astype("uint8")
        u8 = cv2.GaussianBlur(u8, (0, 0), sigmaX=0.60, sigmaY=0.60)
        try:
            _threshold, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception:
            threshold = max(24, min(132, int(round(float(np.percentile(u8, 74.0) or 68.0)))))
            _threshold, binary = cv2.threshold(u8, threshold, 255, cv2.THRESH_BINARY)

        foreground_ratio = float(np.count_nonzero(binary)) / max(1.0, float(height * width))
        if foreground_ratio < 0.002 or foreground_ratio > 0.62:
            threshold = max(22, min(142, int(round(float(np.percentile(u8, 78.0) or 76.0)))))
            _threshold, binary = cv2.threshold(u8, threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return fallback_edge

        min_area = max(2.0, float(height * width) * 0.000012)
        max_area = float(height * width) * 0.42
        filled = np.zeros((height, width), dtype="uint8")
        for contour in contours:
            if contour is None or len(contour) < 4:
                continue
            area = abs(float(cv2.contourArea(contour)))
            perimeter = float(cv2.arcLength(contour, True) or 0.0)
            if area < min_area or area > max_area or perimeter < 4.0:
                continue
            points = contour.reshape(-1, 2).astype("float32")
            if len(points) >= 7:
                passes = 2 if len(points) < 64 else 3
                for _pass in range(passes):
                    points = (
                        points * 0.46
                        + np.roll(points, 1, axis=0) * 0.24
                        + np.roll(points, -1, axis=0) * 0.24
                        + np.roll(points, 2, axis=0) * 0.03
                        + np.roll(points, -2, axis=0) * 0.03
                    )
            points = np.round(points).astype("int32").reshape(-1, 1, 2)
            cv2.fillPoly(filled, [points], 255, lineType=cv2.LINE_AA)

        if not np.any(filled):
            return fallback_edge
        smooth_fill = filled.astype("float32") / 255.0
        smooth_fill = cv2.GaussianBlur(smooth_fill, (0, 0), sigmaX=0.62, sigmaY=0.62)
        grad_x = cv2.Sobel(smooth_fill, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smooth_fill, cv2.CV_32F, 0, 1, ksize=3)
        downstream = np.clip(-(grad_x * lx + grad_y * ly), 0.0, None)
        scale = float(np.percentile(downstream, 99.0) or 0.0)
        if scale > 0.0001:
            downstream = np.clip(downstream / scale, 0.0, 1.0)
        downstream = cv2.GaussianBlur(downstream, (0, 0), sigmaX=0.34, sigmaY=0.34)
        if fallback_edge is not None:
            try:
                fallback = np.clip(np.asarray(fallback_edge, dtype="float32"), 0.0, 1.0)
                if fallback.shape == downstream.shape:
                    downstream = np.clip(np.maximum(downstream, fallback * 0.08), 0.0, 1.0)
            except Exception:
                pass
        if float(downstream.max() or 0.0) <= 0.01:
            return fallback_edge
        return np.clip(downstream * 1.20, 0.0, 1.0).astype("float32", copy=False)
    except Exception:
        return fallback_edge


def _resize_float_mask(mask, width: int, height: int):
    if mask is None or np is None:
        return None
    try:
        local = np.asarray(mask, dtype="float32")
        if local.shape[:2] != (height, width):
            local = cv2.resize(local, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.clip(local, 0.0, 1.0)
    except Exception:
        return None


def _surface_float_mask(surface: dict | None, key: str, width: int, height: int, *, signed: bool = False):
    """Read a cached geometry mask and resize it without rebuilding contours."""
    if not isinstance(surface, dict) or np is None or cv2 is None:
        return None
    try:
        value = surface.get(key)
        if value is None:
            return None
        local = np.asarray(value, dtype="float32")
        if local.shape[:2] != (height, width):
            local = cv2.resize(local, (width, height), interpolation=cv2.INTER_LINEAR)
        if signed:
            return np.clip(local, -1.0, 1.0).astype("float32", copy=False)
        return np.clip(local, 0.0, 1.0).astype("float32", copy=False)
    except Exception:
        return None


def _normalized_gradient_fields(height_map) -> tuple[object, object, object]:
    """Return normalized x/y gradients and their energy for a 2.5D surface."""
    if np is None or cv2 is None:
        return None, None, None
    try:
        local = np.asarray(height_map, dtype="float32")
        grad_x = cv2.Sobel(local, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(local, cv2.CV_32F, 0, 1, ksize=3)
        energy = np.abs(grad_x) + np.abs(grad_y)
        scale = float(np.percentile(energy, 99.0) or 0.0)
        if scale > 0.0001:
            grad_x = grad_x / scale
            grad_y = grad_y / scale
            energy = np.clip(energy / scale, 0.0, 1.0)
        else:
            energy = np.clip(energy, 0.0, 1.0)
        return grad_x.astype("float32", copy=False), grad_y.astype("float32", copy=False), energy.astype("float32", copy=False)
    except Exception:
        return None, None, None


def _estimate_plate_shadow_floor(image, symbol_mask=None):
    """Estimate the darkest plausible plate-material value under a cast shadow."""
    if np is None or cv2 is None:
        return None
    try:
        base = np.asarray(image, dtype="float32")
        if base.ndim != 3 or base.shape[2] < 3:
            return None
        height, width = base.shape[:2]
        if height < 2 or width < 2:
            return None
        if symbol_mask is not None:
            ink = np.clip(np.asarray(symbol_mask, dtype="float32"), 0.0, 1.0)
            if ink.shape[:2] != (height, width):
                ink = cv2.resize(ink, (width, height), interpolation=cv2.INTER_LINEAR)
            receiver = np.clip(1.0 - cv2.GaussianBlur(ink, (0, 0), sigmaX=0.85, sigmaY=0.85) * 0.92, 0.0, 1.0)
        else:
            receiver = np.ones((height, width), dtype="float32")

        smooth = cv2.GaussianBlur(base, (0, 0), sigmaX=max(1.2, width * 0.010), sigmaY=max(1.0, height * 0.016))
        plate_color = smooth * 0.64 + base * 0.36
        luma = (
            plate_color[:, :, 0] * 0.114
            + plate_color[:, :, 1] * 0.587
            + plate_color[:, :, 2] * 0.299
        )
        bright_reference = float(np.percentile(luma[receiver > 0.45] if np.any(receiver > 0.45) else luma, 58.0) or 0.0)
        ambient_ratio = max(0.48, min(0.78, 0.54 + 0.18 * (bright_reference / 255.0)))
        floor = np.clip(plate_color * ambient_ratio, 18.0, 255.0)
        return floor.astype("float32", copy=False), receiver.astype("float32", copy=False)
    except Exception:
        return None


def _build_contour_blanket_surface(
    image,
    profile: AugmentationProfile,
    rng=None,
    *,
    relief_mask=None,
    include_water: bool = True,
    base_surface: dict | None = None,
) -> dict:
    """Build the shared 2.5D "blanket" laid over symbols and plate edges.

    The image remains a 2D rectified crop, but material effects read one
    consistent height/thickness field: ink contours, optional mud/rain relief,
    water pooling under raised edges and a smooth film over everything.
    """
    empty: dict = {}
    if np is None or cv2 is None:
        return empty
    try:
        profile = (profile or AugmentationProfile()).normalized()
        height, width = image.shape[:2]
        if height < 2 or width < 2:
            return empty

        symbol_height = _surface_float_mask(base_surface, "symbol_height", width, height)
        if symbol_height is None:
            symbol_height = _build_symbol_contour_height_map(image, _contour_detection_sensitivity(profile))
        if symbol_height is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            symbol_height = np.clip((0.66 - gray.astype("float32") / 255.0) / 0.52, 0.0, 1.0)
        symbol_height = np.clip(symbol_height.astype("float32"), 0.0, 1.0)

        local_relief = _resize_float_mask(relief_mask, width, height)
        relief_base = np.maximum(symbol_height, local_relief) if local_relief is not None else symbol_height.copy()
        relief_base = np.clip(relief_base, 0.0, 1.0)

        grad_x = cv2.Sobel(relief_base, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(relief_base, cv2.CV_32F, 0, 1, ksize=3)
        lower_symbol_edge = np.clip(-grad_y, 0.0, None)
        lower_edge_scale = float(np.percentile(lower_symbol_edge, 98.5) or 0.0)
        if lower_edge_scale > 0.0001:
            lower_symbol_edge = np.clip(lower_symbol_edge / lower_edge_scale, 0.0, 1.0)
        else:
            lower_symbol_edge = np.clip(lower_symbol_edge, 0.0, 1.0)
        lower_symbol_edge = cv2.GaussianBlur(lower_symbol_edge.astype("float32"), (0, 0), sigmaX=0.56, sigmaY=0.94)

        contour_edge = np.clip(np.abs(grad_x) + np.abs(grad_y), 0.0, 1.0)
        edge_scale = float(np.percentile(contour_edge, 99.0) or 0.0)
        if edge_scale > 0.0001:
            contour_edge = np.clip(contour_edge / edge_scale, 0.0, 1.0)
        contour_edge = cv2.GaussianBlur(contour_edge.astype("float32"), (0, 0), sigmaX=0.55, sigmaY=0.55)
        contour_profile_height = _surface_float_mask(base_surface, "contour_profile", width, height)
        contour_profile_edge = _surface_float_mask(base_surface, "contour_curve", width, height)
        contour_profile_wire_curve = _surface_float_mask(base_surface, "contour_profile_wire_curve", width, height)
        contour_profile_normal_x = _surface_float_mask(base_surface, "contour_profile_normal_x", width, height, signed=True)
        contour_profile_normal_y = _surface_float_mask(base_surface, "contour_profile_normal_y", width, height, signed=True)
        contour_profile_normal_energy = _surface_float_mask(base_surface, "contour_profile_normal_energy", width, height)
        if (
            contour_profile_height is None
            or contour_profile_normal_x is None
            or contour_profile_normal_y is None
            or contour_profile_normal_energy is None
        ):
            contour_profile = _build_contour_profile_surface(symbol_height, profile)
            contour_profile_height = contour_profile.get("height") if contour_profile else None
            contour_profile_edge = contour_profile.get("contour_curve") if contour_profile else None
            contour_profile_wire_curve = contour_profile.get("wire_curve") if contour_profile else None
            contour_profile_normal_x = contour_profile.get("normal_x") if contour_profile else None
            contour_profile_normal_y = contour_profile.get("normal_y") if contour_profile else None
            contour_profile_normal_energy = contour_profile.get("normal_energy") if contour_profile else None
        if contour_profile_edge is not None:
            contour_edge = np.clip(np.maximum(contour_edge * 0.42, contour_profile_edge), 0.0, 1.0)

        water_strength = max(0.0, min(1.0, float(getattr(profile, "water_film_strength", 0.0) or 0.0))) if include_water else 0.0
        unevenness = max(0.0, min(1.0, float(getattr(profile, "water_film_unevenness", 0.35) if getattr(profile, "water_film_unevenness", None) is not None else 0.35)))
        contour_response = max(0.0, min(1.0, float(getattr(profile, "water_film_contour_response", 0.55) if getattr(profile, "water_film_contour_response", None) is not None else 0.55)))
        relief_strength = max(0.0, min(3.0, float(getattr(profile, "dark_relief_strength", 0.0) or 0.0)))

        contour_pool = cv2.GaussianBlur(
            np.clip(relief_base * 0.62 + contour_edge * 1.18, 0.0, 1.0),
            (0, 0),
            sigmaX=1.0 + 1.8 * contour_response,
            sigmaY=1.0 + 1.8 * contour_response,
        )

        yy, xx = np.mgrid[0:height, 0:width].astype("float32")
        nx = xx / max(1.0, float(width - 1))
        ny = yy / max(1.0, float(height - 1))
        norm_y_metric = max(0.001, float(height) / max(1.0, float(width)))
        bottom_band = np.clip((ny - 0.68) / 0.32, 0.0, 1.0)
        bottom_band = bottom_band * bottom_band * (3.0 - 2.0 * bottom_band)
        bottom_edge_pool = np.exp(-((1.0 - ny) ** 2) / (2.0 * (0.055 + 0.030 * unevenness) ** 2)).astype("float32")
        corner_sigma_x = 0.12 + 0.045 * unevenness
        corner_sigma_y = 0.16 + 0.055 * unevenness
        left_corner_pool = np.exp(
            -(
                ((nx - 0.055) ** 2) / (2.0 * corner_sigma_x * corner_sigma_x)
                + ((ny - 0.935) ** 2) / (2.0 * corner_sigma_y * corner_sigma_y)
            )
        ).astype("float32")
        right_corner_pool = np.exp(
            -(
                ((nx - 0.945) ** 2) / (2.0 * corner_sigma_x * corner_sigma_x)
                + ((ny - 0.935) ** 2) / (2.0 * corner_sigma_y * corner_sigma_y)
            )
        ).astype("float32")
        drip_shift = max(1, int(round(float(height) * (0.020 + 0.035 * max(water_strength, 0.12)))))
        lower_edge_pool = cv2.warpAffine(
            lower_symbol_edge,
            np.float32([[1, 0, 0], [0, 1, drip_shift]]),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        lower_edge_pool = cv2.GaussianBlur(
            np.maximum(lower_symbol_edge * 0.72, lower_edge_pool),
            (0, 0),
            sigmaX=0.65 + 1.35 * unevenness,
            sigmaY=1.20 + 2.40 * unevenness,
        )
        water_accumulation = np.clip(
            lower_edge_pool * bottom_band * contour_response * (0.58 + 0.78 * max(water_strength, 0.18))
            + bottom_edge_pool * (0.18 + 0.54 * max(water_strength, 0.18))
            + np.maximum(left_corner_pool, right_corner_pool) * (0.36 + 0.82 * max(water_strength, 0.18)),
            0.0,
            1.0,
        )
        water_accumulation = cv2.GaussianBlur(
            water_accumulation.astype("float32"),
            (0, 0),
            sigmaX=0.80 + 1.40 * unevenness,
            sigmaY=0.92 + 1.80 * unevenness,
        )

        if water_strength > 0.001:
            noise_h = max(4, min(42, height // 10 or 4))
            noise_w = max(4, min(84, width // 10 or 4))
            local_rng = rng or random.Random(int(getattr(profile, "seed", 42) or 42) + 557)
            noise = np.array(
                [[_rng_float(local_rng, 0.0, 1.0) for _x in range(noise_w)] for _y in range(noise_h)],
                dtype="float32",
            )
            noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
            noise = cv2.GaussianBlur(
                noise,
                (0, 0),
                sigmaX=max(1.0, min(width, height) * (0.012 + 0.030 * unevenness)),
                sigmaY=max(1.0, min(width, height) * (0.010 + 0.026 * unevenness)),
            )
            noise_range = float(noise.max() - noise.min())
            if noise_range > 0.0001:
                noise = (noise - float(noise.min())) / noise_range
        else:
            noise = np.zeros((height, width), dtype="float32")

        flow_bias = np.clip(0.18 + ny * 0.54 + np.sin((nx * 1.7 + ny * 0.35) * math.tau) * 0.035, 0.0, 1.0)
        film = np.clip(
            0.34
            + noise * (0.32 + 0.42 * unevenness)
            + flow_bias * (0.12 + 0.20 * max(water_strength, 0.12))
            + contour_pool * contour_response * (0.18 + 0.42 * max(water_strength, 0.12))
            + water_accumulation * (0.24 + 0.48 * max(water_strength, 0.12)),
            0.0,
            1.0,
        )
        film = cv2.GaussianBlur(film.astype("float32"), (0, 0), sigmaX=0.65 + 1.6 * unevenness, sigmaY=0.65 + 1.6 * unevenness)
        water_mask = np.clip((film + water_accumulation * (0.18 + 0.44 * water_strength)) * water_strength, 0.0, 1.0)

        if contour_profile_height is not None:
            dry_relief_height = np.clip(
                (relief_base ** 1.18) * (0.045 + 0.180 * relief_strength)
                + contour_profile_height * (0.105 + 0.520 * relief_strength)
                + contour_edge * (0.020 + 0.105 * relief_strength),
                0.0,
                1.0,
            )
        else:
            dry_relief_height = np.clip(
                (relief_base ** 1.12) * (0.10 + 0.42 * relief_strength)
                + contour_edge * (0.04 + 0.22 * relief_strength),
                0.0,
                1.0,
            )
        if water_strength > 0.001:
            blanket_height = np.clip(
                dry_relief_height
                + film * water_strength * (0.24 + 0.22 * unevenness)
                + contour_pool * contour_response * water_strength * (0.18 + 0.36 * water_strength)
                + water_accumulation * water_strength * (0.60 + 0.72 * water_strength),
                0.0,
                1.0,
            )
        else:
            blanket_height = dry_relief_height
        blanket_height = cv2.GaussianBlur(
            blanket_height.astype("float32"),
            (0, 0),
            sigmaX=0.42 + 1.05 * (unevenness if water_strength > 0.001 else 0.20),
            sigmaY=0.42 + 1.05 * (unevenness if water_strength > 0.001 else 0.20),
        )
        normal_x, normal_y, normal_energy = _normalized_gradient_fields(blanket_height)
        if normal_x is None or normal_y is None or normal_energy is None:
            normal_x = np.zeros_like(blanket_height, dtype="float32")
            normal_y = np.zeros_like(blanket_height, dtype="float32")
            normal_energy = np.zeros_like(blanket_height, dtype="float32")
        if contour_profile_normal_x is not None and contour_profile_normal_y is not None and contour_profile_normal_energy is not None:
            profile_weight = np.clip(contour_profile_height if contour_profile_height is not None else contour_profile_normal_energy, 0.0, 1.0)
            normal_x = np.clip(
                normal_x * (1.0 - profile_weight * 0.62) + contour_profile_normal_x * profile_weight * 0.62,
                -1.0,
                1.0,
            )
            normal_y = np.clip(
                normal_y * (1.0 - profile_weight * 0.62) + contour_profile_normal_y * profile_weight * 0.62,
                -1.0,
                1.0,
            )
            normal_energy = np.clip(np.maximum(normal_energy, contour_profile_normal_energy * profile_weight), 0.0, 1.0)
        plate_surface = _plate_curvature_surface(profile, height, width)
        plate_z = None
        plate_normal_z = None
        if plate_surface:
            try:
                plate_gradient_x = np.asarray(plate_surface.get("gradient_x"), dtype="float32")
                plate_gradient_y = np.asarray(plate_surface.get("gradient_y"), dtype="float32")
                plate_normal_z = np.asarray(plate_surface.get("normal_z"), dtype="float32")
                plate_energy = np.asarray(plate_surface.get("normal_energy"), dtype="float32")
                plate_z = np.asarray(plate_surface.get("z"), dtype="float32")
                if (
                    plate_gradient_x.shape == (height, width)
                    and plate_gradient_y.shape == (height, width)
                    and plate_normal_z.shape == (height, width)
                    and plate_energy.shape == (height, width)
                ):
                    curve_weight = 0.42 + 0.48 * float(plate_surface.get("gain", 0.0) or 0.0)
                    normal_x = np.clip(normal_x + plate_gradient_x * curve_weight, -1.0, 1.0)
                    normal_y = np.clip(normal_y + plate_gradient_y * curve_weight, -1.0, 1.0)
                    normal_energy = np.clip(np.maximum(normal_energy, plate_energy * (0.35 + 0.45 * curve_weight)), 0.0, 1.0)
            except Exception:
                plate_z = None
                plate_normal_z = None
        return {
            "symbol_height": symbol_height.astype("float32", copy=False),
            "relief_base": relief_base.astype("float32", copy=False),
            "contour_edge": contour_edge.astype("float32", copy=False),
            "contour_profile": contour_profile_height.astype("float32", copy=False) if contour_profile_height is not None else None,
            "contour_curve": contour_profile_edge.astype("float32", copy=False) if contour_profile_edge is not None else None,
            "contour_profile_wire_curve": contour_profile_wire_curve.astype("float32", copy=False) if contour_profile_wire_curve is not None else None,
            "contour_profile_normal_x": contour_profile_normal_x.astype("float32", copy=False) if contour_profile_normal_x is not None else None,
            "contour_profile_normal_y": contour_profile_normal_y.astype("float32", copy=False) if contour_profile_normal_y is not None else None,
            "contour_profile_normal_energy": contour_profile_normal_energy.astype("float32", copy=False) if contour_profile_normal_energy is not None else None,
            "lower_symbol_edge": lower_symbol_edge.astype("float32", copy=False),
            "lower_edge_pool": lower_edge_pool.astype("float32", copy=False),
            "bottom_edge_pool": bottom_edge_pool.astype("float32", copy=False),
            "left_corner_pool": left_corner_pool.astype("float32", copy=False),
            "right_corner_pool": right_corner_pool.astype("float32", copy=False),
            "contour_pool": contour_pool.astype("float32", copy=False),
            "water_accumulation": water_accumulation.astype("float32", copy=False),
            "water_mask": water_mask.astype("float32", copy=False),
            "film": film.astype("float32", copy=False),
            "height": blanket_height.astype("float32", copy=False),
            "normal_x": normal_x,
            "normal_y": normal_y,
            "normal_energy": normal_energy,
            "plate_curve_z": plate_z.astype("float32", copy=False) if plate_z is not None and getattr(plate_z, "shape", None) == (height, width) else None,
            "plate_curve_normal_z": plate_normal_z.astype("float32", copy=False) if plate_normal_z is not None and getattr(plate_normal_z, "shape", None) == (height, width) else None,
            "grid_x": xx,
            "grid_y": yy,
        }
    except Exception:
        return empty


def _build_stable_contour_geometry(image, profile: AugmentationProfile) -> dict:
    """Build source-image contour geometry before visual effects mutate pixels."""
    if np is None or cv2 is None:
        return {}
    try:
        return _build_contour_blanket_surface(
            image,
            (profile or AugmentationProfile()).normalized(),
            None,
            relief_mask=None,
            include_water=False,
            base_surface=None,
        )
    except Exception:
        return {}


def _amplify_sample_profile_randomness(
    base: AugmentationProfile,
    jittered: AugmentationProfile,
    rng: random.Random,
    mode: str,
) -> AugmentationProfile:
    """Apply wider realistic scene variation on top of the legacy light jitter."""
    scale = _augmentation_randomness_extra_scale(mode)
    if scale <= 0:
        return jittered

    def vary(
        value: float,
        *,
        minimum: float,
        maximum: float,
        sigma_abs: float,
        sigma_ratio: float = 0.10,
        max_delta_abs: float | None = None,
        max_delta_ratio: float = 0.22,
        zero_stays_zero: bool = True,
    ) -> float:
        return _jitter_float(
            value,
            rng,
            minimum=minimum,
            maximum=maximum,
            sigma_abs=sigma_abs * scale,
            sigma_ratio=sigma_ratio * scale,
            max_delta_abs=(sigma_abs * 2.8 if max_delta_abs is None else max_delta_abs) * scale,
            max_delta_ratio=max_delta_ratio * scale,
            zero_stays_zero=zero_stays_zero,
        )

    def vary_int(
        value: int,
        *,
        minimum: int,
        maximum: int,
        sigma_abs: float,
        sigma_ratio: float = 0.10,
        max_delta_ratio: float = 0.22,
    ) -> int:
        return _jitter_int(
            value,
            rng,
            minimum=minimum,
            maximum=maximum,
            sigma_abs=sigma_abs * scale,
            sigma_ratio=sigma_ratio * scale,
            max_delta_ratio=max_delta_ratio * scale,
        )

    def vary_coord(value: float, default: float, spread: float) -> float:
        try:
            current = float(value)
        except Exception:
            current = -1.0
        anchor = float(default) if current < 0.0 else current
        return _jitter_float(
            anchor,
            rng,
            minimum=0.0,
            maximum=1.0,
            sigma_abs=spread * 0.42 * scale,
            sigma_ratio=0.0,
            max_delta_abs=spread * scale,
            max_delta_ratio=0.0,
            zero_stays_zero=False,
        )

    def vary_world(value: float, *, minimum: float, maximum: float, spread: float) -> float:
        try:
            current = float(value)
        except Exception:
            current = -999.0
        if current <= -998.0:
            return current
        return _jitter_float(
            current,
            rng,
            minimum=minimum,
            maximum=maximum,
            sigma_abs=spread * 0.42 * scale,
            sigma_ratio=0.0,
            max_delta_abs=spread * scale,
            max_delta_ratio=0.0,
            zero_stays_zero=False,
        )

    updates: dict[str, object] = {
        "brightness_limit": vary(jittered.brightness_limit, minimum=0.0, maximum=0.25, sigma_abs=0.018, max_delta_abs=0.060),
        "contrast_limit": vary(jittered.contrast_limit, minimum=0.0, maximum=0.25, sigma_abs=0.018, max_delta_abs=0.060),
        "noise_strength": vary(jittered.noise_strength, minimum=0.0, maximum=0.08, sigma_abs=0.006, max_delta_abs=0.020),
        "blur_strength": vary(jittered.blur_strength, minimum=0.0, maximum=1.0, sigma_abs=0.035, max_delta_abs=0.110),
        "vehicle_speed": vary(jittered.vehicle_speed, minimum=0.0, maximum=1.0, sigma_abs=0.032, max_delta_abs=0.110),
        "flare_strength": vary(jittered.flare_strength, minimum=0.0, maximum=1.0, sigma_abs=0.032, max_delta_abs=0.105),
        "overexposure_strength": vary(jittered.overexposure_strength, minimum=0.0, maximum=1.0, sigma_abs=0.032, max_delta_abs=0.105),
    }
    if abs(float(base.saturation_limit or 0.0) - 1.0) > 0.001:
        updates["saturation_limit"] = vary(
            jittered.saturation_limit,
            minimum=0.0,
            maximum=3.0,
            sigma_abs=0.080,
            sigma_ratio=0.08,
            max_delta_abs=0.240,
            max_delta_ratio=0.22,
            zero_stays_zero=False,
        )
    if float(base.rain_strength or 0.0) > 0.001:
        rain_min, rain_max = _jitter_range_pair(
            jittered.rain_drop_size_min,
            jittered.rain_drop_size_max,
            rng,
            minimum=0.0,
            maximum=1.0,
            sigma_abs=0.028 * scale,
            sigma_ratio=0.08 * scale,
        )
        updates.update(
            {
                "rain_strength": vary(jittered.rain_strength, minimum=0.0, maximum=1.0, sigma_abs=0.045, max_delta_abs=0.150),
                "rain_drop_size_min": rain_min,
                "rain_drop_size_max": rain_max,
                "rain_drop_size": (rain_min + rain_max) / 2.0,
                "rain_vector_field_strength": vary(jittered.rain_vector_field_strength, minimum=0.0, maximum=1.0, sigma_abs=0.050, max_delta_abs=0.160),
                "rain_vortex_strength": vary(jittered.rain_vortex_strength, minimum=0.0, maximum=1.0, sigma_abs=0.050, max_delta_abs=0.160),
                "rain_alpha": vary(jittered.rain_alpha, minimum=0.0, maximum=1.0, sigma_abs=0.035, max_delta_abs=0.110, zero_stays_zero=False),
                "rain_lens_strength": vary(jittered.rain_lens_strength, minimum=0.0, maximum=1.0, sigma_abs=0.050, max_delta_abs=0.160),
                "rain_edge_mist_strength": vary(jittered.rain_edge_mist_strength, minimum=0.0, maximum=1.0, sigma_abs=0.050, max_delta_abs=0.160),
                "rain_edge_mist_radius": vary(jittered.rain_edge_mist_radius, minimum=0.0, maximum=1.0, sigma_abs=0.035, max_delta_abs=0.120, zero_stays_zero=False),
                "tyndall_strength": vary(jittered.tyndall_strength, minimum=0.0, maximum=1.0, sigma_abs=0.040, max_delta_abs=0.130, zero_stays_zero=False),
            }
        )
    if float(base.night_strength or 0.0) > 0.001 or float(base.night_light_strength or 0.0) > 0.001:
        updates.update(
            {
                "night_strength": vary(jittered.night_strength, minimum=0.0, maximum=1.0, sigma_abs=0.040, max_delta_abs=0.140),
                "night_light_strength": vary(jittered.night_light_strength, minimum=0.0, maximum=1.0, sigma_abs=0.040, max_delta_abs=0.140),
                "night_bloom_strength": vary(jittered.night_bloom_strength, minimum=0.0, maximum=1.0, sigma_abs=0.040, max_delta_abs=0.140),
                "night_iso_noise_strength": vary(jittered.night_iso_noise_strength, minimum=0.0, maximum=1.0, sigma_abs=0.040, max_delta_abs=0.140),
                "night_light_warmth": vary(jittered.night_light_warmth, minimum=0.0, maximum=1.0, sigma_abs=0.035, max_delta_abs=0.110, zero_stays_zero=False),
            }
        )

    for index in (1, 2, 3):
        if _active_headlight_strength(base, index) <= 0.001:
            continue
        source_x, source_y, target_x, target_y = _headlight_default_geometry(index)
        spread = 0.11 if mode == "realistic" else 0.17
        strength_name = _headlight_field_name(index, "strength")
        updates[strength_name] = vary(getattr(jittered, strength_name), minimum=0.0, maximum=1.0, sigma_abs=0.050, max_delta_abs=0.170)
        updates[_headlight_field_name(index, "source_x")] = vary_coord(getattr(jittered, _headlight_field_name(index, "source_x")), source_x, spread)
        updates[_headlight_field_name(index, "source_y")] = vary_coord(getattr(jittered, _headlight_field_name(index, "source_y")), source_y, spread)
        updates[_headlight_field_name(index, "target_x")] = vary_coord(getattr(jittered, _headlight_field_name(index, "target_x")), target_x, spread * 0.9)
        updates[_headlight_field_name(index, "target_y")] = vary_coord(getattr(jittered, _headlight_field_name(index, "target_y")), target_y, spread * 0.9)
        updates[_scene_headlight_world_name(index, "source", "x")] = vary_world(
            getattr(jittered, _scene_headlight_world_name(index, "source", "x")),
            minimum=-3.0,
            maximum=3.0,
            spread=0.11 if mode == "realistic" else 0.18,
        )
        updates[_scene_headlight_world_name(index, "source", "y")] = vary_world(
            getattr(jittered, _scene_headlight_world_name(index, "source", "y")),
            minimum=-3.0,
            maximum=3.0,
            spread=0.11 if mode == "realistic" else 0.18,
        )
        updates[_scene_headlight_world_name(index, "source", "z")] = vary_world(
            getattr(jittered, _scene_headlight_world_name(index, "source", "z")),
            minimum=0.0,
            maximum=8.0,
            spread=0.10 if mode == "realistic" else 0.16,
        )
        updates[_scene_headlight_world_name(index, "target", "x")] = vary_world(
            getattr(jittered, _scene_headlight_world_name(index, "target", "x")),
            minimum=-0.5,
            maximum=0.5,
            spread=0.045 if mode == "realistic" else 0.075,
        )
        updates[_scene_headlight_world_name(index, "target", "y")] = vary_world(
            getattr(jittered, _scene_headlight_world_name(index, "target", "y")),
            minimum=-0.12,
            maximum=0.12,
            spread=0.018 if mode == "realistic" else 0.035,
        )
        updates[_scene_headlight_world_name(index, "target", "z")] = 0.0
        for suffix in ("warmth", "r", "g", "b"):
            name = _headlight_field_name(index, suffix)
            value = float(getattr(jittered, name, -1.0) if getattr(jittered, name, None) is not None else -1.0)
            if value >= 0.0:
                updates[name] = vary(value, minimum=0.0, maximum=1.0, sigma_abs=0.045, max_delta_abs=0.140, zero_stays_zero=False)
        updates[_headlight_field_name(index, "cone")] = vary(
            getattr(jittered, _headlight_field_name(index, "cone")),
            minimum=0.02,
            maximum=2.5,
            sigma_abs=0.060,
            max_delta_abs=0.190,
            zero_stays_zero=False,
        )
        updates[_headlight_field_name(index, "source_radius")] = vary(
            getattr(jittered, _headlight_field_name(index, "source_radius")),
            minimum=0.0,
            maximum=2.5,
            sigma_abs=0.045,
            max_delta_abs=0.150,
        )

    if float(base.dirt_flow_points or 0) > 0:
        mass_min, mass_max = _jitter_range_pair(
            jittered.dirt_flow_mass_min,
            jittered.dirt_flow_mass_max,
            rng,
            minimum=0.05,
            maximum=2.8,
            sigma_abs=0.080 * scale,
            sigma_ratio=0.08 * scale,
        )
        opacity_min, opacity_max = _jitter_range_pair(
            jittered.dirt_flow_opacity_min,
            jittered.dirt_flow_opacity_max,
            rng,
            minimum=0.0,
            maximum=1.0,
            sigma_abs=0.045 * scale,
            sigma_ratio=0.08 * scale,
        )
        updates.update(
            {
                "dirt_flow_points": vary_int(jittered.dirt_flow_points, minimum=0, maximum=MAX_DIRT_FLOW_POINTS, sigma_abs=5.0),
                "dirt_flow_mass_min": mass_min,
                "dirt_flow_mass_max": mass_max,
                "dirt_flow_splash_scale": vary(jittered.dirt_flow_splash_scale, minimum=0.0, maximum=1.0, sigma_abs=0.040, max_delta_abs=0.130, zero_stays_zero=False),
                "dirt_flow_trail_length": vary(jittered.dirt_flow_trail_length, minimum=0.0, maximum=1.0, sigma_abs=0.040, max_delta_abs=0.130, zero_stays_zero=False),
                "dirt_flow_humidity": vary(jittered.dirt_flow_humidity, minimum=0.0, maximum=1.0, sigma_abs=0.040, max_delta_abs=0.130, zero_stays_zero=False),
                "dirt_flow_air_angle": vary(jittered.dirt_flow_air_angle, minimum=-180.0, maximum=180.0, sigma_abs=5.0, max_delta_abs=15.0, zero_stays_zero=False),
                "dirt_flow_wind_strength": vary(jittered.dirt_flow_wind_strength, minimum=0.0, maximum=1.0, sigma_abs=0.040, max_delta_abs=0.130, zero_stays_zero=False),
                "dirt_flow_opacity_min": opacity_min,
                "dirt_flow_opacity_max": opacity_max,
            }
        )

    if float(getattr(base, "water_film_strength", 0.0) or 0.0) > 0.001:
        updates.update(
            {
                "water_film_strength": vary(jittered.water_film_strength, minimum=0.0, maximum=1.0, sigma_abs=0.050, max_delta_abs=0.170),
                "water_film_unevenness": vary(jittered.water_film_unevenness, minimum=0.0, maximum=1.0, sigma_abs=0.055, max_delta_abs=0.180, zero_stays_zero=False),
                "water_film_lens_strength": vary(jittered.water_film_lens_strength, minimum=0.0, maximum=1.0, sigma_abs=0.050, max_delta_abs=0.160),
                "water_film_contour_response": vary(jittered.water_film_contour_response, minimum=0.0, maximum=1.0, sigma_abs=0.045, max_delta_abs=0.140, zero_stays_zero=False),
                "water_film_gloss_strength": vary(jittered.water_film_gloss_strength, minimum=0.0, maximum=1.0, sigma_abs=0.050, max_delta_abs=0.170, zero_stays_zero=False),
            }
        )

    if float(base.plate_reflect_gradient_strength or 0.0) > 0.001 or float(base.plate_reflect_glare_strength or 0.0) > 0.001:
        updates.update(
            {
                "dark_relief_strength": vary(jittered.dark_relief_strength, minimum=0.0, maximum=3.0, sigma_abs=0.090, max_delta_abs=0.280),
                "overhang_shadow_strength": vary(jittered.overhang_shadow_strength, minimum=0.0, maximum=1.0, sigma_abs=0.050, max_delta_abs=0.160),
                "overhang_shadow_depth": vary(jittered.overhang_shadow_depth, minimum=0.0, maximum=0.60, sigma_abs=0.040, max_delta_abs=0.130, zero_stays_zero=False),
                "overhang_shadow_skew": vary(jittered.overhang_shadow_skew, minimum=-1.0, maximum=1.0, sigma_abs=0.070, max_delta_abs=0.220),
                "plate_reflect_gradient_strength": vary(jittered.plate_reflect_gradient_strength, minimum=0.0, maximum=2.0, sigma_abs=0.080, max_delta_abs=0.260),
                "plate_reflect_glare_strength": vary(jittered.plate_reflect_glare_strength, minimum=0.0, maximum=2.0, sigma_abs=0.080, max_delta_abs=0.260),
                "plate_reflect_curve_strength": vary(jittered.plate_reflect_curve_strength, minimum=0.0, maximum=2.0, sigma_abs=0.080, max_delta_abs=0.260),
                "relief_bounce_strength": vary(jittered.relief_bounce_strength, minimum=0.0, maximum=1.0, sigma_abs=0.040, max_delta_abs=0.130, zero_stays_zero=False),
            }
        )

    return replace(jittered, **updates)


def _manual_randomness_field_amount(config: dict, group_key: str, field_key: str) -> float:
    groups = config.get("groups") if isinstance(config, dict) else {}
    group = groups.get(group_key, {}) if isinstance(groups, dict) else {}
    if not isinstance(group, dict) or not bool(group.get("enabled")):
        return 0.0
    fields = group.get("fields", {})
    field_state = fields.get(field_key, {}) if isinstance(fields, dict) else {}
    if not isinstance(field_state, dict) or not bool(field_state.get("enabled")):
        return 0.0
    return (_percent(group.get("amount", 100.0), 100.0) / 100.0) * (
        _percent(field_state.get("amount", 100.0), 100.0) / 100.0
    )


def _blend_manual_randomness_value(base_value: object, jittered_value: object, amount: float) -> object:
    amount = max(0.0, min(1.0, float(amount or 0.0)))
    if amount <= 0.0001:
        return base_value
    if amount >= 0.9999:
        return jittered_value
    try:
        if isinstance(base_value, int) and not isinstance(base_value, bool):
            return int(round(float(base_value) + (float(jittered_value) - float(base_value)) * amount))
        return float(base_value) + (float(jittered_value) - float(base_value)) * amount
    except Exception:
        return jittered_value if amount >= 0.5 else base_value


def _apply_manual_randomness_config(base: AugmentationProfile, jittered: AugmentationProfile) -> AugmentationProfile:
    target = _profile_task_target(base)
    config = normalize_manual_randomness_config(getattr(base, "manual_randomness", {}), target)
    updates: dict[str, object] = {}
    for group in manual_randomness_group_specs(target):
        group_key = str(group.get("key") or "")
        for field in group.get("fields", []):
            field_key = str(field.get("key") or "")
            if not field_key or not hasattr(base, field_key) or not hasattr(jittered, field_key):
                continue
            amount = _manual_randomness_field_amount(config, group_key, field_key)
            updates[field_key] = _blend_manual_randomness_value(
                getattr(base, field_key),
                getattr(jittered, field_key),
                amount,
            )
    if target == "char":
        updates.update({"rotation_limit": 0.0, "translate_limit": 0.0, "scale_limit": 0.0})
    return replace(base, **updates).normalized()


def _jitter_augmentation_profile(profile: AugmentationProfile, rng: random.Random) -> AugmentationProfile:
    """Return a per-sample variation around the user profile.

    The profile set in the modal is treated as the expected value. Dataset
    generation only varies active effects, so the synthetic split gains natural
    diversity without drifting away from the user's intent.
    """
    base = (profile or AugmentationProfile()).normalized()
    randomness_mode = normalize_augmentation_randomness_mode(getattr(base, "randomness_mode", "realistic"))
    if randomness_mode == "fixed":
        return base
    effective_randomness_mode = "realistic" if randomness_mode == "manual" else randomness_mode
    rain_min, rain_max = _jitter_range_pair(
        base.rain_drop_size_min,
        base.rain_drop_size_max,
        rng,
        minimum=0.0,
        maximum=1.0,
        sigma_abs=0.010,
        sigma_ratio=0.05,
    )
    mass_min, mass_max = _jitter_range_pair(
        base.dirt_flow_mass_min,
        base.dirt_flow_mass_max,
        rng,
        minimum=0.05,
        maximum=2.8,
        sigma_abs=0.035,
        sigma_ratio=0.045,
    )
    opacity_min, opacity_max = _jitter_range_pair(
        base.dirt_flow_opacity_min,
        base.dirt_flow_opacity_max,
        rng,
        minimum=0.0,
        maximum=1.0,
        sigma_abs=0.018,
        sigma_ratio=0.05,
    )
    stickiness_min, stickiness_max = _jitter_range_pair(
        base.dirt_flow_stickiness_min,
        base.dirt_flow_stickiness_max,
        rng,
        minimum=0.0,
        maximum=1.0,
        sigma_abs=0.018,
        sigma_ratio=0.05,
    )

    jittered = replace(
        base,
        rotation_limit=_jitter_float(base.rotation_limit, rng, minimum=-15.0, maximum=15.0, sigma_abs=1.25, sigma_ratio=0.06, max_delta_abs=3.0, max_delta_ratio=0.16),
        translate_limit=_jitter_float(base.translate_limit, rng, minimum=0.0, maximum=0.03, sigma_abs=0.0015, sigma_ratio=0.10, max_delta_abs=0.004, max_delta_ratio=0.20),
        scale_limit=_jitter_float(base.scale_limit, rng, minimum=0.0, maximum=0.08, sigma_abs=0.003, sigma_ratio=0.10, max_delta_abs=0.010, max_delta_ratio=0.20),
        brightness_limit=_jitter_float(base.brightness_limit, rng, minimum=0.0, maximum=0.25, sigma_abs=0.010, sigma_ratio=0.08, max_delta_abs=0.025, max_delta_ratio=0.16),
        contrast_limit=_jitter_float(base.contrast_limit, rng, minimum=0.0, maximum=0.25, sigma_abs=0.010, sigma_ratio=0.08, max_delta_abs=0.025, max_delta_ratio=0.16),
        saturation_limit=(
            1.0
            if abs(float(base.saturation_limit or 0.0) - 1.0) <= 0.001
            else _jitter_float(base.saturation_limit, rng, minimum=0.0, maximum=3.0, sigma_abs=0.030, sigma_ratio=0.06, max_delta_abs=0.090, max_delta_ratio=0.12)
        ),
        noise_strength=_jitter_float(base.noise_strength, rng, minimum=0.0, maximum=0.08, sigma_abs=0.003, sigma_ratio=0.10, max_delta_abs=0.010, max_delta_ratio=0.22),
        noise_grain_size=_jitter_int(base.noise_grain_size, rng, minimum=1, maximum=12, sigma_abs=0.65, sigma_ratio=0.08, max_delta_ratio=0.22, zero_stays_zero=False),
        rain_strength=_jitter_float(base.rain_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.08, max_delta_abs=0.055, max_delta_ratio=0.16),
        rain_drop_size_min=rain_min,
        rain_drop_size_max=rain_max,
        rain_drop_size=(rain_min + rain_max) / 2.0,
        rain_vector_field_strength=_jitter_float(base.rain_vector_field_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.055, max_delta_ratio=0.16),
        rain_vortex_strength=_jitter_float(base.rain_vortex_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.055, max_delta_ratio=0.16),
        rain_alpha=_jitter_float(base.rain_alpha, rng, minimum=0.0, maximum=1.0, sigma_abs=0.015, sigma_ratio=0.05, max_delta_abs=0.045, max_delta_ratio=0.12, zero_stays_zero=False),
        rain_lens_strength=_jitter_float(base.rain_lens_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.055, max_delta_ratio=0.16),
        rain_edge_mist_strength=_jitter_float(base.rain_edge_mist_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.055, max_delta_ratio=0.16),
        rain_edge_mist_radius=_jitter_float(base.rain_edge_mist_radius, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.05, max_delta_abs=0.050, max_delta_ratio=0.12, zero_stays_zero=False),
        tyndall_strength=_jitter_float(base.tyndall_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.07, max_delta_abs=0.050, max_delta_ratio=0.14),
        wet_reflection_strength=_jitter_float(base.wet_reflection_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.08, max_delta_abs=0.050, max_delta_ratio=0.16),
        vehicle_speed=_jitter_float(base.vehicle_speed, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.07, max_delta_abs=0.050, max_delta_ratio=0.14),
        night_strength=_jitter_float(base.night_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.07, max_delta_abs=0.050, max_delta_ratio=0.14),
        night_light_strength=_jitter_float(base.night_light_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.07, max_delta_abs=0.050, max_delta_ratio=0.14),
        night_bloom_strength=_jitter_float(base.night_bloom_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.07, max_delta_abs=0.050, max_delta_ratio=0.14),
        night_iso_noise_strength=_jitter_float(base.night_iso_noise_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.08, max_delta_abs=0.050, max_delta_ratio=0.16),
        night_light_warmth=_jitter_float(base.night_light_warmth, rng, minimum=0.0, maximum=1.0, sigma_abs=0.015, sigma_ratio=0.05, max_delta_abs=0.045, max_delta_ratio=0.12, zero_stays_zero=False),
        scene_plate_width=1.0,
        scene_plate_height=0.24,
        scene_camera_x=0.0,
        scene_camera_y=0.0,
        scene_camera_z=_jitter_float(base.scene_camera_z, rng, minimum=0.35, maximum=4.0, sigma_abs=0.035, sigma_ratio=0.020, max_delta_abs=0.100, max_delta_ratio=0.050, zero_stays_zero=False),
        scene_camera_target_x=0.0,
        scene_camera_target_y=0.0,
        scene_camera_target_z=0.0,
        scene_plate_texture_enabled=bool(getattr(base, "scene_plate_texture_enabled", True)),
        traffic_headlight_strength=_jitter_float(base.traffic_headlight_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.060, max_delta_ratio=0.16),
        traffic_headlight_source_x=_jitter_coordinate(base.traffic_headlight_source_x, rng),
        traffic_headlight_source_y=_jitter_coordinate(base.traffic_headlight_source_y, rng),
        traffic_headlight_target_x=_jitter_coordinate(base.traffic_headlight_target_x, rng),
        traffic_headlight_target_y=_jitter_coordinate(base.traffic_headlight_target_y, rng),
        traffic_headlight_source_world_x=_jitter_world_coordinate(base.traffic_headlight_source_world_x, rng),
        traffic_headlight_source_world_y=_jitter_world_coordinate(base.traffic_headlight_source_world_y, rng),
        traffic_headlight_source_world_z=_jitter_float(base.traffic_headlight_source_world_z, rng, minimum=0.0, maximum=8.0, sigma_abs=0.035, sigma_ratio=0.025, max_delta_abs=0.110, max_delta_ratio=0.070, zero_stays_zero=False),
        traffic_headlight_target_world_x=_jitter_world_coordinate(base.traffic_headlight_target_world_x, rng),
        traffic_headlight_target_world_y=_jitter_world_coordinate(base.traffic_headlight_target_world_y, rng),
        traffic_headlight_target_world_z=0.0,
        traffic_headlight_1_warmth=_jitter_optional_unit(base.traffic_headlight_1_warmth, rng),
        traffic_headlight_1_r=_jitter_optional_unit(base.traffic_headlight_1_r, rng),
        traffic_headlight_1_g=_jitter_optional_unit(base.traffic_headlight_1_g, rng),
        traffic_headlight_1_b=_jitter_optional_unit(base.traffic_headlight_1_b, rng),
        traffic_headlight_1_cone=_jitter_float(base.traffic_headlight_1_cone, rng, minimum=0.02, maximum=2.5, sigma_abs=0.026, sigma_ratio=0.05, max_delta_abs=0.080, max_delta_ratio=0.12, zero_stays_zero=False),
        traffic_headlight_1_source_radius=_jitter_float(base.traffic_headlight_1_source_radius, rng, minimum=0.0, maximum=2.5, sigma_abs=0.016, sigma_ratio=0.07, max_delta_abs=0.060, max_delta_ratio=0.16),
        traffic_headlight_2_strength=_jitter_float(base.traffic_headlight_2_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.060, max_delta_ratio=0.16),
        traffic_headlight_2_warmth=_jitter_optional_unit(base.traffic_headlight_2_warmth, rng, allow_negative_sentinel=False),
        traffic_headlight_2_r=_jitter_optional_unit(base.traffic_headlight_2_r, rng),
        traffic_headlight_2_g=_jitter_optional_unit(base.traffic_headlight_2_g, rng),
        traffic_headlight_2_b=_jitter_optional_unit(base.traffic_headlight_2_b, rng),
        traffic_headlight_2_cone=_jitter_float(base.traffic_headlight_2_cone, rng, minimum=0.02, maximum=2.5, sigma_abs=0.026, sigma_ratio=0.05, max_delta_abs=0.080, max_delta_ratio=0.12, zero_stays_zero=False),
        traffic_headlight_2_source_radius=_jitter_float(base.traffic_headlight_2_source_radius, rng, minimum=0.0, maximum=2.5, sigma_abs=0.016, sigma_ratio=0.07, max_delta_abs=0.060, max_delta_ratio=0.16),
        traffic_headlight_2_source_x=_jitter_coordinate(base.traffic_headlight_2_source_x, rng),
        traffic_headlight_2_source_y=_jitter_coordinate(base.traffic_headlight_2_source_y, rng),
        traffic_headlight_2_target_x=_jitter_coordinate(base.traffic_headlight_2_target_x, rng),
        traffic_headlight_2_target_y=_jitter_coordinate(base.traffic_headlight_2_target_y, rng),
        traffic_headlight_2_source_world_x=_jitter_world_coordinate(base.traffic_headlight_2_source_world_x, rng),
        traffic_headlight_2_source_world_y=_jitter_world_coordinate(base.traffic_headlight_2_source_world_y, rng),
        traffic_headlight_2_source_world_z=_jitter_float(base.traffic_headlight_2_source_world_z, rng, minimum=0.0, maximum=8.0, sigma_abs=0.035, sigma_ratio=0.025, max_delta_abs=0.110, max_delta_ratio=0.070, zero_stays_zero=False),
        traffic_headlight_2_target_world_x=_jitter_world_coordinate(base.traffic_headlight_2_target_world_x, rng),
        traffic_headlight_2_target_world_y=_jitter_world_coordinate(base.traffic_headlight_2_target_world_y, rng),
        traffic_headlight_2_target_world_z=0.0,
        traffic_headlight_3_strength=_jitter_float(base.traffic_headlight_3_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.060, max_delta_ratio=0.16),
        traffic_headlight_3_warmth=_jitter_optional_unit(base.traffic_headlight_3_warmth, rng, allow_negative_sentinel=False),
        traffic_headlight_3_r=_jitter_optional_unit(base.traffic_headlight_3_r, rng),
        traffic_headlight_3_g=_jitter_optional_unit(base.traffic_headlight_3_g, rng),
        traffic_headlight_3_b=_jitter_optional_unit(base.traffic_headlight_3_b, rng),
        traffic_headlight_3_cone=_jitter_float(base.traffic_headlight_3_cone, rng, minimum=0.02, maximum=2.5, sigma_abs=0.026, sigma_ratio=0.05, max_delta_abs=0.080, max_delta_ratio=0.12, zero_stays_zero=False),
        traffic_headlight_3_source_radius=_jitter_float(base.traffic_headlight_3_source_radius, rng, minimum=0.0, maximum=2.5, sigma_abs=0.016, sigma_ratio=0.07, max_delta_abs=0.060, max_delta_ratio=0.16),
        traffic_headlight_3_source_x=_jitter_coordinate(base.traffic_headlight_3_source_x, rng),
        traffic_headlight_3_source_y=_jitter_coordinate(base.traffic_headlight_3_source_y, rng),
        traffic_headlight_3_target_x=_jitter_coordinate(base.traffic_headlight_3_target_x, rng),
        traffic_headlight_3_target_y=_jitter_coordinate(base.traffic_headlight_3_target_y, rng),
        traffic_headlight_3_source_world_x=_jitter_world_coordinate(base.traffic_headlight_3_source_world_x, rng),
        traffic_headlight_3_source_world_y=_jitter_world_coordinate(base.traffic_headlight_3_source_world_y, rng),
        traffic_headlight_3_source_world_z=_jitter_float(base.traffic_headlight_3_source_world_z, rng, minimum=0.0, maximum=8.0, sigma_abs=0.035, sigma_ratio=0.025, max_delta_abs=0.110, max_delta_ratio=0.070, zero_stays_zero=False),
        traffic_headlight_3_target_world_x=_jitter_world_coordinate(base.traffic_headlight_3_target_world_x, rng),
        traffic_headlight_3_target_world_y=_jitter_world_coordinate(base.traffic_headlight_3_target_world_y, rng),
        traffic_headlight_3_target_world_z=0.0,
        wet_mud_gloss_strength=_jitter_float(base.wet_mud_gloss_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.060, max_delta_ratio=0.16),
        water_film_strength=_jitter_float(base.water_film_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.060, max_delta_ratio=0.16),
        water_film_unevenness=_jitter_float(base.water_film_unevenness, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.07, max_delta_abs=0.055, max_delta_ratio=0.15, zero_stays_zero=False),
        water_film_lens_strength=_jitter_float(base.water_film_lens_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.08, max_delta_abs=0.055, max_delta_ratio=0.16),
        water_film_contour_response=_jitter_float(base.water_film_contour_response, rng, minimum=0.0, maximum=1.0, sigma_abs=0.016, sigma_ratio=0.06, max_delta_abs=0.050, max_delta_ratio=0.14, zero_stays_zero=False),
        water_film_gloss_strength=_jitter_float(base.water_film_gloss_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.07, max_delta_abs=0.055, max_delta_ratio=0.15, zero_stays_zero=False),
        flare_strength=_jitter_float(base.flare_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.055, max_delta_ratio=0.16),
        overexposure_strength=_jitter_float(base.overexposure_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.055, max_delta_ratio=0.16),
        dirt_streak_strength=_jitter_float(base.dirt_streak_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.055, max_delta_ratio=0.16),
        dirt_flow_strength=_jitter_float(base.dirt_flow_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.060, max_delta_ratio=0.16),
        dirt_flow_points=_jitter_int(base.dirt_flow_points, rng, minimum=0, maximum=MAX_DIRT_FLOW_POINTS, sigma_abs=2.0, sigma_ratio=0.06, max_delta_ratio=0.14),
        dirt_flow_mass_min=mass_min,
        dirt_flow_mass_max=mass_max,
        dirt_flow_splash_scale=_jitter_float(base.dirt_flow_splash_scale, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.06, max_delta_abs=0.055, max_delta_ratio=0.14, zero_stays_zero=False),
        dirt_flow_trail_length=_jitter_float(base.dirt_flow_trail_length, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.06, max_delta_abs=0.055, max_delta_ratio=0.14, zero_stays_zero=False),
        dirt_flow_humidity=_jitter_float(base.dirt_flow_humidity, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.06, max_delta_abs=0.055, max_delta_ratio=0.14, zero_stays_zero=False),
        dirt_flow_stickiness_min=stickiness_min,
        dirt_flow_stickiness_max=stickiness_max,
        dirt_flow_stickiness=(stickiness_min + stickiness_max) / 2.0,
        dirt_flow_air_angle=_jitter_float(base.dirt_flow_air_angle, rng, minimum=-180.0, maximum=180.0, sigma_abs=2.0, sigma_ratio=0.04, max_delta_abs=5.0, max_delta_ratio=0.10),
        dirt_flow_wind_strength=_jitter_float(base.dirt_flow_wind_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.018, sigma_ratio=0.06, max_delta_abs=0.055, max_delta_ratio=0.14, zero_stays_zero=False),
        dirt_flow_opacity_min=opacity_min,
        dirt_flow_opacity_max=opacity_max,
        dark_relief_strength=_jitter_float(base.dark_relief_strength, rng, minimum=0.0, maximum=3.0, sigma_abs=0.040, sigma_ratio=0.08, max_delta_abs=0.140, max_delta_ratio=0.16),
        dark_relief_light_angle=135.0,
        relief_bounce_depth=base.relief_bounce_depth,
        relief_bounce_strength=_jitter_float(base.relief_bounce_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.060, max_delta_ratio=0.16, zero_stays_zero=False),
        light_normal_strength=base.light_normal_strength,
        overhang_shadow_strength=_jitter_float(base.overhang_shadow_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.060, max_delta_ratio=0.16),
        overhang_shadow_depth=_jitter_float(base.overhang_shadow_depth, rng, minimum=0.0, maximum=0.60, sigma_abs=0.018, sigma_ratio=0.05, max_delta_abs=0.055, max_delta_ratio=0.12, zero_stays_zero=False),
        overhang_shadow_skew=_jitter_float(base.overhang_shadow_skew, rng, minimum=-1.0, maximum=1.0, sigma_abs=0.030, sigma_ratio=0.05, max_delta_abs=0.080, max_delta_ratio=0.14),
        plate_reflect_gradient_strength=_jitter_float(base.plate_reflect_gradient_strength, rng, minimum=0.0, maximum=2.0, sigma_abs=0.035, sigma_ratio=0.07, max_delta_abs=0.110, max_delta_ratio=0.15),
        plate_reflect_glare_strength=_jitter_float(base.plate_reflect_glare_strength, rng, minimum=0.0, maximum=2.0, sigma_abs=0.035, sigma_ratio=0.07, max_delta_abs=0.110, max_delta_ratio=0.15),
        plate_reflect_curve_strength=_jitter_float(base.plate_reflect_curve_strength, rng, minimum=0.0, maximum=2.0, sigma_abs=0.035, sigma_ratio=0.07, max_delta_abs=0.110, max_delta_ratio=0.15),
        blur_strength=_jitter_float(base.blur_strength, rng, minimum=0.0, maximum=1.0, sigma_abs=0.020, sigma_ratio=0.08, max_delta_abs=0.060, max_delta_ratio=0.16),
    )
    jittered = _amplify_sample_profile_randomness(base, jittered, rng, effective_randomness_mode)
    if _profile_task_target(base) == "char":
        jittered = replace(jittered, rotation_limit=0.0, translate_limit=0.0, scale_limit=0.0)
    if randomness_mode == "manual":
        jittered = _apply_manual_randomness_config(base, jittered)
    return jittered.normalized()


def _image_extensions() -> set[str]:
    return {str(ext).lower() for ext in CONFIG.IMAGE_EXTENSIONS}


def _load_dataset_config(dataset_dir: Path) -> dict:
    yaml_path = Path(dataset_dir) / "data.yaml"
    if not yaml_path.exists():
        return {}
    return safe_load_yaml(yaml_path)


def _parse_kpt_shape(config: dict) -> tuple[int, int]:
    raw = config.get("kpt_shape")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return int(raw[0]), int(raw[1])
        except Exception:
            return 0, 0
    if isinstance(raw, str):
        digits = []
        for chunk in raw.replace("[", " ").replace("]", " ").replace(",", " ").split():
            try:
                digits.append(int(float(chunk)))
            except Exception:
                continue
        if len(digits) >= 2:
            return digits[0], digits[1]
    return 0, 0


def _coerce_names(config: dict) -> dict[int, str]:
    names = config.get("names", {})
    result: dict[int, str] = {}
    if isinstance(names, dict):
        for key, value in names.items():
            try:
                idx = int(key)
            except Exception:
                continue
            result[idx] = str(value)
    elif isinstance(names, list):
        for idx, value in enumerate(names):
            result[idx] = str(value)
    return result


def _dump_dataset_config(dataset_dir: Path, config: dict) -> None:
    yaml_path = Path(dataset_dir) / "data.yaml"
    if YAML_AVAILABLE and yaml is not None:
        yaml_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return

    lines: list[str] = []
    for key in ("path", "train", "val", "test", "nc"):
        if key in config:
            lines.append(f"{key}: {config[key]}")
    names = _coerce_names(config)
    if names:
        lines.append("names:")
        for idx in sorted(names):
            lines.append(f"  {idx}: {names[idx]}")
    if "kpt_shape" in config:
        lines.append(f"kpt_shape: {config['kpt_shape']}")
    if "flip_idx" in config:
        lines.append(f"flip_idx: {config['flip_idx']}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_yolo_dataset_yaml_points_to_root(dataset_dir: Path) -> tuple[bool, str, bool]:
    """Ensure data.yaml resolves relative splits from the dataset directory itself."""

    dataset_dir = Path(dataset_dir)
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return False, "Brak data.yaml.", False

    try:
        dataset_root = dataset_dir.resolve()
    except Exception:
        dataset_root = dataset_dir

    try:
        config = _load_dataset_config(dataset_dir)
    except Exception as exc:
        return False, f"Nie udało się odczytać data.yaml: {exc}", False
    if not isinstance(config, dict):
        config = {}

    def _safe_resolve(path: Path) -> Path:
        try:
            return path.resolve()
        except Exception:
            return path

    def _is_inside_dataset(path: Path) -> bool:
        try:
            _safe_resolve(path).relative_to(dataset_root)
            return True
        except Exception:
            return False

    raw_path = str(config.get("path") or "").strip()
    current_path = None
    if raw_path:
        try:
            raw_candidate = Path(raw_path)
            if not raw_candidate.is_absolute():
                raw_candidate = dataset_dir / raw_candidate
            current_path = _safe_resolve(raw_candidate)
        except Exception:
            current_path = Path(raw_path)

    needs_rewrite = current_path is not None and current_path != dataset_root
    split_root = current_path or dataset_root
    for split_name in ("train", "val", "test"):
        raw_split = str(config.get(split_name) or "").strip()
        if not raw_split:
            if split_name in {"train", "val"}:
                needs_rewrite = True
            continue
        split_path = Path(raw_split)
        if not split_path.is_absolute():
            split_path = split_root / split_path
        if not _is_inside_dataset(split_path):
            needs_rewrite = True
            break

    if not needs_rewrite:
        return True, "data.yaml wskazuje właściwy katalog datasetu.", False

    def _default_split_path(split_name: str) -> str:
        canonical = dataset_root / "images" / split_name
        nested = dataset_root / split_name / "images"
        if canonical.exists() or not nested.exists():
            return f"images/{split_name}"
        return f"{split_name}/images"

    config["path"] = str(dataset_root)
    config["train"] = _default_split_path("train")
    config["val"] = _default_split_path("val")
    if (dataset_dir / "images" / "test").exists() or "test" in config:
        config["test"] = _default_split_path("test")
    if "nc" not in config:
        names = _coerce_names(config)
        config["nc"] = max(1, len(names) or 1)

    try:
        _dump_dataset_config(dataset_dir, config)
    except Exception as exc:
        return False, f"Nie udało się zapisać poprawionego data.yaml: {exc}", False
    return True, "Poprawiono data.yaml tak, aby wskazywał bieżący wariant datasetu.", True


def update_yolo_dataset_class_names(dataset_dir: Path, names_by_index: dict[int, str]) -> tuple[bool, str]:
    """Update data.yaml class labels without touching images or labels."""
    dataset_dir = Path(dataset_dir)
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return False, "Brak data.yaml, nie można zmienić nazw klas."

    updates = {
        int(idx): str(name or "").strip()
        for idx, name in dict(names_by_index or {}).items()
        if str(name or "").strip()
    }
    if not updates:
        return True, "Nazwy klas bez zmian."

    config = _load_dataset_config(dataset_dir)
    names = _coerce_names(config)
    if not names:
        nc = int(config.get("nc", max(updates.keys()) + 1) or 1)
        names = {idx: f"class_{idx}" for idx in range(nc)}

    for idx, name in updates.items():
        names[idx] = name

    config["names"] = {idx: names[idx] for idx in sorted(names)}
    config["nc"] = max(int(config.get("nc", 0) or 0), len(config["names"]))
    _dump_dataset_config(dataset_dir, config)
    return True, "Nazwy klas w data.yaml zostały zaktualizowane."


def _discover_train_items(dataset_dir: Path) -> list[tuple[Path, Path]]:
    image_dir = Path(dataset_dir) / "images" / "train"
    label_dir = Path(dataset_dir) / "labels" / "train"
    if not image_dir.exists() or not label_dir.exists():
        return []

    items: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in _image_extensions():
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and label_path.read_text(encoding="utf-8", errors="ignore").strip():
            items.append((image_path, label_path))
    return items


_BALANCE_EXPLICIT_AUG_SUFFIX_RE = re.compile(
    r"(?i)(?:__aug[_-]?\d+|[_-]aug(?:mented)?[_-]?\d*)$"
)


def _load_augmentation_source_map(dataset_dir: Path) -> dict[str, str]:
    manifest_path = Path(dataset_dir) / "augmentation_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    generated_files = payload.get("generated_files")
    if not isinstance(generated_files, list):
        return {}
    mapping: dict[str, str] = {}
    for item in generated_files:
        if not isinstance(item, dict):
            continue
        source_key = str(item.get("source_key") or "").strip()
        source = item.get("source_label") or item.get("source_image")
        generated = item.get("label") or item.get("image")
        if not generated:
            continue
        generated_stem = Path(str(generated)).stem
        source_stem = source_key or Path(str(source or "")).stem
        if generated_stem and source_stem:
            mapping[generated_stem] = source_stem
    return mapping


def _normalize_balance_source_key(stem: str, source_map: dict[str, str] | None = None) -> str:
    source_map = source_map or {}
    current = str(stem or "").strip()
    seen: set[str] = set()
    while current in source_map and current not in seen:
        seen.add(current)
        current = str(source_map.get(current) or current).strip()
    while True:
        stripped = _BALANCE_EXPLICIT_AUG_SUFFIX_RE.sub("", current)
        if stripped == current:
            break
        current = stripped
    return current or str(stem or "")


def _is_original_balance_source_label(label_path: Path, source_key: str, source_map: dict[str, str]) -> bool:
    stem = Path(label_path).stem
    if stem in source_map:
        return False
    return stem == str(source_key or stem)


def _balance_candidate_attr(candidate: Any, key: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _iter_balance_plan_candidates(balance_plan: Any) -> list[Any]:
    if not balance_plan:
        return []
    if isinstance(balance_plan, Mapping):
        candidates = balance_plan.get("candidates") or []
    else:
        candidates = getattr(balance_plan, "candidates", []) or []
    return list(candidates) if isinstance(candidates, (list, tuple)) else []


def _has_balance_plan(balance_plan: Any) -> bool:
    if balance_plan is None:
        return False
    if isinstance(balance_plan, Mapping):
        return bool(balance_plan)
    return bool(getattr(balance_plan, "schema", None) or hasattr(balance_plan, "candidates"))


def _balance_plan_attr(balance_plan: Any, key: str, default: Any = None) -> Any:
    if isinstance(balance_plan, Mapping):
        return balance_plan.get(key, default)
    return getattr(balance_plan, key, default)


def _path_rel_key(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/").lower()
    except Exception:
        try:
            return str(Path(path).relative_to(root)).replace("\\", "/").lower()
        except Exception:
            return str(path).replace("\\", "/").lower()


def _path_rel_key_if_under(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/").lower()
    except Exception:
        return ""


def _candidate_label_rel_key(candidate: Any, dataset_dir: Path, base_dataset: Path | None) -> str:
    raw_label = str(_balance_candidate_attr(candidate, "label_path", "") or "").strip()
    if not raw_label:
        return ""
    label_path = Path(raw_label)
    if label_path.is_absolute():
        if base_dataset is not None:
            key = _path_rel_key_if_under(label_path, base_dataset)
            if key:
                return key
        key = _path_rel_key_if_under(label_path, dataset_dir)
        return key or str(label_path).replace("\\", "/").lower()
    return str(label_path).replace("\\", "/").lower()


def _build_balance_augmented_source_state(
    dataset_dir: Path,
    items: list[tuple[Path, Path]],
    *,
    balance_plan: Any = None,
    max_augmented_variants_per_source: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int | None], dict[str, int], dict[str, Any]]:
    source_map = _load_augmentation_source_map(dataset_dir)
    raw_candidates = _iter_balance_plan_candidates(balance_plan)
    plan_enabled = _has_balance_plan(balance_plan)
    base_dataset_text = str(_balance_plan_attr(balance_plan, "base_dataset", "") or "").strip()
    base_dataset = Path(base_dataset_text) if base_dataset_text else None
    plan_max = _balance_plan_attr(balance_plan, "max_augmented_variants_per_source", None)
    try:
        default_limit = int(max_augmented_variants_per_source if max_augmented_variants_per_source is not None else plan_max)
    except Exception:
        default_limit = 0 if plan_enabled else -1
    if not plan_enabled and max_augmented_variants_per_source is None:
        default_limit = -1

    plan_by_label: dict[str, dict[str, Any]] = {}
    allowed_sources: set[str] = set()
    source_limits: dict[str, int | None] = {}
    source_priorities: dict[str, float] = {}
    source_planned_variants: dict[str, int] = {}
    for candidate in raw_candidates:
        source_key = str(_balance_candidate_attr(candidate, "source_key", "") or "").strip()
        if not source_key:
            label_stem = Path(str(_balance_candidate_attr(candidate, "label_path", "") or "")).stem
            source_key = _normalize_balance_source_key(label_stem, source_map)
        if not source_key:
            continue
        allowed_sources.add(source_key)
        try:
            priority = float(_balance_candidate_attr(candidate, "priority", 1.0) or 1.0)
        except Exception:
            priority = 1.0
        try:
            limit = int(_balance_candidate_attr(candidate, "max_augmented_variants", default_limit) or default_limit)
        except Exception:
            limit = default_limit
        try:
            planned_variants_raw = _balance_candidate_attr(candidate, "planned_variants", None)
            planned_variants = None if planned_variants_raw in (None, "") else max(0, int(planned_variants_raw or 0))
        except Exception:
            planned_variants = None
        source_limits[source_key] = None if limit < 0 else max(0, limit)
        source_priorities[source_key] = max(float(source_priorities.get(source_key, 0.0)), float(priority))
        if planned_variants is not None:
            source_planned_variants[source_key] = max(
                int(source_planned_variants.get(source_key, 0) or 0),
                int(planned_variants),
            )
        label_key = _candidate_label_rel_key(candidate, dataset_dir, base_dataset)
        if label_key:
            plan_by_label[label_key] = {
                "source_key": source_key,
                "priority": max(0.0, float(priority)),
                "limit": source_limits[source_key],
                "planned_variants": planned_variants,
            }

    records: list[dict[str, Any]] = []
    initial_generated_by_source: dict[str, int] = {}
    for image_path, label_path in items:
        label_key = _path_rel_key(label_path, dataset_dir)
        plan_meta = plan_by_label.get(label_key)
        source_key = str((plan_meta or {}).get("source_key") or "").strip()
        if not source_key:
            source_key = _normalize_balance_source_key(label_path.stem, source_map)
        source_key = source_key or label_path.stem
        is_original = _is_original_balance_source_label(label_path, source_key, source_map)
        if not is_original:
            initial_generated_by_source[source_key] = int(initial_generated_by_source.get(source_key, 0) or 0) + 1
        priority = float((plan_meta or {}).get("priority") or source_priorities.get(source_key, 1.0) or 1.0)
        limit = (plan_meta or {}).get("limit", source_limits.get(source_key))
        planned_variants = (plan_meta or {}).get("planned_variants", source_planned_variants.get(source_key))
        if source_key not in source_limits:
            source_limits[source_key] = None if default_limit < 0 else max(0, default_limit)
        records.append(
            {
                "image_path": image_path,
                "label_path": label_path,
                "label_key": label_key,
                "source_key": source_key,
                "priority": max(0.0, float(priority)),
                "limit": limit if limit is not None else source_limits.get(source_key),
                "planned_variants": planned_variants,
                "is_original": bool(is_original),
            }
        )

    if plan_enabled:
        exact_records = [record for record in records if str(record.get("label_key") or "") in plan_by_label]
        if exact_records:
            records = exact_records
        else:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for record in records:
                source_key = str(record.get("source_key") or "")
                if source_key in allowed_sources:
                    grouped.setdefault(source_key, []).append(record)
            selected: list[dict[str, Any]] = []
            for group in grouped.values():
                group.sort(key=lambda item: (0 if item.get("is_original") else 1, str(item.get("label_key") or "")))
                selected.append(group[0])
            records = selected

    stats = {
        "balance_plan_enabled": bool(plan_enabled),
        "balance_plan_candidates": len(raw_candidates),
        "balance_pool": len(records),
        "max_augmented_variants_per_source": None if default_limit < 0 else max(0, default_limit),
        "planned_images": sum(int(value or 0) for value in source_planned_variants.values()) if plan_enabled else None,
        "planned_variants_by_source": dict(sorted(source_planned_variants.items())),
        "initial_generated_by_source": dict(sorted(initial_generated_by_source.items())),
    }
    return records, source_limits, initial_generated_by_source, stats


def _select_augmentation_sample_pool(
    rng: random.Random,
    pool: list[dict[str, Any]],
    sample_size: int,
    *,
    balance_plan_enabled: bool = False,
) -> list[dict[str, Any]]:
    limit = min(len(pool), max(0, int(sample_size or 0)))
    if limit <= 0:
        return []
    if balance_plan_enabled:
        return sorted(
            pool,
            key=lambda record: (
                -float(record.get("priority") or 0.0),
                str(record.get("source_key") or ""),
                str(record.get("label_key") or ""),
            ),
        )[:limit]
    rng.shuffle(pool)
    return pool[:limit]


def _select_balance_record(
    rng: random.Random,
    pool: list[dict[str, Any]],
    *,
    generated_by_source: dict[str, int],
    initial_generated_by_source: dict[str, int],
    source_limits: dict[str, int | None],
) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    weights: list[float] = []
    for record in pool:
        source_key = str(record.get("source_key") or "")
        planned_variants = record.get("planned_variants")
        if planned_variants not in (None, ""):
            try:
                if int(generated_by_source.get(source_key, 0) or 0) >= max(0, int(planned_variants or 0)):
                    continue
            except Exception:
                pass
        limit = source_limits.get(source_key, record.get("limit"))
        total_generated = int(initial_generated_by_source.get(source_key, 0) or 0) + int(generated_by_source.get(source_key, 0) or 0)
        if limit is not None and total_generated >= int(limit):
            continue
        eligible.append(record)
        weights.append(max(0.0001, float(record.get("priority") or 1.0)))
    if not eligible:
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        return rng.choice(eligible)
    marker = rng.random() * total_weight
    upto = 0.0
    for record, weight in zip(eligible, weights):
        upto += weight
        if marker <= upto:
            return record
    return eligible[-1]


def _finalize_train_augmentation_result(stats: Mapping[str, Any], requested_extra: int) -> tuple[bool, str]:
    requested = max(0, int(requested_extra or 0))
    generated = max(0, int((stats or {}).get("generated", 0) or 0))
    skipped = max(0, int((stats or {}).get("skipped", 0) or 0))
    stop_reason = str((stats or {}).get("stop_reason") or "").strip()
    suffix = f" Powód zatrzymania: {stop_reason}." if stop_reason else ""
    if generated <= 0:
        return (
            False,
            f"Zwiększanie syntetyczne train nie utworzyło żadnego obrazu z planowanych {requested}.{suffix}",
        )
    if requested > 0 and generated < requested:
        return (
            False,
            (
                "Zwiększanie syntetyczne train nie zostało ukończone: "
                f"dodano {generated} z {requested} obrazów, pominięto {skipped}. "
                "Wariant nie zostanie oznaczony jako gotowy, bo żądana liczba syntetyków nie została osiągnięta."
                f"{suffix}"
            ),
        )
    return (
        True,
        f"Zwiększanie syntetyczne train zakończone: dodano {generated} obrazów, pominięto {skipped}.",
    )


def _parse_yolo_label(label_path: Path, *, kpt_count: int, kpt_dim: int) -> list[YoloObject]:
    objects: list[YoloObject] = []
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return objects

    for raw_line in lines:
        tokens = str(raw_line or "").strip().split()
        if len(tokens) < 5:
            continue
        try:
            class_id = int(float(tokens[0]))
            bbox = [float(value) for value in tokens[1:5]]
            rest = [float(value) for value in tokens[5:]]
        except Exception:
            continue

        keypoints: list[tuple[float, float, float | None]] = []
        if kpt_count > 0 and kpt_dim >= 2:
            expected = kpt_count * kpt_dim
            if len(rest) < expected:
                continue
            for idx in range(kpt_count):
                offset = idx * kpt_dim
                visibility = rest[offset + 2] if kpt_dim >= 3 else None
                keypoints.append((rest[offset], rest[offset + 1], visibility))

        objects.append(
            YoloObject(
                class_id=class_id,
                bbox=[_clamp01(value) for value in bbox],
                keypoints=keypoints,
            )
        )
    return objects


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _format_float(value: float) -> str:
    return f"{_clamp01(value):.6f}"


def _rng_int(rng, low: int, high: int) -> int:
    try:
        return int(rng.randint(low, high))
    except Exception:
        return random.randint(low, high)


def _rng_float(rng, low: float, high: float) -> float:
    try:
        return float(rng.uniform(low, high))
    except Exception:
        return random.uniform(low, high)


def _apply_coarse_noise(image, profile: AugmentationProfile, rng) -> object:
    if np is None or profile.noise_strength <= 0:
        return image

    try:
        height, width = image.shape[:2]
        grain = max(1, int(profile.noise_grain_size or 1))
        noise_h = max(1, int((height + grain - 1) / grain))
        noise_w = max(1, int((width + grain - 1) / grain))
        std = max(1.0, float(profile.noise_strength) * 255.0)
        try:
            seed = _rng_int(rng, 0, 2**31 - 1)
            noise_rng = np.random.default_rng(seed)
            noise = noise_rng.normal(0.0, std, size=(noise_h, noise_w, 1)).astype("float32")
        except Exception:
            noise = np.random.normal(0.0, std, size=(noise_h, noise_w, 1)).astype("float32")
        if grain > 1 and cv2 is not None:
            noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_NEAREST)
            if len(noise.shape) == 2:
                noise = noise[:, :, None]
        else:
            noise = noise[:height, :width]
        out = image.astype("float32") + noise
        return np.clip(out, 0, 255).astype("uint8")
    except Exception:
        return image


def _build_headlight_night_relief_mask(profile: AugmentationProfile, height: int, width: int):
    if np is None or height <= 1 or width <= 1:
        return None

    try:
        scene = _combined_scene_headlight_fields(profile, height, width)
        if not scene:
            return None
        field = np.asarray(scene.get("field"), dtype="float32")
        hotspot = np.asarray(scene.get("hotspot"), dtype="float32")
        incidence = np.asarray(scene.get("incidence"), dtype="float32")
        mask = np.clip(field * (0.74 + 0.26 * incidence) + hotspot * 0.24, 0.0, 1.0)
        return np.clip(mask, 0.0, 1.0)
    except Exception:
        return None


def _apply_night_effect(image, profile: AugmentationProfile, *, base_surface: dict | None = None) -> object:
    if np is None:
        return image

    try:
        strength = max(0.0, min(1.0, float(profile.night_strength or 0.0)))
        light_strength = max(0.0, min(1.0, float(getattr(profile, "night_light_strength", 0.0) or 0.0)))
        bloom_strength = max(0.0, min(1.0, float(getattr(profile, "night_bloom_strength", 0.0) or 0.0)))
        iso_strength = max(0.0, min(1.0, float(getattr(profile, "night_iso_noise_strength", 0.0) or 0.0)))
        if strength <= 0.001 and light_strength <= 0.001 and bloom_strength <= 0.001 and iso_strength <= 0.001:
            return image

        night = math.pow(strength, 0.82)
        out = image.astype("float32") / 255.0
        height, width = image.shape[:2]
        luma = (
            out[:, :, 0] * 0.114
            + out[:, :, 1] * 0.587
            + out[:, :, 2] * 0.299
        )
        channel_max = np.max(out, axis=2)
        channel_min = np.min(out, axis=2)
        chroma_mask = np.clip((channel_max - channel_min) / np.maximum(channel_max, 0.08), 0.0, 1.0)
        chroma_mask = chroma_mask * chroma_mask * (3.0 - 2.0 * chroma_mask)
        # Fixed camera-like response is easier to understand than exposing a
        # user-facing luminance range. Bright surfaces still react first, but
        # the user controls the effect through night, light, bloom and ISO.
        bright_mask = np.clip((luma - 0.42) / 0.46, 0.0, 1.0)
        bright_mask = bright_mask * bright_mask * (3.0 - 2.0 * bright_mask)

        warmth = max(0.0, min(1.0, float(getattr(profile, "night_light_warmth", 0.35) if getattr(profile, "night_light_warmth", None) is not None else 0.35)))
        # Base light is the scene ambient: global, directionless and separate
        # from R1/R2/R3, which are rendered later as local light sources.
        iso_strength = max(iso_strength, night * 0.24)

        # Night camera response: dim the scene, crush midtones, but leave room
        # for retro-reflective highlights to bloom back into the image.
        ambient = np.array([0.020, 0.008, -0.050], dtype="float32")
        headlight_relief = _build_headlight_night_relief_mask(profile, height, width)
        if headlight_relief is not None:
            symbol_height = _surface_float_mask(base_surface, "symbol_height", width, height)
            if symbol_height is None:
                symbol_height = _build_symbol_contour_height_map(image, _contour_detection_sensitivity(profile))
            if symbol_height is None:
                symbol_height = np.zeros_like(luma, dtype="float32")
            else:
                symbol_height = np.clip(symbol_height.astype("float32"), 0.0, 1.0)
            if cv2 is not None:
                symbol_height = cv2.GaussianBlur(symbol_height, (0, 0), sigmaX=0.42, sigmaY=0.42)
            # Reflectors recover exposure on the reflective plate background,
            # but dark ink should stay visibly darker unless the light is so
            # strong that glare starts to wash the plate out.
            headlight_relief = np.clip(headlight_relief * (1.0 - 0.46 * symbol_height), 0.0, 1.0)
            effective_night = np.clip(night * (1.0 - 0.86 * headlight_relief), 0.0, 1.0).astype("float32")
        else:
            effective_night = np.full_like(luma, night, dtype="float32")
        effective_night_3d = effective_night[:, :, None]
        color_darken = chroma_mask * (1.0 - bright_mask * 0.35)
        darken = np.clip(0.34 * effective_night + 0.52 * effective_night * bright_mask + 0.30 * effective_night * color_darken, 0.0, 0.96)
        out *= 1.0 - darken[:, :, None]
        gamma = 1.0 + 1.10 * effective_night
        out = np.power(np.clip(out, 0.0, 1.0), gamma[:, :, None])
        luma_after_darken = np.clip(
            out[:, :, 0] * 0.114
            + out[:, :, 1] * 0.587
            + out[:, :, 2] * 0.299,
            0.0,
            1.0,
        )
        cool_gray = np.empty_like(out)
        cool_gray[:, :, 0] = luma_after_darken * 1.08 + 0.010 * effective_night
        cool_gray[:, :, 1] = luma_after_darken * 0.94 + 0.004 * effective_night
        cool_gray[:, :, 2] = luma_after_darken * 0.66
        desaturate = np.clip(0.34 * effective_night + 0.44 * effective_night * chroma_mask + 0.12 * effective_night * bright_mask, 0.0, 0.88)
        out = out * (1.0 - desaturate[:, :, None]) + cool_gray * desaturate[:, :, None]
        out += ambient * (0.50 * effective_night_3d + 0.34 * effective_night_3d * (1.0 - bright_mask[:, :, None]))

        cold = np.array([1.12, 1.06, 0.96], dtype="float32")
        warm = np.array([0.78, 1.02, 1.28], dtype="float32")
        bloom_source = np.clip(bright_mask * (0.36 + 0.36 * effective_night), 0.0, 1.0)
        bloom_light_color = cold * (1.0 - warmth) + warm * warmth
        if light_strength > 0.001 and height > 1 and width > 1:
            ambient_color = cold * (1.0 - warmth) + warm * warmth
            ambient_response = max(strength, 0.36)
            exposure_gain = light_strength * (0.055 + 0.245 * ambient_response)
            tint_gain = light_strength * (0.10 + 0.22 * ambient_response)
            floor_gain = light_strength * (0.010 + 0.040 * ambient_response)
            out = np.clip(out * (1.0 + exposure_gain), 0.0, 1.0)
            tinted_out = np.clip(out * ambient_color, 0.0, 1.0)
            out = out * (1.0 - tint_gain) + tinted_out * tint_gain
            ambient_floor = np.clip(ambient_color, 0.0, 1.0)
            out = out + (ambient_floor - out) * floor_gain
            bloom_light_color = ambient_color

        if cv2 is not None and bloom_strength > 0.001 and height > 1 and width > 1:
            grad_x = cv2.Sobel(luma.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(luma.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)
            edge_source = np.clip(np.abs(grad_x) + np.abs(grad_y), 0.0, 1.0)
            edge_source = cv2.GaussianBlur(edge_source, (0, 0), sigmaX=0.65, sigmaY=0.65)
            source = np.clip(
                bloom_source + edge_source * (0.10 + 0.22 * bloom_strength + 0.16 * max(strength, light_strength)),
                0.0,
                1.0,
            )
            sigma_wide = max(1.2, min(width, height) * (0.010 + 0.036 * bloom_strength))
            sigma_core = max(0.6, min(width, height) * (0.003 + 0.010 * bloom_strength))
            bloom_wide = cv2.GaussianBlur(source.astype("float32"), (0, 0), sigmaX=sigma_wide, sigmaY=sigma_wide)
            bloom_core = cv2.GaussianBlur(source.astype("float32"), (0, 0), sigmaX=sigma_core, sigmaY=sigma_core)
            bloom = np.clip(bloom_wide * 0.72 + bloom_core * 0.28, 0.0, 1.0)
            bloom_percentile = float(np.percentile(bloom, 99.2) or 0.0)
            if bloom_percentile > 0:
                bloom = np.clip(bloom / bloom_percentile, 0.0, 1.0)
            glow_gain = bloom_strength * (0.11 + 0.18 * strength + 0.16 * light_strength + 0.10 * max(0.0, 1.0 - strength))
            out += bloom[:, :, None] * bloom_light_color * glow_gain

        if iso_strength > 0.001:
            rng = np.random.default_rng(int(getattr(profile, "seed", 42) or 42) + 1931)
            luma_after = np.clip(out[:, :, 0] * 0.114 + out[:, :, 1] * 0.587 + out[:, :, 2] * 0.299, 0.0, 1.0)
            noise_scale = (0.010 + 0.065 * iso_strength) * (0.45 + 0.90 * (1.0 - luma_after))
            mono = rng.normal(0.0, noise_scale, size=luma_after.shape).astype("float32")
            chroma = rng.normal(0.0, noise_scale[:, :, None] * 0.45, size=out.shape).astype("float32")
            out += mono[:, :, None] + chroma

        return np.clip(out * 255.0, 0, 255).astype("uint8")
    except Exception:
        return image


def _apply_overexposure_effect(image, profile: AugmentationProfile, rng) -> object:
    if np is None or profile.overexposure_strength <= 0:
        return image

    try:
        strength = max(0.0, min(1.0, float(profile.overexposure_strength)))
        height, width = image.shape[:2]
        out = image.astype("float32")
        out += 42.0 * strength

        if cv2 is not None and height > 8 and width > 8:
            mask = np.zeros((height, width), dtype="float32")
            center_x = _rng_int(rng, int(width * 0.18), max(int(width * 0.82), int(width * 0.18) + 1))
            center_y = _rng_int(rng, int(height * 0.12), max(int(height * 0.55), int(height * 0.12) + 1))
            axis_x = max(8, int(width * (0.12 + 0.26 * strength)))
            axis_y = max(8, int(height * (0.08 + 0.18 * strength)))
            cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1.0, -1)
            kernel = max(3, int(min(width, height) * (0.08 + 0.08 * strength)))
            if kernel % 2 == 0:
                kernel += 1
            mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
            if float(mask.max() or 0.0) > 0:
                mask = mask / float(mask.max())
            out += mask[:, :, None] * (115.0 * strength)

        return np.clip(out, 0, 255).astype("uint8")
    except Exception:
        return image


def _rain_drop_direction(profile: AugmentationProfile, average_drop_size: float) -> tuple[float, float]:
    wind_strength = max(0.0, min(1.0, float(profile.dirt_flow_wind_strength if profile.dirt_flow_wind_strength is not None else 0.0)))
    wind_angle = math.radians(float(profile.dirt_flow_air_angle or 0.0))
    drop_wind_sensitivity = max(0.20, 1.0 - 0.72 * average_drop_size)
    intensity_wind_sensitivity = max(0.65, 1.0 - 0.22 * max(0.0, min(1.0, float(profile.rain_strength or 0.0))))
    wind_x = math.cos(wind_angle) * wind_strength * (0.35 + 1.95 * drop_wind_sensitivity * intensity_wind_sensitivity)
    wind_y = -math.sin(wind_angle) * wind_strength * 0.42
    fall_y = max(0.35, 1.0 + wind_y)
    direction_norm = max(0.001, math.sqrt(wind_x * wind_x + fall_y * fall_y))
    return wind_x / direction_norm, fall_y / direction_norm


def _build_rain_edge_response(image, step_x: float, step_y: float):
    if np is None or cv2 is None:
        return None
    try:
        height, width = image.shape[:2]
        if width <= 3 or height <= 3:
            return None
        if width * height > MAX_RAIN_EDGE_PIXELS:
            scale = math.sqrt(MAX_RAIN_EDGE_PIXELS / float(width * height))
            small_w = max(4, int(round(width * scale)))
            small_h = max(4, int(round(height * scale)))
            small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
            small_response = _build_rain_edge_response(small, step_x, step_y)
            if not small_response:
                return None
            gate = cv2.resize(small_response["gate"], (width, height), interpolation=cv2.INTER_LINEAR)
            reflect_x = cv2.resize(small_response["reflect_x"], (width, height), interpolation=cv2.INTER_LINEAR)
            reflect_y = cv2.resize(small_response["reflect_y"], (width, height), interpolation=cv2.INTER_LINEAR)
            reflect_norm = np.sqrt(reflect_x * reflect_x + reflect_y * reflect_y) + 1e-6
            return {
                "gate": gate,
                "reflect_x": reflect_x / reflect_norm,
                "reflect_y": reflect_y / reflect_norm,
            }
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        normal_x = grad_x / (grad_mag + 1e-6)
        normal_y = grad_y / (grad_mag + 1e-6)
        edge_strength = np.clip((grad_mag - 0.035) / 0.26, 0.0, 1.0)
        # Edge normals parallel to the drop velocity mean the edge line is a pseudo-surface
        # positioned across the falling rain, so it is a good candidate for impact mist.
        edge_alignment = np.abs((normal_x * step_x) + (normal_y * step_y))
        edge_gate = np.clip(edge_strength * (edge_alignment ** 2.35), 0.0, 1.0)
        dot = (step_x * normal_x) + (step_y * normal_y)
        reflect_x = step_x - 2.0 * dot * normal_x
        reflect_y = step_y - 2.0 * dot * normal_y
        reflect_norm = np.sqrt(reflect_x * reflect_x + reflect_y * reflect_y) + 1e-6
        return {
            "gate": cv2.GaussianBlur(edge_gate, (0, 0), sigmaX=0.55, sigmaY=0.55),
            "reflect_x": reflect_x / reflect_norm,
            "reflect_y": reflect_y / reflect_norm,
        }
    except Exception:
        return None


def _build_rain_edge_gate(image, step_x: float, step_y: float):
    response = _build_rain_edge_response(image, step_x, step_y)
    if not response:
        return None
    return response.get("gate")


def build_rain_edge_debug_mask(image, profile: AugmentationProfile):
    """Return a preview-only pseudo-surface mask; it is never written to datasets."""
    profile = (profile or AugmentationProfile()).normalized()
    if np is None or cv2 is None:
        return None
    legacy_drop_size = max(0.0, min(1.0, float(profile.rain_drop_size if profile.rain_drop_size is not None else 0.35)))
    drop_size_min = max(0.0, min(1.0, float(getattr(profile, "rain_drop_size_min", legacy_drop_size) if getattr(profile, "rain_drop_size_min", None) is not None else legacy_drop_size)))
    drop_size_max = max(0.0, min(1.0, float(getattr(profile, "rain_drop_size_max", legacy_drop_size) if getattr(profile, "rain_drop_size_max", None) is not None else legacy_drop_size)))
    if drop_size_min > drop_size_max:
        drop_size_min, drop_size_max = drop_size_max, drop_size_min
    step_x, step_y = _rain_drop_direction(profile, (drop_size_min + drop_size_max) / 2.0)
    edge_gate = _build_rain_edge_gate(image, step_x, step_y)
    symbol_mask = _build_symbol_contour_height_map(image, _contour_detection_sensitivity(profile))
    symbol_edges = None
    profile_curve = None
    profile_height = None
    profile_normals = None
    if symbol_mask is not None:
        try:
            grad_x = cv2.Sobel(symbol_mask.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(symbol_mask.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)
            symbol_edges = np.clip(np.abs(grad_x) + np.abs(grad_y), 0.0, 1.0)
            symbol_edges = cv2.GaussianBlur(symbol_edges, (0, 0), sigmaX=0.42, sigmaY=0.42)
        except Exception:
            symbol_edges = None
        try:
            contour_profile = _build_contour_profile_surface(symbol_mask, profile)
            if contour_profile:
                profile_curve = contour_profile.get("contour_curve")
                profile_height = contour_profile.get("height")
                profile_normals = contour_profile.get("normal_energy")
        except Exception:
            profile_curve = None
            profile_height = None
            profile_normals = None
    fallback_edges = None
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype("uint8", copy=False)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        median_gray = float(np.median(blurred) or 0.0)
        lower = int(max(8.0, min(120.0, median_gray * 0.52)))
        upper = int(max(float(lower + 24), min(235.0, median_gray * 1.34)))
        edges = cv2.Canny(blurred, lower, upper)
        if int(np.count_nonzero(edges)) < max(8, int(image.shape[0] * image.shape[1] * 0.0005)):
            edges = cv2.Canny(blurred, 16, 88)
        kernel = np.ones((3, 3), dtype="uint8")
        fallback_edges = cv2.dilate((edges > 0).astype("uint8"), kernel, iterations=1).astype("float32")
        fallback_edges = cv2.GaussianBlur(fallback_edges, (0, 0), sigmaX=0.35, sigmaY=0.35)
    except Exception:
        fallback_edges = None

    adaptive_edges = None
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype("uint8", copy=False)
        h, w = gray.shape[:2]
        if h >= 8 and w >= 8:
            try:
                clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
                local_gray = clahe.apply(gray)
            except Exception:
                local_gray = gray
            block = max(9, int(round(min(h, w) / 7.0)))
            if block % 2 == 0:
                block += 1
            block = min(block, 41 if 41 % 2 == 1 else 39)
            binary = cv2.adaptiveThreshold(
                local_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                max(3, block),
                3,
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            adaptive = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
            adaptive_edges = cv2.GaussianBlur((adaptive > 0).astype("float32"), (0, 0), sigmaX=0.42, sigmaY=0.42)
    except Exception:
        adaptive_edges = None

    masks = []
    for mask in (
        edge_gate,
        symbol_edges,
        profile_curve,
        profile_height,
        profile_normals,
        fallback_edges,
        adaptive_edges,
    ):
        if mask is None:
            continue
        try:
            local = np.clip(np.asarray(mask, dtype="float32"), 0.0, 1.0)
            peak = float(np.percentile(local, 99.0) or 0.0)
            scale = peak if peak > 0.0001 else float(local.max() or 0.0)
            if scale <= 0.0001:
                continue
            masks.append(np.clip(local / scale, 0.0, 1.0))
        except Exception:
            continue
    if not masks:
        return None
    combined = np.zeros_like(masks[0], dtype="float32")
    for mask in masks:
        if mask.shape[:2] != combined.shape[:2]:
            mask = cv2.resize(mask, (combined.shape[1], combined.shape[0]), interpolation=cv2.INTER_LINEAR)
        combined = np.maximum(combined, np.clip(mask.astype("float32"), 0.0, 1.0))
    combined = cv2.GaussianBlur(np.clip(combined, 0.0, 1.0), (0, 0), sigmaX=0.30, sigmaY=0.30)
    # Make the diagnostic layer intentionally legible. It is not exported to
    # datasets; it only lets us judge whether the contour/profile model sees
    # the same glyph parts that the user sees.
    peak = float(np.percentile(combined, 98.5) or 0.0)
    if peak > 0.0001:
        combined = np.clip(combined / peak, 0.0, 1.0)
    return combined.astype("float32", copy=False)


def build_contour_profile_wireframe_segments(image, profile: AugmentationProfile):
    """Return contour relief as normalized 3D wireframe line segments.

    Coordinates are normalized to the plate texture: x/y in 0..1 and z in
    0..1, where z is the contour relief height above the plate plane.
    """
    profile = (profile or AugmentationProfile()).normalized()
    if np is None or cv2 is None or image is None:
        return []
    try:
        base = np.asarray(image)
        if base.ndim == 2:
            base = np.repeat(base[:, :, None], 3, axis=2)
        elif base.ndim == 3 and base.shape[2] > 3:
            base = base[:, :, :3]
        if base.ndim != 3 or base.shape[2] != 3:
            return []
        height, width = base.shape[:2]
        if height < 4 or width < 4:
            return []
        pixel_count = int(height) * int(width)
        if pixel_count > MAX_CONTOUR_WIREFRAME_PIXELS:
            scale = math.sqrt(float(MAX_CONTOUR_WIREFRAME_PIXELS) / max(1.0, float(pixel_count)))
            new_width = max(32, int(round(float(width) * scale)))
            new_height = max(18, int(round(float(height) * scale)))
            base = cv2.resize(base, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = base.shape[:2]

        symbol_mask = _build_symbol_contour_height_map(base, _contour_detection_sensitivity(profile))
        contour_profile = _build_contour_profile_surface(symbol_mask, profile) if symbol_mask is not None else {}
        profile_height = contour_profile.get("height") if contour_profile else None
        contour_curve = contour_profile.get("wire_curve") if contour_profile else None
        if contour_curve is None and contour_profile:
            contour_curve = contour_profile.get("contour_curve")
        profile_radius = float(contour_profile.get("profile_radius", 3.0) if contour_profile else 3.0)
        stroke_half_width = float(contour_profile.get("stroke_half_width", profile_radius) if contour_profile else profile_radius)
        relief_curve = relief_profile_curve_for_preset(
            getattr(profile, "relief_profile_preset", DEFAULT_RELIEF_PROFILE_PRESET),
            getattr(profile, "relief_profile_curve", DEFAULT_RELIEF_PROFILE_CURVE),
        )

        def normalized_mask(mask):
            if mask is None:
                return None
            local = np.clip(np.asarray(mask, dtype="float32"), 0.0, 1.0)
            if local.shape[:2] != (height, width):
                local = cv2.resize(local, (width, height), interpolation=cv2.INTER_LINEAR)
            peak = float(np.percentile(local, 99.0) or 0.0)
            scale = peak if peak > 0.0001 else float(local.max() or 0.0)
            if scale <= 0.0001:
                return None
            return np.clip(local / scale, 0.0, 1.0)

        curve_mask = normalized_mask(contour_curve)
        if curve_mask is None:
            curve_mask = normalized_mask(build_rain_edge_debug_mask(base, profile))
        if curve_mask is None:
            return []
        height_mask = normalized_mask(profile_height)

        def add_segment(segments, start, end, kind: str) -> None:
            if start == end:
                return
            if len(segments) >= MAX_CONTOUR_WIREFRAME_SEGMENTS:
                return
            segments.append({"start": start, "end": end, "kind": kind})

        def normalized_surface_point(x: int, y: int, surface) -> tuple[float, float, float]:
            ix = max(0, min(width - 1, int(x)))
            iy = max(0, min(height - 1, int(y)))
            x01 = float(ix) / max(1.0, float(width - 1))
            y01 = float(iy) / max(1.0, float(height - 1))
            z01 = max(0.0, min(1.0, float(surface[iy, ix])))
            return x01, y01, z01

        def build_heightfield_wireframe(surface) -> list[dict]:
            if surface is None:
                return []
            surface = np.clip(np.asarray(surface, dtype="float32"), 0.0, 1.0)
            active = surface > 0.12
            if int(np.count_nonzero(active)) <= 4:
                return []
            ys, xs = np.where(active)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            min_dim = max(1.0, float(min(height, width)))
            step_x = max(4, min(14, int(round(min_dim * 0.055))))
            step_y = max(3, min(11, int(round(min_dim * 0.048))))
            x_values = sorted(set([x_min, x_max] + list(range(x_min, x_max + 1, step_x))))
            y_values = sorted(set([y_min, y_max] + list(range(y_min, y_max + 1, step_y))))
            segments: list[dict] = []

            for y in y_values:
                previous = None
                for x in x_values:
                    if not bool(active[y, x]):
                        previous = None
                        continue
                    current = normalized_surface_point(x, y, surface)
                    add_segment(segments, (current[0], current[1], 0.0), current, "strut")
                    if previous is not None:
                        add_segment(segments, previous, current, "rail")
                    previous = current

            for x in x_values:
                previous = None
                for y in y_values:
                    if not bool(active[y, x]):
                        previous = None
                        continue
                    current = normalized_surface_point(x, y, surface)
                    if previous is not None:
                        add_segment(segments, previous, current, "rib")
                    previous = current

            return segments

        surface_segments = build_heightfield_wireframe(height_mask)
        if surface_segments:
            return surface_segments

        def smooth_closed_points(points):
            if points is None or len(points) < 7:
                return points
            local_points = points.astype("float32", copy=True)
            passes = 2 if len(local_points) < 96 else 3
            for _pass in range(passes):
                local_points = (
                    local_points * 0.52
                    + np.roll(local_points, 1, axis=0) * 0.18
                    + np.roll(local_points, -1, axis=0) * 0.18
                    + np.roll(local_points, 2, axis=0) * 0.06
                    + np.roll(local_points, -2, axis=0) * 0.06
                )
            return local_points

        def sample_closed_polyline(points, sample_count: int) -> list[tuple[np.ndarray, np.ndarray]]:
            if points is None or len(points) < 4 or sample_count <= 0:
                return []
            closed = np.vstack([points, points[:1]])
            deltas = closed[1:] - closed[:-1]
            lengths = np.sqrt((deltas[:, 0] ** 2) + (deltas[:, 1] ** 2))
            total = float(np.sum(lengths) or 0.0)
            if total <= 0.001:
                return []
            cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
            samples: list[tuple[np.ndarray, np.ndarray]] = []
            for sample_index in range(sample_count):
                target = (float(sample_index) / float(sample_count)) * total
                segment_index = int(np.searchsorted(cumulative, target, side="right") - 1)
                segment_index = max(0, min(len(lengths) - 1, segment_index))
                segment_length = float(lengths[segment_index] or 1.0)
                ratio = max(0.0, min(1.0, (target - float(cumulative[segment_index])) / segment_length))
                point = closed[segment_index] * (1.0 - ratio) + closed[segment_index + 1] * ratio
                tangent = deltas[segment_index]
                tangent_len = float(math.hypot(float(tangent[0]), float(tangent[1])))
                if tangent_len <= 0.001:
                    continue
                samples.append((point.astype("float32"), (tangent / tangent_len).astype("float32")))
            return samples

        min_dim = max(1.0, float(min(height, width)))
        rib_half_cap = max(1.20, min(max(1.45, stroke_half_width * 0.74), max(1.45, min_dim * 0.026)))
        rib_half_len = max(1.05, min(8.0, profile_radius * 1.05 + min_dim * 0.0020, rib_half_cap))
        surface_eps = 0.045
        rail_params = np.array([-1.0, -0.38, 0.38, 1.0], dtype="float32")
        rib_params = np.array([-1.0, -0.50, 0.0, 0.50, 1.0], dtype="float32")

        def profile_z(local_t: float) -> float:
            return float(_sample_relief_profile_y(relief_curve, (max(-1.0, min(1.0, float(local_t))) + 1.0) * 0.5))

        def sampled_profile_height(point: np.ndarray, normal: np.ndarray, local_t: float) -> float:
            fallback = profile_z(local_t)
            if height_mask is None:
                return fallback
            sample = point + normal * (float(local_t) * rib_half_len)
            ix = int(round(float(sample[0])))
            iy = int(round(float(sample[1])))
            if ix < 0 or iy < 0 or ix >= width or iy >= height:
                return 0.0
            local_height = float(height_mask[iy, ix])
            if local_height <= surface_eps:
                return 0.0
            return max(0.0, min(1.0, local_height * 0.88 + fallback * 0.12))

        def normalized_point(point: np.ndarray, normal: np.ndarray, local_t: float, height_value: float) -> tuple[float, float, float]:
            sample = point + normal * (float(local_t) * rib_half_len)
            x01 = max(0.0, min(1.0, float(sample[0]) / max(1.0, float(width - 1))))
            y01 = max(0.0, min(1.0, float(sample[1]) / max(1.0, float(height - 1))))
            return x01, y01, max(0.0, min(1.0, float(height_value)))

        curve_u8 = (curve_mask > 0.16).astype("uint8") * 255
        contours, _hierarchy = cv2.findContours(curve_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        segments: list[dict] = []
        accepted_contours = 0
        for contour in contours:
            if contour is None or len(contour) < 8:
                continue
            perimeter = float(cv2.arcLength(contour, True) or 0.0)
            if perimeter < 9.0:
                continue
            points = smooth_closed_points(contour.reshape(-1, 2).astype("float32"))
            rail_count = max(16, min(48, int(round(perimeter / max(8.0, min_dim * 0.065)))))
            rib_count = max(5, min(14, int(round(perimeter / max(28.0, min_dim * 0.190)))))
            rail_samples = sample_closed_polyline(points, rail_count)
            rib_samples = sample_closed_polyline(points, rib_count)
            if len(rail_samples) < 3 or len(rib_samples) < 3:
                continue

            def normal_samples_from(samples_list):
                sample_points = [point for point, _tangent in samples_list]
                output: list[tuple[np.ndarray, np.ndarray]] = []
                for sample_index, point in enumerate(sample_points):
                    previous_point = sample_points[sample_index - 1]
                    next_point = sample_points[(sample_index + 1) % len(sample_points)]
                    tangent = next_point - previous_point
                    normal = np.array([-float(tangent[1]), float(tangent[0])], dtype="float32")
                    normal_len = float(math.hypot(float(normal[0]), float(normal[1])))
                    if normal_len <= 0.001:
                        continue
                    output.append((point, normal / normal_len))
                return output

            rail_normal_samples = normal_samples_from(rail_samples)
            rib_normal_samples = normal_samples_from(rib_samples)
            if len(rail_normal_samples) < 3 or len(rib_normal_samples) < 3:
                continue
            accepted_contours += 1

            for point, normal in rib_normal_samples:
                for t_value in (-0.50, 0.0, 0.50):
                    top_h = sampled_profile_height(point, normal, float(t_value))
                    if top_h <= surface_eps:
                        continue
                    base_point = normalized_point(point, normal, float(t_value), 0.0)
                    top_point = normalized_point(point, normal, float(t_value), top_h)
                    add_segment(segments, base_point, top_point, "strut")

            for t_value in rail_params:
                rail = []
                for point, normal in rail_normal_samples:
                    height_value = sampled_profile_height(point, normal, float(t_value))
                    rail.append(
                        normalized_point(point, normal, float(t_value), height_value)
                        if height_value > surface_eps
                        else None
                    )
                kind = "edge" if abs(float(t_value)) > 0.98 else "rail"
                for idx, start in enumerate(rail):
                    end = rail[(idx + 1) % len(rail)]
                    if start is not None and end is not None:
                        add_segment(segments, start, end, kind)

            for point, normal in rib_normal_samples:
                rib = []
                for t_value in rib_params:
                    height_value = sampled_profile_height(point, normal, float(t_value))
                    rib.append(
                        normalized_point(point, normal, float(t_value), height_value)
                        if height_value > surface_eps
                        else None
                    )
                for idx in range(len(rib) - 1):
                    if rib[idx] is not None and rib[idx + 1] is not None:
                        add_segment(segments, rib[idx], rib[idx + 1], "rib")
            if len(segments) >= MAX_CONTOUR_WIREFRAME_SEGMENTS:
                break

        if accepted_contours <= 0:
            return []
        return segments
    except Exception:
        return []


def build_contour_profile_debug_overlay(image, profile: AugmentationProfile):
    """Return a preview-only RGB wireframe of the contour/profile surface."""
    profile = (profile or AugmentationProfile()).normalized()
    if np is None or cv2 is None or image is None:
        return image
    try:
        base = np.asarray(image)
        if base.ndim == 2:
            base = np.repeat(base[:, :, None], 3, axis=2)
        elif base.ndim == 3 and base.shape[2] > 3:
            base = base[:, :, :3]
        if base.ndim != 3 or base.shape[2] != 3:
            return image
        height, width = base.shape[:2]
        if height < 4 or width < 4:
            return image

        symbol_mask = _build_symbol_contour_height_map(base, _contour_detection_sensitivity(profile))
        contour_profile = _build_contour_profile_surface(symbol_mask, profile) if symbol_mask is not None else {}
        profile_height = contour_profile.get("height") if contour_profile else None
        contour_curve = contour_profile.get("wire_curve") if contour_profile else None
        if contour_curve is None and contour_profile:
            contour_curve = contour_profile.get("contour_curve")
        profile_radius = float(contour_profile.get("profile_radius", 3.0) if contour_profile else 3.0)
        stroke_half_width = float(contour_profile.get("stroke_half_width", profile_radius) if contour_profile else profile_radius)
        relief_curve = relief_profile_curve_for_preset(
            getattr(profile, "relief_profile_preset", DEFAULT_RELIEF_PROFILE_PRESET),
            getattr(profile, "relief_profile_curve", DEFAULT_RELIEF_PROFILE_CURVE),
        )

        def normalized_mask(mask):
            if mask is None:
                return None
            local = np.clip(np.asarray(mask, dtype="float32"), 0.0, 1.0)
            if local.shape[:2] != (height, width):
                local = cv2.resize(local, (width, height), interpolation=cv2.INTER_LINEAR)
            peak = float(np.percentile(local, 99.0) or 0.0)
            scale = peak if peak > 0.0001 else float(local.max() or 0.0)
            if scale <= 0.0001:
                return None
            return np.clip(local / scale, 0.0, 1.0)

        curve_mask = normalized_mask(contour_curve)
        if curve_mask is None:
            curve_mask = normalized_mask(build_rain_edge_debug_mask(base, profile))
        if curve_mask is None:
            return image
        height_mask = normalized_mask(profile_height)

        # Old-school CAD/HUD view: no filled masks, no legend, only wireframe
        # lines on a darkened plate context.
        luma = (
            base[:, :, 0].astype("float32") * 0.114
            + base[:, :, 1].astype("float32") * 0.587
            + base[:, :, 2].astype("float32") * 0.299
        )
        context = np.repeat(luma[:, :, None], 3, axis=2) * 0.10
        mesh_f = context + np.array([4.0, 9.0, 13.0], dtype="float32")
        mesh = np.clip(mesh_f, 0, 255).astype("uint8")
        min_dim = max(1.0, float(min(height, width)))
        z_lift = max(5.0, min(18.0, profile_radius * 2.15 + min_dim * 0.0120))
        z_axis = np.array([-0.48 * z_lift, -1.0 * z_lift], dtype="float32")

        def draw_heightfield_mesh(surface) -> bool:
            if surface is None:
                return False
            surface = np.clip(np.asarray(surface, dtype="float32"), 0.0, 1.0)
            active = surface > 0.12
            if int(np.count_nonzero(active)) <= 4:
                return False
            ys, xs = np.where(active)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            step_x = max(4, min(14, int(round(min_dim * 0.055))))
            step_y = max(3, min(11, int(round(min_dim * 0.048))))
            x_values = sorted(set([x_min, x_max] + list(range(x_min, x_max + 1, step_x))))
            y_values = sorted(set([y_min, y_max] + list(range(y_min, y_max + 1, step_y))))

            def projected_point(x: int, y: int) -> tuple[int, int]:
                z = max(0.0, min(1.0, float(surface[y, x])))
                return int(round(float(x) + float(z_axis[0]) * z)), int(round(float(y) + float(z_axis[1]) * z))

            drawn = 0
            for y in y_values:
                previous = None
                for x in x_values:
                    if not bool(active[y, x]):
                        previous = None
                        continue
                    current = projected_point(x, y)
                    if previous is not None:
                        cv2.line(mesh, previous, current, (82, 226, 204), thickness=1, lineType=cv2.LINE_AA)
                        drawn += 1
                    previous = current
            for x in x_values:
                previous = None
                for y in y_values:
                    if not bool(active[y, x]):
                        previous = None
                        continue
                    current = projected_point(x, y)
                    if previous is not None:
                        cv2.line(mesh, previous, current, (42, 154, 158), thickness=1, lineType=cv2.LINE_AA)
                        drawn += 1
                    previous = current
            return drawn > 0

        if draw_heightfield_mesh(height_mask):
            return mesh

        def smooth_closed_points(points):
            if points is None or len(points) < 7:
                return points
            local_points = points.astype("float32", copy=True)
            passes = 2 if len(local_points) < 96 else 3
            for _pass in range(passes):
                local_points = (
                    local_points * 0.52
                    + np.roll(local_points, 1, axis=0) * 0.18
                    + np.roll(local_points, -1, axis=0) * 0.18
                    + np.roll(local_points, 2, axis=0) * 0.06
                    + np.roll(local_points, -2, axis=0) * 0.06
                )
            return local_points

        def sample_closed_polyline(points, sample_count: int) -> list[tuple[np.ndarray, np.ndarray]]:
            if points is None or len(points) < 4 or sample_count <= 0:
                return []
            closed = np.vstack([points, points[:1]])
            deltas = closed[1:] - closed[:-1]
            lengths = np.sqrt((deltas[:, 0] ** 2) + (deltas[:, 1] ** 2))
            total = float(np.sum(lengths) or 0.0)
            if total <= 0.001:
                return []
            cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
            samples: list[tuple[np.ndarray, np.ndarray]] = []
            for sample_index in range(sample_count):
                target = (float(sample_index) / float(sample_count)) * total
                segment_index = int(np.searchsorted(cumulative, target, side="right") - 1)
                segment_index = max(0, min(len(lengths) - 1, segment_index))
                segment_length = float(lengths[segment_index] or 1.0)
                ratio = max(0.0, min(1.0, (target - float(cumulative[segment_index])) / segment_length))
                point = closed[segment_index] * (1.0 - ratio) + closed[segment_index + 1] * ratio
                tangent = deltas[segment_index]
                tangent_len = float(math.hypot(float(tangent[0]), float(tangent[1])))
                if tangent_len <= 0.001:
                    continue
                samples.append((point.astype("float32"), (tangent / tangent_len).astype("float32")))
            return samples

        curve_u8 = (curve_mask > 0.16).astype("uint8") * 255
        contours, _hierarchy = cv2.findContours(curve_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        rib_half_cap = max(1.20, min(max(1.45, stroke_half_width * 0.74), max(1.45, min_dim * 0.026)))
        rib_half_len = max(1.05, min(8.0, profile_radius * 1.05 + min_dim * 0.0020, rib_half_cap))
        surface_eps = 0.045
        rail_params = np.array([-1.0, -0.38, 0.38, 1.0], dtype="float32")
        rib_params = np.array([-1.0, -0.50, 0.0, 0.50, 1.0], dtype="float32")

        def profile_z(local_t: float) -> float:
            return float(_sample_relief_profile_y(relief_curve, (max(-1.0, min(1.0, float(local_t))) + 1.0) * 0.5))

        def sampled_profile_height(point: np.ndarray, normal: np.ndarray, local_t: float) -> float:
            fallback = profile_z(local_t)
            if height_mask is None:
                return fallback
            sample = point + normal * (float(local_t) * rib_half_len)
            ix = int(round(float(sample[0])))
            iy = int(round(float(sample[1])))
            if ix < 0 or iy < 0 or ix >= width or iy >= height:
                return 0.0
            local_height = float(height_mask[iy, ix])
            if local_height <= surface_eps:
                return 0.0
            return max(0.0, min(1.0, local_height * 0.88 + fallback * 0.12))

        def plate_plane_point(point: np.ndarray, normal: np.ndarray, local_t: float) -> tuple[int, int]:
            projected = point + normal * (float(local_t) * rib_half_len)
            return int(round(float(projected[0]))), int(round(float(projected[1])))

        def mesh_point(point: np.ndarray, normal: np.ndarray, local_t: float) -> tuple[int, int] | None:
            z = sampled_profile_height(point, normal, local_t)
            if z <= surface_eps:
                return None
            projected = point + normal * (float(local_t) * rib_half_len) + z_axis * z
            return int(round(float(projected[0]))), int(round(float(projected[1])))

        def rail_color(t_value: float) -> tuple[int, int, int]:
            t_abs = abs(float(t_value))
            if t_abs < 0.02:
                return (255, 190, 80)
            if t_abs > 0.98:
                return (38, 118, 126)
            return (72, 218, 205)

        accepted_contours = 0
        for contour in contours:
            if contour is None or len(contour) < 8:
                continue
            perimeter = float(cv2.arcLength(contour, True) or 0.0)
            if perimeter < 9.0:
                continue
            points = contour.reshape(-1, 2).astype("float32")
            points = smooth_closed_points(points)
            rail_count = max(16, min(48, int(round(perimeter / max(8.0, min_dim * 0.065)))))
            rib_count = max(5, min(14, int(round(perimeter / max(28.0, min_dim * 0.190)))))
            rail_samples = sample_closed_polyline(points, rail_count)
            rib_samples = sample_closed_polyline(points, rib_count)
            if len(rail_samples) < 3 or len(rib_samples) < 3:
                continue

            def normal_samples_from(samples_list):
                sample_points = [point for point, _tangent in samples_list]
                output: list[tuple[np.ndarray, np.ndarray]] = []
                for sample_index, point in enumerate(sample_points):
                    previous_point = sample_points[sample_index - 1]
                    next_point = sample_points[(sample_index + 1) % len(sample_points)]
                    tangent = next_point - previous_point
                    normal = np.array([-float(tangent[1]), float(tangent[0])], dtype="float32")
                    normal_len = float(math.hypot(float(normal[0]), float(normal[1])))
                    if normal_len <= 0.001:
                        continue
                    output.append((point, normal / normal_len))
                return output

            rail_normal_samples = normal_samples_from(rail_samples)
            rib_normal_samples = normal_samples_from(rib_samples)
            if len(rail_normal_samples) < 3 or len(rib_normal_samples) < 3:
                continue
            accepted_contours += 1

            for point, normal in rib_normal_samples:
                for t_value in (0.0,):
                    base_pt = plate_plane_point(point, normal, float(t_value))
                    top_pt = mesh_point(point, normal, float(t_value))
                    if top_pt is not None:
                        cv2.line(mesh, base_pt, top_pt, (78, 176, 158), thickness=1, lineType=cv2.LINE_AA)

            # Draw the raised rails and ribs. The whole model is only polylines,
            # so it reads as a polygon mesh rather than a colored mask.
            for t_value in rail_params:
                rail_points = [mesh_point(point, normal, float(t_value)) for point, normal in rail_normal_samples]
                for idx, start in enumerate(rail_points):
                    end = rail_points[(idx + 1) % len(rail_points)]
                    if start is not None and end is not None:
                        cv2.line(mesh, start, end, rail_color(float(t_value)), thickness=1, lineType=cv2.LINE_AA)

            for sample_idx, (point, normal) in enumerate(rib_normal_samples):
                rib_points = [mesh_point(point, normal, float(t_value)) for t_value in rib_params]
                rib_color = (96, 236, 214) if sample_idx % 4 == 0 else (44, 150, 154)
                for idx in range(len(rib_points) - 1):
                    if rib_points[idx] is not None and rib_points[idx + 1] is not None:
                        cv2.line(mesh, rib_points[idx], rib_points[idx + 1], rib_color, thickness=1, lineType=cv2.LINE_AA)

        if accepted_contours <= 0:
            return image

        return mesh
    except Exception:
        return image


def _apply_rain_lens_distortion(image, rain_mask, lens_strength: float, profile: AugmentationProfile | None = None):
    if np is None or cv2 is None or rain_mask is None or lens_strength <= 0:
        return image
    try:
        height, width = image.shape[:2]
        if width <= 3 or height <= 3:
            return image
        if width * height > MAX_RAIN_LENS_PIXELS:
            scale = math.sqrt(MAX_RAIN_LENS_PIXELS / float(width * height))
            small_w = max(4, int(round(width * scale)))
            small_h = max(4, int(round(height * scale)))
            small_image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(rain_mask, (small_w, small_h), interpolation=cv2.INTER_AREA)
            small_result = _apply_rain_lens_distortion(small_image, small_mask, lens_strength, profile)
            return cv2.resize(small_result, (width, height), interpolation=cv2.INTER_LINEAR)
        lens = cv2.GaussianBlur(np.clip(rain_mask.astype("float32"), 0.0, 1.0), (0, 0), sigmaX=0.55, sigmaY=0.55)
        # Lens strength controls optical bending/glints, not rain coverage.
        # Keeping the droplet mask energy stable prevents the slider from
        # acting like a hidden exposure boost.
        lens = np.clip(lens, 0.0, 1.0)
        if float(lens.max() or 0.0) <= 0.001:
            return image
        rain_alpha = max(
            0.0,
            min(
                1.0,
                float(getattr(profile, "rain_alpha", 0.22) if profile is not None and getattr(profile, "rain_alpha", None) is not None else 0.22),
            ),
        )
        material_visibility = 0.48 + 0.52 * rain_alpha
        surface_fine = np.clip(lens ** 0.62, 0.0, 1.0)
        surface_mid = cv2.GaussianBlur(surface_fine, (0, 0), sigmaX=1.05 + 0.95 * lens_strength, sigmaY=1.05 + 0.95 * lens_strength)
        surface_broad = cv2.GaussianBlur(surface_fine, (0, 0), sigmaX=2.30 + 1.80 * lens_strength, sigmaY=2.30 + 1.80 * lens_strength)
        grad_x = (
            cv2.Sobel(surface_fine, cv2.CV_32F, 1, 0, ksize=3) * 0.58
            + cv2.Sobel(surface_mid, cv2.CV_32F, 1, 0, ksize=3) * 0.30
            + cv2.Sobel(surface_broad, cv2.CV_32F, 1, 0, ksize=3) * 0.12
        )
        grad_y = (
            cv2.Sobel(surface_fine, cv2.CV_32F, 0, 1, ksize=3) * 0.58
            + cv2.Sobel(surface_mid, cv2.CV_32F, 0, 1, ksize=3) * 0.30
            + cv2.Sobel(surface_broad, cv2.CV_32F, 0, 1, ksize=3) * 0.12
        )
        grad_energy = np.abs(grad_x) + np.abs(grad_y)
        grad_scale = float(np.percentile(grad_energy, 99.2) or 0.0)
        if grad_scale > 0.0001:
            normal_x = np.clip(grad_x / grad_scale, -1.0, 1.0)
            normal_y = np.clip(grad_y / grad_scale, -1.0, 1.0)
            grad_energy = np.clip(grad_energy / grad_scale, 0.0, 1.0)
        else:
            normal_x = grad_x
            normal_y = grad_y
            grad_energy = np.clip(grad_energy, 0.0, 1.0)
        normal_xy_len = np.clip(normal_x * normal_x + normal_y * normal_y, 0.0, 0.96)
        normal_z = np.sqrt(np.clip(1.0 - normal_xy_len, 0.04, 1.0))
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype="float32"), np.arange(height, dtype="float32"))
        thickness = np.clip(surface_fine * (0.38 + 0.62 * surface_mid), 0.0, 1.0)
        back_surface = cv2.GaussianBlur(thickness, (0, 0), sigmaX=1.65 + 1.35 * lens_strength, sigmaY=1.65 + 1.35 * lens_strength)
        back_grad_x = cv2.Sobel(back_surface, cv2.CV_32F, 1, 0, ksize=3)
        back_grad_y = cv2.Sobel(back_surface, cv2.CV_32F, 0, 1, ksize=3)
        back_grad_energy = np.abs(back_grad_x) + np.abs(back_grad_y)
        back_grad_scale = float(np.percentile(back_grad_energy, 99.2) or 0.0)
        if back_grad_scale > 0.0001:
            back_normal_x = np.clip(-back_grad_x / back_grad_scale * 0.58, -0.86, 0.86)
            back_normal_y = np.clip(-back_grad_y / back_grad_scale * 0.58, -0.86, 0.86)
        else:
            back_normal_x = np.zeros_like(normal_x, dtype="float32")
            back_normal_y = np.zeros_like(normal_y, dtype="float32")
        back_normal_xy_len = np.clip(back_normal_x * back_normal_x + back_normal_y * back_normal_y, 0.0, 0.92)
        back_normal_z = -np.sqrt(np.clip(1.0 - back_normal_xy_len, 0.08, 1.0))
        mask_gain = np.clip(0.16 + thickness * (0.84 + 0.36 * lens_strength), 0.0, 1.0)

        def _normalize_vector(vx, vy, vz):
            length = np.sqrt(vx * vx + vy * vy + vz * vz) + 1e-6
            return vx / length, vy / length, vz / length

        def _refract_or_reflect(ix, iy, iz, nx, ny, nz, eta: float):
            dot = ix * nx + iy * ny + iz * nz
            cos_i = np.clip(-dot, -1.0, 1.0)
            k = 1.0 - float(eta) * float(eta) * (1.0 - cos_i * cos_i)
            sqrt_k = np.sqrt(np.maximum(k, 0.0))
            refr_x = float(eta) * ix + (float(eta) * cos_i - sqrt_k) * nx
            refr_y = float(eta) * iy + (float(eta) * cos_i - sqrt_k) * ny
            refr_z = float(eta) * iz + (float(eta) * cos_i - sqrt_k) * nz
            refl_x = ix - 2.0 * dot * nx
            refl_y = iy - 2.0 * dot * ny
            refl_z = iz - 2.0 * dot * nz
            use_reflect = k < 0.0
            out_x = np.where(use_reflect, refl_x, refr_x)
            out_y = np.where(use_reflect, refl_y, refr_y)
            out_z = np.where(use_reflect, refl_z, refr_z)
            return _normalize_vector(out_x, out_y, out_z)

        water_ior = 1.333
        entry_nx = -normal_x
        entry_ny = -normal_y
        entry_nz = -normal_z
        inside_x, inside_y, inside_z = _refract_or_reflect(
            0.0,
            0.0,
            1.0,
            entry_nx,
            entry_ny,
            entry_nz,
            1.0 / water_ior,
        )
        displacement = 1.6 + 22.0 * lens_strength
        inner_distance = displacement * (0.34 + 1.45 * thickness) * mask_gain
        exit_probe_x = grid_x + (inside_x / np.maximum(np.abs(inside_z), 0.12)) * inner_distance
        exit_probe_y = grid_y + (inside_y / np.maximum(np.abs(inside_z), 0.12)) * inner_distance
        exit_nx = cv2.remap(back_normal_x, exit_probe_x.astype("float32"), exit_probe_y.astype("float32"), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        exit_ny = cv2.remap(back_normal_y, exit_probe_x.astype("float32"), exit_probe_y.astype("float32"), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        exit_nz = cv2.remap(back_normal_z, exit_probe_x.astype("float32"), exit_probe_y.astype("float32"), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        exit_x, exit_y, exit_z = _refract_or_reflect(
            inside_x,
            inside_y,
            inside_z,
            exit_nx,
            exit_ny,
            exit_nz,
            water_ior / 1.0,
        )
        exit_distance = displacement * (0.42 + 1.08 * thickness) * mask_gain
        ray_shift_x = (
            (inside_x / np.maximum(np.abs(inside_z), 0.12)) * inner_distance
            + (exit_x / np.maximum(np.abs(exit_z), 0.12)) * exit_distance
        )
        ray_shift_y = (
            (inside_y / np.maximum(np.abs(inside_z), 0.12)) * inner_distance
            + (exit_y / np.maximum(np.abs(exit_z), 0.12)) * exit_distance
        )
        chroma_gain = 0.018 + 0.026 * lens_strength
        refracted_primary = cv2.remap(
            image,
            (grid_x + ray_shift_x).astype("float32"),
            (grid_y + ray_shift_y).astype("float32"),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        refracted_warm = cv2.remap(
            image,
            (grid_x + ray_shift_x * (1.0 - chroma_gain)).astype("float32"),
            (grid_y + ray_shift_y * (1.0 - chroma_gain)).astype("float32"),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        refracted_cold = cv2.remap(
            image,
            (grid_x + ray_shift_x * (1.0 + chroma_gain)).astype("float32"),
            (grid_y + ray_shift_y * (1.0 + chroma_gain)).astype("float32"),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        distorted = refracted_primary.astype("float32")
        if image.ndim == 3 and image.shape[2] >= 3:
            distorted[:, :, 0] = refracted_cold[:, :, 0].astype("float32")
            distorted[:, :, 1] = refracted_primary[:, :, 1].astype("float32")
            distorted[:, :, 2] = refracted_warm[:, :, 2].astype("float32")
        alpha_occlusion = thickness * (0.026 + 0.255 * rain_alpha)
        lens_occlusion = thickness * lens_strength * (0.085 + 0.295 * rain_alpha)
        glass_alpha = np.clip((alpha_occlusion + lens_occlusion) * material_visibility, 0.0, 0.68)
        if image.ndim == 2:
            out = image.astype("float32") * (1.0 - glass_alpha) + distorted.astype("float32") * glass_alpha
        else:
            out = image.astype("float32") * (1.0 - glass_alpha[:, :, None]) + distorted * glass_alpha[:, :, None]

        light_context = _active_relief_light_context(profile, height, width) if profile is not None else None
        if light_context:
            light_x = float(light_context.get("x", 0.0) or 0.0)
            light_y = float(light_context.get("y", 0.0) or 0.0)
            light_strength = max(0.0, min(1.0, float(light_context.get("strength", 0.0) or 0.0)))
            light_source_z = max(0.035, min(4.0, float(light_context.get("source_z", 1.25) or 1.25)))
            source_color = np.array(light_context.get("color_bgr") or (255.0, 255.0, 255.0), dtype="float32")
            if source_color.shape != (3,):
                source_color = np.array([255.0, 255.0, 255.0], dtype="float32")
            source_color = np.clip(source_color, 0.0, 255.0)
            source_color_field = light_context.get("color_field")
            if source_color_field is not None:
                try:
                    source_color_field = np.clip(np.asarray(source_color_field, dtype="float32"), 0.0, 255.0)
                    if source_color_field.shape[:2] != (height, width):
                        source_color_field = cv2.resize(source_color_field, (width, height), interpolation=cv2.INTER_LINEAR)
                    if source_color_field.shape != (height, width, 3):
                        source_color_field = None
                except Exception:
                    source_color_field = None
            light_gate = light_context.get("light_gate")
            if light_gate is not None:
                try:
                    light_gate = np.clip(np.asarray(light_gate, dtype="float32"), 0.0, 1.0)
                    if light_gate.shape != (height, width):
                        light_gate = cv2.resize(light_gate, (width, height), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    light_gate = None
        else:
            light_x, light_y, light_strength = -0.32, -0.72, 0.0
            light_source_z = 1.45
            source_color = np.array([0.0, 0.0, 0.0], dtype="float32")
            source_color_field = None
            light_gate = None
        light_norm = math.hypot(light_x, light_y) or 1.0
        light_x /= light_norm
        light_y /= light_norm
        light_z = 0.55 + 0.35 * light_strength
        light_len = math.sqrt(light_x * light_x + light_y * light_y + light_z * light_z) or 1.0
        light_3d_x = light_x / light_len
        light_3d_y = light_y / light_len
        light_3d_z = light_z / light_len

        rim = cv2.dilate(lens, np.ones((3, 3), dtype="uint8"), iterations=1) - cv2.erode(lens, np.ones((3, 3), dtype="uint8"), iterations=1)
        rim = cv2.GaussianBlur(np.clip(rim, 0.0, 1.0), (0, 0), sigmaX=0.34, sigmaY=0.34)
        ndotl = np.clip(normal_x * light_3d_x + normal_y * light_3d_y + normal_z * light_3d_z, 0.0, 1.0)
        half_x = light_3d_x
        half_y = light_3d_y
        half_z = light_3d_z + 1.0
        half_len = np.sqrt(half_x * half_x + half_y * half_y + half_z * half_z) + 1e-6
        ndoth = np.clip((normal_x * half_x + normal_y * half_y + normal_z * half_z) / half_len, 0.0, 1.0)
        fresnel = np.clip(0.020 + 0.98 * ((1.0 - normal_z) ** 5.0), 0.0, 1.0)
        front_response = np.clip(ndotl * 0.74 + fresnel * 0.38, 0.0, 1.0)
        back_response = np.clip((1.0 - ndotl) * (0.32 + 0.68 * fresnel), 0.0, 1.0)
        env_source = cv2.GaussianBlur(image, (0, 0), sigmaX=1.7 + 2.6 * lens_strength, sigmaY=1.2 + 2.0 * lens_strength)
        env_map_x = grid_x - normal_x * displacement * 0.52 * mask_gain
        env_map_y = grid_y - normal_y * displacement * 0.52 * mask_gain
        reflected_env = cv2.remap(env_source, env_map_x, env_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101).astype("float32")
        directional_gate = np.ones((height, width), dtype="float32")
        if light_gate is not None:
            directional_gate = np.clip(0.22 + 0.78 * light_gate, 0.0, 1.0)
        plate_bounce = None
        if image.ndim == 3 and image.shape[2] >= 3 and lens_strength > 0.001 and light_strength > 0.001:
            try:
                # A reflected plate ray can hit a droplet away from the source
                # pixel. Approximate that lateral transport by sampling the
                # plate along the reflected in-plane light direction and
                # attenuating farther samples with an inverse-square-like law.
                image_f = image.astype("float32")
                ray_span = (3.0 + 32.0 * lens_strength) * (0.55 + 0.80 * light_strength)
                ray_span *= max(0.68, min(2.65, 1.18 / max(0.12, light_source_z)))
                attenuation_scale = max(12.0, 22.0 + 44.0 * light_source_z)
                bounce_sum = np.zeros_like(image_f, dtype="float32")
                bounce_peak = np.zeros_like(image_f, dtype="float32")
                bounce_weight = np.zeros((height, width), dtype="float32")
                for ratio in (0.0, 0.18, 0.36, 0.58, 0.82, 1.0):
                    distance = 1.0 + ray_span * float(ratio)
                    attenuation = 1.0 / (1.0 + (distance / attenuation_scale) ** 2)
                    sigma = 0.35 + 0.42 * float(ratio) + 0.32 * lens_strength
                    source = cv2.GaussianBlur(image_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
                    sample_x = (grid_x - light_x * distance).astype("float32")
                    sample_y = (grid_y - light_y * distance).astype("float32")
                    sample = cv2.remap(
                        source,
                        sample_x,
                        sample_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT101,
                    ).astype("float32")
                    gate_sample = cv2.remap(
                        directional_gate,
                        sample_x,
                        sample_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT101,
                    ).astype("float32")
                    weight = (
                        float(attenuation)
                        * (1.0 - 0.08 * float(ratio))
                        * (0.42 + 0.58 * gate_sample)
                    )
                    bounce_sum += sample * weight[:, :, None]
                    bounce_peak = np.maximum(
                        bounce_peak,
                        sample * np.clip(weight * 2.35, 0.0, 1.0)[:, :, None],
                    )
                    bounce_weight += weight
                if float(bounce_weight.max() or 0.0) > 0.0001:
                    bounce_average = bounce_sum / np.maximum(bounce_weight[:, :, None], 1e-6)
                    plate_bounce = np.clip(bounce_average * 0.52 + bounce_peak * 0.92, 0.0, 255.0)
                    reflected_env = np.clip(reflected_env * 0.56 + plate_bounce * 0.44, 0.0, 255.0)
            except Exception:
                plate_bounce = None
        reflection_angle = np.clip(
            fresnel * 0.66
            + rim * 0.30
            + (ndoth ** (15.0 + 18.0 * lens_strength)) * 0.48
            + np.clip(1.0 - ndotl, 0.0, 1.0) * 0.16,
            0.0,
            1.0,
        )
        reflection_coeff = np.clip(
            reflection_angle
            * thickness
            * (0.10 + 0.65 * lens_strength)
            * material_visibility
            * (0.44 + 0.56 * directional_gate),
            0.0,
            0.80,
        )
        if image.ndim == 2:
            out = out * (1.0 - reflection_coeff) + reflected_env * reflection_coeff
        else:
            out = out * (1.0 - reflection_coeff[:, :, None]) + reflected_env * reflection_coeff[:, :, None]
        micro_map_x = grid_x + normal_x * displacement * (0.82 + 0.72 * lens_strength) * mask_gain
        micro_map_y = grid_y + normal_y * displacement * (0.82 + 0.72 * lens_strength) * mask_gain
        micro_reflection = cv2.remap(
            image,
            micro_map_x.astype("float32"),
            micro_map_y.astype("float32"),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        ).astype("float32")
        if plate_bounce is not None:
            micro_reflection = np.clip(micro_reflection * 0.54 + plate_bounce * 0.46, 0.0, 255.0)
        micro_coeff = np.clip(
            (rim * 0.42 + fresnel * 0.30 + grad_energy * 0.22)
            * thickness
            * lens_strength
            * (0.16 + 0.52 * rain_alpha)
            * material_visibility
            * (0.54 + 0.46 * directional_gate),
            0.0,
            0.64,
        )
        if image.ndim == 2:
            out = out * (1.0 - micro_coeff) + micro_reflection * micro_coeff
        else:
            out = out * (1.0 - micro_coeff[:, :, None]) + micro_reflection * micro_coeff[:, :, None]
        if plate_bounce is not None:
            bounce_coeff = np.clip(
                (thickness * 0.22 + rim * 0.20 + grad_energy * 0.12)
                * lens_strength
                * light_strength
                * material_visibility
                * (0.28 + 0.72 * directional_gate),
                0.0,
                0.36,
            )
            out = out * (1.0 - bounce_coeff[:, :, None]) + plate_bounce * bounce_coeff[:, :, None]
        specular = np.clip(
            ((ndoth ** (12.0 + 24.0 * lens_strength)) * 1.18 + front_response * 0.26 + grad_energy * 0.12)
            * rim
            * lens
            * lens_strength
            * (0.22 + 0.86 * light_strength)
            * directional_gate,
            0.0,
            0.95,
        )
        internal = np.clip(
            (back_response * 0.64 + grad_energy * 0.18)
            * rim
            * lens
            * lens_strength
            * (0.12 + 0.58 * light_strength),
            0.0,
            0.62,
        )
        internal = np.clip(internal * (0.38 + 0.62 * directional_gate), 0.0, 0.62)
        shift = max(1, int(round((1.4 + 4.8 * lens_strength) * (0.65 + 0.55 * light_strength))))
        internal = cv2.warpAffine(
            internal,
            np.float32([[1, 0, light_x * shift], [0, 1, light_y * shift]]),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        caustic = np.abs(cv2.Laplacian(lens, cv2.CV_32F, ksize=3))
        caustic_scale = float(np.percentile(caustic, 99.0) or 0.0)
        if caustic_scale > 0.0001:
            caustic = np.clip(caustic / caustic_scale, 0.0, 1.0)
            caustic = cv2.warpAffine(
                caustic,
                np.float32([[1, 0, light_x * shift * 1.35], [0, 1, light_y * shift * 1.35]]),
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            caustic = cv2.GaussianBlur(caustic, (0, 0), sigmaX=0.48 + 0.52 * lens_strength, sigmaY=0.42 + 0.46 * lens_strength)
            caustic = np.clip(
                caustic * lens_strength * (0.06 + 0.32 * light_strength) * (0.32 + 0.68 * directional_gate),
                0.0,
                0.38,
            )
        else:
            caustic = np.zeros((height, width), dtype="float32")
        if image.ndim == 2:
            out = out + (247.0 - out) * specular * (0.62 + 0.30 * lens_strength)
            out = out + (255.0 - out) * internal * 0.44
            out = out + (255.0 - out) * caustic * 0.38
        else:
            refracted_color = cv2.GaussianBlur(
                refracted_primary.astype("float32"),
                (0, 0),
                sigmaX=0.65 + 0.95 * lens_strength,
                sigmaY=0.65 + 0.95 * lens_strength,
            )
            reflected_color = cv2.GaussianBlur(
                reflected_env.astype("float32"),
                (0, 0),
                sigmaX=0.80 + 1.20 * lens_strength,
                sigmaY=0.80 + 1.20 * lens_strength,
            )
            if source_color_field is not None:
                source_glint = np.clip(
                    source_color_field * 0.82 + source_color.reshape(1, 1, 3) * 0.18,
                    0.0,
                    255.0,
                )
            else:
                source_glint = source_color.reshape(1, 1, 3)
            direct_light = np.clip(
                directional_gate
                * lens
                * lens_strength
                * (0.10 + 0.72 * light_strength)
                * (0.42 + 0.58 * front_response),
                0.0,
                0.42,
            )
            reflected_tint = np.clip(
                reflected_color * (1.0 + 0.08 * front_response[:, :, None])
                + source_glint * (0.025 + 0.080 * light_strength) * directional_gate[:, :, None],
                0.0,
                255.0,
            )
            refracted_tint = np.clip(
                refracted_color
                + source_glint * direct_light[:, :, None],
                0.0,
                255.0,
            )
            glint_tint = np.clip(
                reflected_tint * (0.74 - 0.12 * light_strength)
                + source_glint * (0.18 + 0.16 * light_strength) * directional_gate[:, :, None],
                0.0,
                255.0,
            )
            internal_tint = np.clip(reflected_tint * 0.82 + refracted_tint * 0.18, 0.0, 255.0)
            caustic_tint = np.clip(
                refracted_tint
                + reflected_tint * (0.10 + 0.08 * light_strength),
                0.0,
                255.0,
            )
            out = out + (glint_tint - out) * specular[:, :, None] * (0.68 + 0.28 * lens_strength)
            out = out + (internal_tint - out) * internal[:, :, None] * 0.58
            out = out + (caustic_tint - out) * caustic[:, :, None] * 0.48
        return np.clip(out, 0, 255).astype("uint8")
    except Exception:
        return image


def _rng_gauss(rng, mean: float, sigma: float) -> float:
    try:
        return float(rng.gauss(mean, sigma))
    except Exception:
        return random.gauss(mean, sigma)


def _iter_stream_rain_anchors(
    width: int,
    height: int,
    count: int,
    margin_x: int,
    rng,
    *,
    step_x: float,
    step_y: float,
    strength: float,
    average_drop_size: float,
) -> list[tuple[int, int]]:
    """Return natural rain anchors arranged in stochastic streaks, not a grid."""
    if count <= 0 or width <= 0 or height <= 0:
        return []
    margin_x = max(0, int(margin_x or 0))
    direction_norm = max(0.001, math.sqrt(step_x * step_x + step_y * step_y))
    dir_x = float(step_x) / direction_norm
    dir_y = float(step_y) / direction_norm
    perp_x = -dir_y
    perp_y = dir_x
    center_x = (float(width) - 1.0) / 2.0
    center_y = (float(height) - 1.0) / 2.0
    span_w = float(width + 2 * margin_x)
    diagonal = max(1.0, math.hypot(span_w, float(height)))
    normal_limit = span_w * 0.49
    tangent_limit = max(float(height) * 0.62, diagonal * 0.36)
    strength = max(0.0, min(1.0, float(strength)))
    average_drop_size = max(0.0, min(1.0, float(average_drop_size)))

    stream_share = max(0.52, min(0.86, 0.66 + 0.18 * strength - 0.08 * average_drop_size))
    collision_share = max(0.03, min(0.14, 0.045 + 0.08 * strength + 0.025 * (1.0 - average_drop_size)))
    stream_drop_count = max(0, min(count, int(round(count * stream_share))))
    collision_count = max(0, min(count - stream_drop_count, int(round(count * collision_share))))
    background_count = max(0, count - stream_drop_count - collision_count)

    stream_count = max(
        3,
        min(
            max(3, stream_drop_count),
            int(round(math.sqrt(max(1.0, float(stream_drop_count))) * (0.65 + 1.15 * strength))),
        ),
    )
    streams = []
    for _stream in range(stream_count):
        streams.append(
            {
                "offset": _rng_float(rng, -normal_limit, normal_limit),
                "width": span_w * _rng_float(
                    rng,
                    0.004 + 0.006 * average_drop_size,
                    0.018 + 0.030 * (1.0 - average_drop_size),
                ),
                "phase": _rng_float(rng, 0.0, math.tau),
                "freq": _rng_float(rng, 0.65, 2.20),
                "weight": _rng_float(rng, 0.45, 1.85) ** 1.35,
            }
        )
    weight_sum = sum(float(item["weight"]) for item in streams) or 1.0
    cumulative = []
    running = 0.0
    for item in streams:
        running += float(item["weight"]) / weight_sum
        cumulative.append(running)

    def choose_stream():
        value = _rng_float(rng, 0.0, 1.0)
        for index, threshold in enumerate(cumulative):
            if value <= threshold:
                return streams[index]
        return streams[-1]

    def to_image(tangent: float, normal: float) -> tuple[int, int]:
        x = center_x + dir_x * tangent + perp_x * normal
        y = center_y + dir_y * tangent + perp_y * normal
        return int(round(x)), int(round(y))

    def clamp_anchor(x: int, y: int) -> tuple[int, int]:
        return (
            max(-margin_x, min(width - 1 + margin_x, int(x))),
            max(0, min(height - 1, int(y))),
        )

    anchors: list[tuple[int, int]] = []
    for _drop in range(stream_drop_count):
        stream = choose_stream()
        tangent = _rng_float(rng, -tangent_limit, tangent_limit)
        bend = math.sin((tangent / diagonal) * math.tau * float(stream["freq"]) + float(stream["phase"]))
        normal = (
            float(stream["offset"])
            + bend * float(stream["width"]) * _rng_float(rng, 0.10, 0.72)
            + _rng_gauss(rng, 0.0, max(0.45, float(stream["width"])))
        )
        anchors.append(clamp_anchor(*to_image(tangent, normal)))

    for _drop in range(collision_count):
        if anchors:
            base_x, base_y = anchors[_rng_int(rng, 0, len(anchors) - 1)]
        else:
            base_x = _rng_int(rng, -margin_x, max(1, width - 1 + margin_x))
            base_y = _rng_int(rng, 0, max(1, height - 1))
        radius = _rng_float(rng, 1.2, 6.5 + 8.0 * average_drop_size)
        angle = _rng_float(rng, 0.0, math.tau)
        # Collision neighbours are close enough to look like interacting drops,
        # but not so close that the mask grows into a single snowball blob.
        offset_x = math.cos(angle) * radius + dir_x * _rng_float(rng, -2.8, 2.8)
        offset_y = math.sin(angle) * radius + dir_y * _rng_float(rng, -2.8, 2.8)
        anchors.append(clamp_anchor(int(round(base_x + offset_x)), int(round(base_y + offset_y))))

    for _drop in range(background_count):
        # Background drops break the stream pattern and keep light rain from
        # looking like a set of artificial bands.
        anchors.append(
            (
                _rng_int(rng, -margin_x, max(1, width - 1 + margin_x)),
                _rng_int(rng, 0, max(1, height - 1)),
            )
        )

    if len(anchors) > count:
        anchors = anchors[:count]
    while len(anchors) < count:
        anchors.append(
            (
                _rng_int(rng, -margin_x, max(1, width - 1 + margin_x)),
                _rng_int(rng, 0, max(1, height - 1)),
            )
        )
    for index in range(len(anchors) - 1, 0, -1):
        swap_index = _rng_int(rng, 0, index)
        anchors[index], anchors[swap_index] = anchors[swap_index], anchors[index]
    return anchors


def _apply_rain_effect(
    image,
    profile: AugmentationProfile,
    rng,
    *,
    return_mask: bool = False,
    base_surface: dict | None = None,
    apply_lens: bool = True,
) -> object:
    if np is None or cv2 is None or profile.rain_strength <= 0:
        return (image, None) if return_mask else image

    try:
        strength = max(0.0, min(1.0, float(profile.rain_strength)))
        drop_size_min = max(0.0, min(1.0, float(getattr(profile, "rain_drop_size_min", profile.rain_drop_size) if getattr(profile, "rain_drop_size_min", None) is not None else profile.rain_drop_size)))
        drop_size_max = max(0.0, min(1.0, float(getattr(profile, "rain_drop_size_max", profile.rain_drop_size) if getattr(profile, "rain_drop_size_max", None) is not None else profile.rain_drop_size)))
        if drop_size_min > drop_size_max:
            drop_size_min, drop_size_max = drop_size_max, drop_size_min
        average_drop_size = (drop_size_min + drop_size_max) / 2.0
        density_soften = max(0.0, min(1.0, (strength - 0.42) / 0.58))
        drop_span = max(0.0, drop_size_max - drop_size_min)
        visual_drop_size_max = max(drop_size_min, drop_size_min + drop_span * (1.0 - 0.46 * density_soften))
        visual_average_drop_size = (drop_size_min + visual_drop_size_max) / 2.0
        vector_field_strength = max(0.0, min(1.0, float(getattr(profile, "rain_vector_field_strength", 0.0) or 0.0)))
        vortex_strength = max(0.0, min(1.0, float(getattr(profile, "rain_vortex_strength", 0.0) or 0.0)))
        rain_alpha = max(0.0, min(1.0, float(getattr(profile, "rain_alpha", 0.22) if getattr(profile, "rain_alpha", None) is not None else 0.22)))
        lens_strength = max(0.0, min(1.0, float(getattr(profile, "rain_lens_strength", 0.0) or 0.0)))
        edge_mist_strength = max(0.0, min(1.0, float(getattr(profile, "rain_edge_mist_strength", 0.0) or 0.0)))
        edge_mist_radius = max(0.0, min(1.0, float(getattr(profile, "rain_edge_mist_radius", 0.45) if getattr(profile, "rain_edge_mist_radius", None) is not None else 0.45)))
        wind_strength = max(0.0, min(1.0, float(profile.dirt_flow_wind_strength if profile.dirt_flow_wind_strength is not None else 0.0)))
        height, width = image.shape[:2]
        size_density = 1.55 - 0.68 * average_drop_size
        # Keep the density slider useful across the whole range. The first
        # version hit a hard cap too early, while the next one saturated the
        # rain mask near 0.8. This keeps growth visible without turning rain
        # into a uniform translucent sheet.
        density_load = 10.0 + 220.0 * strength + 4600.0 * (strength ** 1.75)
        drops = int(max(1, (width * height / 28000.0) * density_load * size_density))
        pixel_drop_cap = int(max(5000, min(MAX_RAIN_DROPS, (width * height) / 2.5)))
        drops = min(drops, pixel_drop_cap)
        base_length = 1.2 + 4.0 * strength + 6.0 * wind_strength
        overlay = image.copy()
        color = (
            int(145 + 34 * strength + 28 * average_drop_size),
            int(150 + 36 * strength + 28 * average_drop_size),
            int(158 + 40 * strength + 28 * average_drop_size),
        )
        step_x, step_y = _rain_drop_direction(profile, average_drop_size)
        max_drop_length_base = base_length * (0.18 + 1.35 * visual_drop_size_max) + 12.0 * visual_drop_size_max
        max_thickness = max(
            1,
            int(round((0.55 + 5.8 * (visual_drop_size_max ** 1.45) + 1.25 * strength) * (1.0 - 0.32 * density_soften))),
        )
        max_slant = int(math.ceil(abs(step_x * max_drop_length_base * 1.28))) + max_thickness + 3
        rain_anchors = _iter_stream_rain_anchors(
            width,
            height,
            drops,
            max_slant,
            rng,
            step_x=step_x,
            step_y=step_y,
            strength=strength,
            average_drop_size=average_drop_size,
        )
        edge_response = None
        edge_gate = None
        if edge_mist_strength > 0.001:
            stable_edge = _surface_float_mask(base_surface, "contour_curve", width, height)
            if stable_edge is None:
                stable_edge = _surface_float_mask(base_surface, "contour_edge", width, height)
            stable_normal_x = _surface_float_mask(base_surface, "contour_profile_normal_x", width, height, signed=True)
            stable_normal_y = _surface_float_mask(base_surface, "contour_profile_normal_y", width, height, signed=True)
            if stable_normal_x is None or stable_normal_y is None:
                stable_normal_x = _surface_float_mask(base_surface, "normal_x", width, height, signed=True)
                stable_normal_y = _surface_float_mask(base_surface, "normal_y", width, height, signed=True)
            if stable_edge is not None and stable_normal_x is not None and stable_normal_y is not None:
                edge_alignment = np.abs((stable_normal_x * step_x) + (stable_normal_y * step_y))
                local_gate = np.clip(stable_edge * (edge_alignment ** 2.35), 0.0, 1.0)
                dot = (step_x * stable_normal_x) + (step_y * stable_normal_y)
                reflect_x = step_x - 2.0 * dot * stable_normal_x
                reflect_y = step_y - 2.0 * dot * stable_normal_y
                reflect_norm = np.sqrt(reflect_x * reflect_x + reflect_y * reflect_y) + 1e-6
                edge_response = {
                    "gate": cv2.GaussianBlur(local_gate, (0, 0), sigmaX=0.55, sigmaY=0.55),
                    "reflect_x": reflect_x / reflect_norm,
                    "reflect_y": reflect_y / reflect_norm,
                }
            if edge_response is None:
                edge_response = _build_rain_edge_response(image, step_x, step_y)
            if edge_response:
                edge_gate = edge_response.get("gate")
        vortices = []
        if vortex_strength > 0.001 and width > 4 and height > 4:
            vortex_count = max(1, min(9, int(round(1 + 5 * vortex_strength + 2 * strength))))
            vortex_scale = max(12.0, min(float(width), float(height)) * (0.16 + 0.18 * vortex_strength))
            for _vortex_index in range(vortex_count):
                radius = vortex_scale * _rng_float(rng, 0.72, 1.55)
                spin = _rng_float(rng, 0.75, 1.85) * vortex_strength
                if _rng_float(rng, 0.0, 1.0) < 0.5:
                    spin = -spin
                vortices.append(
                    (
                        _rng_float(rng, -0.08 * width, 1.08 * width),
                        _rng_float(rng, -0.05 * height, 1.05 * height),
                        max(8.0, radius),
                        spin,
                    )
                )
        light_context = _active_relief_light_context(profile, height, width)
        if light_context:
            light_strength = max(0.0, min(1.0, float(light_context.get("strength", 0.0) or 0.0)))
            light_normal = max(0.0, min(1.0, float(profile.light_normal_strength or 0.0)))
            light_dir_x = float(light_context.get("x", 0.0) or 0.0)
            light_dir_y = float(light_context.get("y", 0.0) or 0.0)
            scene_beam_strength = light_strength
        else:
            light_strength = 0.0
            light_normal = 0.0
            light_dir_x, light_dir_y = step_x, step_y
            scene_beam_strength = 0.0
        gloss_strength = max(
            0.0,
            min(
                1.0,
                strength
                * (0.12 + 0.62 * light_strength + 0.34 * light_normal + 0.28 * scene_beam_strength)
                * (0.55 + 0.45 * average_drop_size),
            ),
        )
        gloss_probability = max(0.0, min(0.68, 0.08 + 0.44 * gloss_strength + 0.16 * average_drop_size))
        rain_mask = np.zeros((height, width), dtype="float32")
        gloss_mask = np.zeros((height, width), dtype="float32") if gloss_strength > 0.015 else None
        for anchor_x, anchor_y in rain_anchors:
            local_drop_size = _rng_float(rng, drop_size_min, visual_drop_size_max)
            drop_opacity = max(
                0.05,
                min(
                    0.46,
                    (0.20 + 0.26 * local_drop_size + 0.05 * strength) * (1.0 - 0.48 * density_soften),
                ),
            )
            local_length_base = (
                base_length * (0.16 + 1.15 * local_drop_size)
                + 8.5 * local_drop_size * (1.0 - 0.25 * density_soften)
            )
            local_thickness = max(
                1,
                int(round((0.45 + 4.3 * (local_drop_size ** 1.35) + 0.74 * strength) * (1.0 - 0.42 * density_soften))),
            )
            x1 = anchor_x
            y1 = anchor_y
            local_length = local_length_base * _rng_float(rng, 0.72, 1.28)
            local_step_x = step_x
            local_step_y = step_y
            if vector_field_strength > 0.001:
                nx = x1 / max(1.0, float(width))
                ny = y1 / max(1.0, float(height))
                phase_jitter = _rng_float(rng, -0.18, 0.18)
                field_a = math.sin((nx * 3.1 + ny * 1.7) * math.tau + phase_jitter)
                field_b = math.cos((nx * 1.2 - ny * 2.6) * math.tau - phase_jitter)
                curl = (field_a * 0.68 + field_b * 0.32) * vector_field_strength
                local_step_x = step_x + curl * (0.45 + 1.15 * (1.0 - local_drop_size))
                local_step_y = max(0.16, step_y + field_b * vector_field_strength * 0.12)
            if vortices:
                vortex_x = 0.0
                vortex_y = 0.0
                for center_x, center_y, radius, spin in vortices:
                    rel_x = (x1 - center_x) / radius
                    rel_y = (y1 - center_y) / radius
                    dist2 = rel_x * rel_x + rel_y * rel_y
                    if dist2 > 5.5:
                        continue
                    falloff = math.exp(-dist2 * 1.35)
                    vortex_x += -rel_y * spin * falloff
                    vortex_y += rel_x * spin * falloff * 0.42
                if abs(vortex_x) > 0.0001 or abs(vortex_y) > 0.0001:
                    small_drop_gain = 0.55 + 1.35 * (1.0 - local_drop_size)
                    local_step_x += vortex_x * small_drop_gain
                    local_step_y = max(0.10, local_step_y + vortex_y * small_drop_gain)
                    local_length *= max(0.36, 1.0 + 0.18 * vortex_x)
            if vector_field_strength > 0.001 or vortices:
                local_norm = max(0.001, math.sqrt(local_step_x * local_step_x + local_step_y * local_step_y))
                local_step_x /= local_norm
                local_step_y /= local_norm
            if vector_field_strength > 0.001:
                local_length *= max(0.38, 1.0 + vector_field_strength * 0.24 * field_a)
            x2 = int(round(x1 + local_step_x * local_length))
            y2 = int(round(y1 + local_step_y * local_length))
            cv2.line(overlay, (x1, y1), (x2, y2), color, local_thickness, lineType=cv2.LINE_AA)
            cv2.line(rain_mask, (x1, y1), (x2, y2), drop_opacity, max(1, local_thickness), lineType=cv2.LINE_AA)
            if gloss_mask is not None and _rng_float(rng, 0.0, 1.0) <= gloss_probability:
                highlight_t = _rng_float(rng, 0.12, 0.52)
                hx = x1 + local_step_x * local_length * highlight_t - light_dir_x * _rng_float(rng, 0.8, 2.8 + 3.0 * local_drop_size)
                hy = y1 + local_step_y * local_length * highlight_t - light_dir_y * _rng_float(rng, 0.8, 2.8 + 3.0 * local_drop_size)
                h_len = max(1.0, local_length * _rng_float(rng, 0.10, 0.26) * (0.55 + 0.75 * local_drop_size))
                h_thick = max(1, int(round(local_thickness * _rng_float(rng, 0.55, 0.95))))
                h_alpha = _rng_float(rng, 0.36, 0.94) * gloss_strength * (1.0 - 0.25 * density_soften)
                hp1 = (
                    int(round(hx - local_step_x * h_len * 0.5)),
                    int(round(hy - local_step_y * h_len * 0.5)),
                )
                hp2 = (
                    int(round(hx + local_step_x * h_len * 0.5)),
                    int(round(hy + local_step_y * h_len * 0.5)),
                )
                cv2.line(gloss_mask, hp1, hp2, h_alpha, h_thick, lineType=cv2.LINE_AA)
                if local_drop_size > 0.35:
                    radius = max(1, int(round((0.8 + 1.5 * local_drop_size + 0.55 * light_normal) * (1.0 - 0.30 * density_soften))))
                    cv2.circle(gloss_mask, (int(round(hx)), int(round(hy))), radius, h_alpha * 0.72, -1, lineType=cv2.LINE_AA)
        alpha = max(0.0, min(1.0, rain_alpha))
        if rain_mask is not None and rain_mask.size:
            rain_mask = cv2.GaussianBlur(rain_mask, (0, 0), sigmaX=0.22 + 0.35 * visual_average_drop_size, sigmaY=0.22 + 0.35 * visual_average_drop_size)
            mask_gain = 0.34 + 0.46 * strength + 0.22 * visual_average_drop_size
            mask_cap = 0.48 + 0.16 * (1.0 - density_soften)
            rain_mask = np.clip(rain_mask * mask_gain, 0.0, mask_cap)
        lens_surface = rain_mask
        # The slider must be authoritative: 0 means no droplet lensing, 1 means
        # the strongest refraction. Rain density/size still shape the mask, but
        # they should not silently add lensing when the user disables it.
        effective_lens_strength = lens_strength
        if apply_lens and effective_lens_strength > 0.001:
            lens_base = _apply_rain_lens_distortion(image, lens_surface, effective_lens_strength, profile)
        else:
            lens_base = image
        if rain_mask is not None and rain_mask.size:
            opacity_surface = rain_mask
            max_local_alpha = max(0.012, min(0.34, 0.018 + 0.30 * alpha * (1.0 - 0.26 * density_soften)))
            local_alpha = np.clip(
                opacity_surface * (0.10 + 0.40 * alpha + 0.10 * strength),
                0.0,
                max_local_alpha,
            )
            if lens_base.ndim == 2:
                out = lens_base.astype("float32") * (1.0 - local_alpha) + overlay.astype("float32") * local_alpha
            else:
                out = lens_base.astype("float32") * (1.0 - local_alpha[:, :, None]) + overlay.astype("float32") * local_alpha[:, :, None]
            result = np.clip(out, 0, 255).astype("uint8")
        else:
            result = cv2.addWeighted(overlay, alpha, lens_base, 1.0 - alpha, 0)
        if gloss_mask is not None and gloss_mask.size:
            gloss_mask = cv2.GaussianBlur(
                gloss_mask,
                (0, 0),
                sigmaX=0.18 + 0.42 * average_drop_size,
                sigmaY=0.18 + 0.36 * average_drop_size,
            )
            gloss_mask = np.clip(gloss_mask * (0.72 + 0.58 * light_normal), 0.0, 1.0)
            out = result.astype("float32")
            tint = np.array([245.0, 250.0, 255.0], dtype="float32")
            screen = out + (tint - out) * gloss_mask[:, :, None]
            out = out * (1.0 - gloss_mask[:, :, None] * 0.10) + screen * (gloss_mask[:, :, None] * 0.68)
            result = np.clip(out, 0, 255).astype("uint8")
        if edge_gate is not None:
            mist_scale = 1.0
            mist_width = width
            mist_height = height
            mist_gate = edge_gate
            reflect_x = edge_response.get("reflect_x") if edge_response else None
            reflect_y = edge_response.get("reflect_y") if edge_response else None
            if width * height > MAX_RAIN_EDGE_PIXELS:
                mist_scale = math.sqrt(MAX_RAIN_EDGE_PIXELS / float(width * height))
                mist_width = max(4, int(round(width * mist_scale)))
                mist_height = max(4, int(round(height * mist_scale)))
                mist_gate = cv2.resize(edge_gate, (mist_width, mist_height), interpolation=cv2.INTER_AREA)
                if reflect_x is not None and reflect_y is not None:
                    reflect_x = cv2.resize(reflect_x, (mist_width, mist_height), interpolation=cv2.INTER_LINEAR)
                    reflect_y = cv2.resize(reflect_y, (mist_width, mist_height), interpolation=cv2.INTER_LINEAR)
                    reflect_norm = np.sqrt(reflect_x * reflect_x + reflect_y * reflect_y) + 1e-6
                    reflect_x = reflect_x / reflect_norm
                    reflect_y = reflect_y / reflect_norm
            impact_mask = np.clip(mist_gate * 0.28, 0.0, 1.0)
            upstream_steps = max(2, int(round(3 + 11 * edge_mist_radius + 5 * strength)))
            step_distance = 0.8 + 8.5 * edge_mist_radius + 2.2 * strength
            for step_index in range(1, upstream_steps + 1):
                decay = 1.0 - (step_index / float(upstream_steps + 1))
                distance = step_index * step_distance * mist_scale
                transform = np.float32(
                    [
                        [1, 0, float(-step_x) * distance],
                        [0, 1, float(-step_y) * distance],
                    ]
                )
                shifted = cv2.warpAffine(
                    mist_gate,
                    transform,
                    (mist_width, mist_height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                impact_mask = np.maximum(impact_mask, np.clip(shifted * decay, 0.0, 1.0))
            if reflect_x is not None and reflect_y is not None:
                reflection_mask = np.zeros_like(impact_mask)
                direction_bins = 10
                min_match = math.cos(math.pi / float(direction_bins))
                reflect_steps = max(2, int(round(2 + 8 * edge_mist_radius + 4 * strength)))
                reflect_distance = step_distance * (0.72 + 0.46 * strength + 0.38 * edge_mist_radius)
                for bin_index in range(direction_bins):
                    angle = (math.tau * float(bin_index)) / float(direction_bins)
                    dir_x = math.cos(angle)
                    dir_y = math.sin(angle)
                    match = ((reflect_x * dir_x) + (reflect_y * dir_y) - min_match) / max(1e-6, 1.0 - min_match)
                    bin_gate = np.clip(mist_gate * np.clip(match, 0.0, 1.0), 0.0, 1.0)
                    if float(np.max(bin_gate) or 0.0) <= 0.002:
                        continue
                    for step_index in range(1, reflect_steps + 1):
                        decay = 1.0 - (step_index / float(reflect_steps + 1))
                        distance = step_index * reflect_distance * mist_scale
                        transform = np.float32(
                            [
                                [1, 0, float(dir_x) * distance],
                                [0, 1, float(dir_y) * distance],
                            ]
                        )
                        shifted = cv2.warpAffine(
                            bin_gate,
                            transform,
                            (mist_width, mist_height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        reflection_mask = np.maximum(
                            reflection_mask,
                            np.clip(shifted * decay, 0.0, 1.0),
                        )
                if reflection_mask.size:
                    impact_mask = np.maximum(
                        impact_mask,
                        reflection_mask * (0.52 + 0.46 * strength + 0.28 * edge_mist_radius),
                    )
            mist_sigma = 0.9 + 10.0 * edge_mist_radius + 2.4 * strength
            mist_mask = cv2.GaussianBlur(
                impact_mask,
                (0, 0),
                sigmaX=max(0.45, mist_sigma * mist_scale),
                sigmaY=max(0.45, mist_sigma * mist_scale * 0.72),
            )
            mist_mask = np.clip(mist_mask * edge_mist_strength * (0.82 + 1.38 * strength), 0.0, 0.88)
            if mist_width != width or mist_height != height:
                mist_mask = cv2.resize(mist_mask, (width, height), interpolation=cv2.INTER_LINEAR)
            if mist_mask.size:
                if width * height > MAX_RAIN_EDGE_PIXELS:
                    mist_u8 = np.clip(mist_mask * 255.0, 0.0, 255.0).astype("uint8")
                    tint_layer = np.empty_like(result)
                    tint_layer[:, :] = (224, 235, 242)
                    blended = cv2.addWeighted(result, 0.58, tint_layer, 0.42, 0)
                    cv2.copyTo(blended, mist_u8, result)
                else:
                    out = result.astype("float32")
                    mist_tint = np.array([224.0, 235.0, 242.0], dtype="float32")
                    out = out * (1.0 - mist_mask[:, :, None] * 0.58) + mist_tint * (mist_mask[:, :, None] * 0.58)
                    result = np.clip(out, 0, 255).astype("uint8")
        return (result, rain_mask) if return_mask else result
    except Exception:
        return (image, None) if return_mask else image


def _apply_wet_reflection_effect(image, profile: AugmentationProfile, rng) -> object:
    if np is None or cv2 is None or profile.rain_strength <= 0 or profile.wet_reflection_strength <= 0:
        return image

    try:
        rain = max(0.0, min(1.0, float(profile.rain_strength or 0.0)))
        strength = max(0.0, min(1.0, float(profile.wet_reflection_strength or 0.0))) * (0.35 + 0.65 * rain)
        if strength <= 0.001:
            return image

        height, width = image.shape[:2]
        if height < 12 or width < 12:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        bright_mask = np.clip((gray - 0.34) / 0.58, 0.0, 1.0)
        bright_mask = cv2.GaussianBlur(bright_mask, (0, 0), sigmaX=1.8 + 3.2 * rain, sigmaY=1.8 + 3.2 * rain)
        noise_h = max(3, int(height / 18))
        noise_w = max(3, int(width / 18))
        film_noise = np.array(
            [[_rng_float(rng, 0.0, 1.0) for _x in range(noise_w)] for _y in range(noise_h)],
            dtype="float32",
        )
        film_noise = cv2.resize(film_noise, (width, height), interpolation=cv2.INTER_CUBIC)
        film_noise = cv2.GaussianBlur(film_noise, (0, 0), sigmaX=4.0 + 8.0 * rain, sigmaY=4.0 + 8.0 * rain)
        wet_mask = np.clip((bright_mask * (0.70 + 0.30 * rain)) + (film_noise * 0.16 * rain), 0.0, 1.0)

        light_dir_x, light_dir_y, scene_beam_strength = _scene_or_fallback_light_direction(profile, height, width)
        light_dir = np.array([light_dir_x, light_dir_y], dtype="float32")
        light_norm = float(np.linalg.norm(light_dir) or 1.0)
        light_dir = light_dir / light_norm
        tangent = np.array([-float(light_dir[1]), float(light_dir[0])], dtype="float32")

        gloss = np.zeros((height, width), dtype="float32")
        yy, xx = np.mgrid[0:height, 0:width].astype("float32")
        xx -= width * 0.5
        yy -= height * 0.5
        proj_t = xx * float(tangent[0]) + yy * float(tangent[1])
        proj_n = xx * float(light_dir[0]) + yy * float(light_dir[1])
        normal_extent = max(1.0, float(np.max(np.abs(proj_n)) or 1.0))
        tangent_extent = max(1.0, float(np.max(np.abs(proj_t)) or 1.0))

        band_count = max(1, int(round(1 + 4 * strength + 2 * rain)))
        for _ in range(band_count):
            center_n = _rng_float(rng, -normal_extent * 0.72, normal_extent * 0.72)
            center_t = _rng_float(rng, -tangent_extent * 0.42, tangent_extent * 0.42)
            sigma_n = _rng_float(rng, 5.0, max(6.0, min(width, height) * (0.035 + 0.075 * strength)))
            sigma_t = _rng_float(rng, max(18.0, tangent_extent * 0.34), max(24.0, tangent_extent * (0.72 + 0.22 * rain)))
            band = np.exp(-((proj_n - center_n) ** 2) / (2.0 * sigma_n * sigma_n))
            band *= np.exp(-((proj_t - center_t) ** 2) / (2.0 * sigma_t * sigma_t))
            gloss += band.astype("float32") * _rng_float(rng, 0.16, 0.50) * strength

        base_len = min(width, height) * (0.08 + 0.28 * strength)
        glint_count = max(1, int(round((width * height / 72000.0) * (1.0 + 8.0 * strength))))
        for _ in range(glint_count):
            cx = _rng_int(rng, int(width * 0.05), max(int(width * 0.95), int(width * 0.05) + 1))
            cy = _rng_int(rng, int(height * 0.08), max(int(height * 0.92), int(height * 0.08) + 1))
            local_angle = math.atan2(float(tangent[1]), float(tangent[0])) + _rng_float(rng, -0.35, 0.35)
            local_dir = np.array([math.cos(local_angle), math.sin(local_angle)], dtype="float32")
            length = base_len * _rng_float(rng, 0.35, 1.05)
            half = local_dir * (length * 0.5)
            offset = light_dir * _rng_float(rng, -5.0, 9.0)
            p1 = (
                int(round(cx - float(half[0]) + float(offset[0]))),
                int(round(cy - float(half[1]) + float(offset[1]))),
            )
            p2 = (
                int(round(cx + float(half[0]) + float(offset[0]))),
                int(round(cy + float(half[1]) + float(offset[1]))),
            )
            thickness = max(1, int(round(min(width, height) * _rng_float(rng, 0.003, 0.012) * (0.70 + strength))))
            cv2.line(gloss, p1, p2, _rng_float(rng, 0.08, 0.34) * strength, thickness=thickness, lineType=cv2.LINE_AA)

        tiny_count = max(1, int(round((width * height / 90000.0) * (1.0 + 8.0 * strength))))
        for _ in range(tiny_count):
            x = _rng_int(rng, 0, max(1, width - 1))
            y = _rng_int(rng, 0, max(1, height - 1))
            if wet_mask[y, x] < 0.18:
                continue
            radius = max(1, int(round(_rng_float(rng, 0.8, 2.2 + 2.8 * strength))))
            cv2.circle(gloss, (x, y), radius, _rng_float(rng, 0.08, 0.38) * strength, -1, lineType=cv2.LINE_AA)

        gloss = cv2.GaussianBlur(gloss, (0, 0), sigmaX=1.2 + 3.5 * strength, sigmaY=0.9 + 2.6 * strength)
        percentile = float(np.percentile(gloss, 99.4) or 0.0)
        if percentile > 0:
            gloss = gloss / percentile
        gloss = np.clip(gloss * wet_mask * (0.10 + 0.55 * strength + 0.22 * scene_beam_strength), 0.0, 0.72)

        tint = np.array([245.0, 248.0, 255.0], dtype="float32")
        out = image.astype("float32")
        damp = wet_mask[:, :, None] * (0.015 + 0.055 * rain * strength)
        out = out * (1.0 - damp) + np.array([205.0, 210.0, 218.0], dtype="float32") * damp
        screen = out + (tint - out) * gloss[:, :, None]
        out = out * (1.0 - gloss[:, :, None] * 0.10) + screen * (gloss[:, :, None] * 0.58)
        return np.clip(out, 0, 255).astype("uint8")
    except Exception:
        return image


def _apply_water_film_effect(
    image,
    profile: AugmentationProfile,
    rng,
    relief_mask=None,
    *,
    return_mask: bool = False,
    base_surface: dict | None = None,
) -> object:
    if np is None or cv2 is None:
        return (image, None) if return_mask else image

    try:
        strength = max(0.0, min(1.0, float(getattr(profile, "water_film_strength", 0.0) or 0.0)))
        unevenness = max(0.0, min(1.0, float(getattr(profile, "water_film_unevenness", 0.35) if getattr(profile, "water_film_unevenness", None) is not None else 0.35)))
        lens_strength = max(0.0, min(1.0, float(getattr(profile, "water_film_lens_strength", 0.0) or 0.0)))
        contour_response = max(0.0, min(1.0, float(getattr(profile, "water_film_contour_response", 0.55) if getattr(profile, "water_film_contour_response", None) is not None else 0.55)))
        gloss_strength = max(0.0, min(1.0, float(getattr(profile, "water_film_gloss_strength", 0.45) if getattr(profile, "water_film_gloss_strength", None) is not None else 0.45)))
        if strength <= 0.001 and lens_strength > 0.001:
            strength = min(1.0, max(0.18, lens_strength * 0.42))
        if strength <= 0.001:
            return (image, None) if return_mask else image
        effective_profile = profile
        try:
            if abs(float(getattr(profile, "water_film_strength", 0.0) or 0.0) - strength) > 0.0001:
                effective_profile = replace(profile, water_film_strength=strength)
        except Exception:
            effective_profile = profile

        height, width = image.shape[:2]
        if height < 8 or width < 8:
            return (image, None) if return_mask else image

        surface = _build_contour_blanket_surface(
            image,
            effective_profile,
            rng,
            relief_mask=relief_mask,
            include_water=True,
            base_surface=base_surface,
        )
        if not surface:
            return (image, None) if return_mask else image
        symbol_height = surface.get("symbol_height")
        contour_edge = surface.get("contour_edge")
        lower_symbol_edge = surface.get("lower_symbol_edge")
        lower_edge_pool = surface.get("lower_edge_pool")
        bottom_edge_pool = surface.get("bottom_edge_pool")
        left_corner_pool = surface.get("left_corner_pool")
        right_corner_pool = surface.get("right_corner_pool")
        water_accumulation = surface.get("water_accumulation")
        water_mask = surface.get("water_mask")
        height_field = surface.get("height")
        xx = surface.get("grid_x")
        yy = surface.get("grid_y")
        if any(
            value is None
            for value in (
                symbol_height,
                contour_edge,
                lower_symbol_edge,
                lower_edge_pool,
                bottom_edge_pool,
                left_corner_pool,
                right_corner_pool,
                water_accumulation,
                water_mask,
                height_field,
                xx,
                yy,
            )
        ):
            return (image, None) if return_mask else image

        out = image.astype("float32")
        if lens_strength > 0.001:
            film_grad_x = cv2.Sobel(height_field, cv2.CV_32F, 1, 0, ksize=3)
            film_grad_y = cv2.Sobel(height_field, cv2.CV_32F, 0, 1, ksize=3)
            grad_scale = float(np.percentile(np.abs(film_grad_x) + np.abs(film_grad_y), 99.0) or 0.0)
            if grad_scale > 0.0001:
                film_grad_x = film_grad_x / grad_scale
                film_grad_y = film_grad_y / grad_scale
            min_dim = max(1.0, float(min(width, height)))
            max_shift = min(
                10.5,
                max(
                    0.65,
                    min_dim
                    * (0.008 + 0.044 * lens_strength)
                    * (0.45 + 0.78 * strength)
                    * (0.58 + 0.72 * unevenness),
                ),
            )
            mask_gain = np.clip(0.30 + water_mask * (0.72 + 0.36 * lens_strength), 0.0, 1.0)
            bulge_lens_gain = np.clip(1.0 + water_accumulation * (0.85 + 1.25 * strength), 1.0, 2.65)
            shift_x = film_grad_x * max_shift * mask_gain * bulge_lens_gain
            shift_y = film_grad_y * max_shift * mask_gain * bulge_lens_gain
            map_x = (xx + shift_x).astype("float32")
            map_y = (yy + shift_y).astype("float32")
            refracted = cv2.remap(
                out,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            if len(image.shape) == 3 and image.shape[2] >= 3:
                chroma_shift = max(0.18, max_shift * (0.16 + 0.18 * lens_strength))
                blue = cv2.remap(
                    out[:, :, 0],
                    (xx + shift_x + film_grad_x * chroma_shift).astype("float32"),
                    (yy + shift_y + film_grad_y * chroma_shift * 0.55).astype("float32"),
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                red = cv2.remap(
                    out[:, :, 2],
                    (xx + shift_x - film_grad_x * chroma_shift * 0.72).astype("float32"),
                    (yy + shift_y - film_grad_y * chroma_shift * 0.38).astype("float32"),
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                refracted[:, :, 0] = blue
                refracted[:, :, 2] = red
            refraction_alpha = np.clip(
                (water_mask + water_accumulation * strength * 0.42) * (0.44 + 0.66 * lens_strength),
                0.0,
                0.95,
            )
            out = out * (1.0 - refraction_alpha[:, :, None]) + refracted * refraction_alpha[:, :, None]
            caustic = cv2.Laplacian(height_field, cv2.CV_32F, ksize=3)
            caustic = np.abs(caustic)
            caustic_scale = float(np.percentile(caustic, 98.5) or 0.0)
            if caustic_scale > 0.0001:
                caustic = np.clip(caustic / caustic_scale, 0.0, 1.0)
                caustic = cv2.GaussianBlur(caustic, (0, 0), sigmaX=0.34 + 0.48 * unevenness, sigmaY=0.34 + 0.48 * unevenness)
                caustic = np.clip(caustic * water_mask * lens_strength * (0.08 + 0.28 * strength), 0.0, 0.34)
                out = out + (255.0 - out) * caustic[:, :, None]

        gray = cv2.cvtColor(np.clip(out, 0, 255).astype("uint8"), cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        bright_mask = cv2.GaussianBlur(np.clip((gray - 0.28) / 0.66, 0.0, 1.0), (0, 0), sigmaX=0.75, sigmaY=0.75)
        scene_lights = _combined_scene_headlight_fields(effective_profile, height, width)
        scene_field = None
        scene_specular = None
        averaged_scene = _average_scene_light_direction(scene_lights)
        if averaged_scene is not None:
            light_x, light_y, scene_beam_strength = averaged_scene
            try:
                local_scene_field = np.clip(np.asarray(scene_lights.get("field"), dtype="float32"), 0.0, 1.0)
                if local_scene_field.shape == (height, width):
                    scene_field = local_scene_field
            except Exception:
                scene_field = None
            try:
                local_scene_specular = np.clip(np.asarray(scene_lights.get("specular"), dtype="float32"), 0.0, 1.0)
                if local_scene_specular.shape == (height, width):
                    scene_specular = local_scene_specular
            except Exception:
                scene_specular = None
        else:
            light_x, light_y, scene_beam_strength = _scene_or_fallback_light_direction(effective_profile, height, width)
        film_grad_x = cv2.Sobel(height_field, cv2.CV_32F, 1, 0, ksize=3)
        film_grad_y = cv2.Sobel(height_field, cv2.CV_32F, 0, 1, ksize=3)
        grad_energy = np.abs(film_grad_x) + np.abs(film_grad_y)
        grad_energy_scale = float(np.percentile(grad_energy, 99.0) or 0.0)
        if grad_energy_scale > 0.0001:
            grad_energy = np.clip(grad_energy / grad_energy_scale, 0.0, 1.0)
            film_grad_x = film_grad_x / grad_energy_scale
            film_grad_y = film_grad_y / grad_energy_scale
        light_response = np.clip((film_grad_x * light_x + film_grad_y * light_y) * 3.4 + 0.40 + grad_energy * 0.34, 0.0, 1.0)
        if scene_field is not None:
            light_response = np.clip(light_response * (0.58 + 0.92 * scene_field), 0.0, 1.0)
        if scene_specular is not None:
            light_response = np.maximum(light_response, np.clip(scene_specular * (0.68 + 0.92 * gloss_strength), 0.0, 1.0))
        beam_strength = max(
            float(getattr(profile, "night_light_strength", 0.0) or 0.0) * 0.48,
            float(getattr(profile, "traffic_headlight_strength", 0.0) or 0.0),
            float(getattr(profile, "traffic_headlight_2_strength", 0.0) or 0.0),
            float(getattr(profile, "traffic_headlight_3_strength", 0.0) or 0.0),
            scene_beam_strength,
            0.18,
        )
        gloss = np.clip(
            water_mask
            * light_response
            * (0.20 + 0.80 * gloss_strength)
            * (0.42 + 0.72 * beam_strength)
            * (0.52 + 0.48 * bright_mask),
            0.0,
            1.0,
        )
        if scene_specular is not None:
            gloss = np.maximum(
                gloss,
                np.clip(water_mask * scene_specular * (0.45 + 0.95 * gloss_strength) * (0.50 + 0.60 * beam_strength), 0.0, 1.0),
            )
        accumulation_gloss = np.clip(
            water_accumulation
            * water_mask
            * light_response
            * (0.34 + 1.10 * gloss_strength)
            * (0.58 + 0.84 * beam_strength),
            0.0,
            1.0,
        )
        gloss = np.maximum(gloss, accumulation_gloss)
        gloss = cv2.GaussianBlur(gloss, (0, 0), sigmaX=0.45 + 2.0 * gloss_strength, sigmaY=0.38 + 1.45 * gloss_strength)
        ridge_gloss = np.clip(
            (contour_edge * 0.72 + grad_energy * 0.46 + lower_symbol_edge * 0.55)
            * water_mask
            * contour_response
            * (0.24 + 0.92 * gloss_strength),
            0.0,
            1.0,
        )
        lower_pool_gloss = np.clip(
            (lower_edge_pool * 0.84 + bottom_edge_pool * 0.42 + np.maximum(left_corner_pool, right_corner_pool) * 0.58)
            * water_mask
            * (0.16 + 0.74 * gloss_strength)
            * (0.52 + 0.68 * beam_strength),
            0.0,
            1.0,
        )
        ridge_gloss = np.maximum(ridge_gloss, lower_pool_gloss)
        ridge_gloss = cv2.GaussianBlur(ridge_gloss, (0, 0), sigmaX=0.42, sigmaY=0.42)

        tint = np.array([238.0, 246.0, 255.0], dtype="float32")
        damp = (water_mask * (0.018 + 0.060 * strength) + water_accumulation * strength * 0.030)[:, :, None]
        out = out * (1.0 - damp) + np.array([202.0, 211.0, 220.0], dtype="float32") * damp
        shine = np.clip(gloss * (1.0 + 0.34 * gloss_strength) + ridge_gloss, 0.0, 0.94)
        out = out + (tint - out) * shine[:, :, None] * (0.28 + 0.68 * gloss_strength)
        out = out + (255.0 - out) * np.clip((shine * water_mask * gloss_strength * 0.20)[:, :, None], 0.0, 0.22)

        result = np.clip(out, 0, 255).astype("uint8")
        return (result, water_mask) if return_mask else result
    except Exception:
        return (image, None) if return_mask else image


def _apply_traffic_headlight_effect(image, profile: AugmentationProfile, rng, dirt_mask=None, *, base_surface: dict | None = None) -> object:
    """Approximate several car headlights illuminating the plate from traffic."""
    if np is None or cv2 is None:
        return image

    try:
        def _profile_float(name: str, default: float) -> float:
            value = getattr(profile, name, default)
            if value is None:
                value = default
            return float(value)

        def _clamp01(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        global_strength = _clamp01(_profile_float("traffic_headlight_strength", 0.0))
        count = max(0, min(6, int(float(getattr(profile, "traffic_headlight_count", 1) if getattr(profile, "traffic_headlight_count", None) is not None else 1))))
        rain_strength = _clamp01(_profile_float("rain_strength", 0.0))
        edge_mist_strength = _clamp01(_profile_float("rain_edge_mist_strength", 0.0))
        tyndall_strength = _clamp01(_profile_float("tyndall_strength", 0.55))
        wet_mud_gloss = max(0.0, min(1.0, float(getattr(profile, "wet_mud_gloss_strength", 0.0) or 0.0)))

        height, width = image.shape[:2]
        if height < 8 or width < 8:
            return image
        plate_width, plate_height = _scene_plate_dimensions(profile, height, width)

        def _world_point_to_normalized(point: np.ndarray) -> tuple[float, float]:
            return (
                float(point[0]) / max(0.0001, plate_width) + 0.5,
                0.5 - float(point[1]) / max(0.0001, plate_height),
            )

        def _headlight_endpoint_norm(index: int, endpoint: str, fallback_norm: tuple[float, float]) -> tuple[float, float]:
            point = _scene_headlight_point(
                profile,
                index,
                endpoint,
                height,
                width,
                default_norm=fallback_norm,
                default_z=0.0 if endpoint == "target" else 0.95,
                scale=1.0 if endpoint == "target" else 1.55,
            )
            return _world_point_to_normalized(point)

        base_warmth = _clamp01(_profile_float("night_light_warmth", 0.35))
        first_warmth = _profile_float("traffic_headlight_1_warmth", -1.0)
        if first_warmth < 0.0:
            first_warmth = base_warmth
        def _headlight_rgb(index: int, warmth: float) -> tuple[float, float, float]:
            return _profile_headlight_rgb(profile, index, warmth)

        light_defs = [
            {
                "index": 1,
                "strength": global_strength,
                "warmth": _clamp01(first_warmth),
                "rgb": _headlight_rgb(1, _clamp01(first_warmth)),
                "cone": max(0.02, min(2.5, _profile_float("traffic_headlight_1_cone", 0.45))),
                "source_radius": max(0.0, min(2.5, _profile_float("traffic_headlight_1_source_radius", 0.08))),
                "source": (
                    _profile_float("traffic_headlight_source_x", -1.0),
                    _profile_float("traffic_headlight_source_y", -1.0),
                ),
                "target": (
                    _profile_float("traffic_headlight_target_x", -1.0),
                    _profile_float("traffic_headlight_target_y", -1.0),
                ),
                "default_source": (0.14, 0.91),
                "default_target": (0.42, 0.44),
            },
            {
                "index": 2,
                "strength": _clamp01(_profile_float("traffic_headlight_2_strength", 0.0)),
                "warmth": _clamp01(_profile_float("traffic_headlight_2_warmth", 0.35)),
                "rgb": _headlight_rgb(2, _clamp01(_profile_float("traffic_headlight_2_warmth", 0.35))),
                "cone": max(0.02, min(2.5, _profile_float("traffic_headlight_2_cone", 0.45))),
                "source_radius": max(0.0, min(2.5, _profile_float("traffic_headlight_2_source_radius", 0.08))),
                "source": (
                    _profile_float("traffic_headlight_2_source_x", -1.0),
                    _profile_float("traffic_headlight_2_source_y", -1.0),
                ),
                "target": (
                    _profile_float("traffic_headlight_2_target_x", -1.0),
                    _profile_float("traffic_headlight_2_target_y", -1.0),
                ),
                "default_source": (0.86, 0.90),
                "default_target": (0.58, 0.48),
            },
            {
                "index": 3,
                "strength": _clamp01(_profile_float("traffic_headlight_3_strength", 0.0)),
                "warmth": _clamp01(_profile_float("traffic_headlight_3_warmth", 0.35)),
                "rgb": _headlight_rgb(3, _clamp01(_profile_float("traffic_headlight_3_warmth", 0.35))),
                "cone": max(0.02, min(2.5, _profile_float("traffic_headlight_3_cone", 0.45))),
                "source_radius": max(0.0, min(2.5, _profile_float("traffic_headlight_3_source_radius", 0.08))),
                "source": (
                    _profile_float("traffic_headlight_3_source_x", -1.0),
                    _profile_float("traffic_headlight_3_source_y", -1.0),
                ),
                "target": (
                    _profile_float("traffic_headlight_3_target_x", -1.0),
                    _profile_float("traffic_headlight_3_target_y", -1.0),
                ),
                "default_source": (0.50, 0.98),
                "default_target": (0.50, 0.36),
            },
        ]
        manual_lights = []
        for item in light_defs:
            local_strength = float(item["strength"])
            if local_strength <= 0.001:
                continue
            try:
                index = int(item.get("index", 1) or 1)
                src_x, src_y = _headlight_endpoint_norm(index, "source", item["default_source"])
                tgt_x, tgt_y = _headlight_endpoint_norm(index, "target", item["default_target"])
            except Exception:
                src_x, src_y = item["source"]
                tgt_x, tgt_y = item["target"]
                if not (0.0 <= src_x <= 1.0 and 0.0 <= src_y <= 1.0):
                    src_x, src_y = item["default_source"]
                if not (0.0 <= tgt_x <= 1.0 and 0.0 <= tgt_y <= 1.0):
                    tgt_x, tgt_y = item["default_target"]
            tgt_x = max(0.0, min(1.0, float(tgt_x)))
            tgt_y = max(0.0, min(1.0, float(tgt_y)))
            manual_lights.append(
                (
                    np.array([src_x, src_y], dtype="float32"),
                    np.array([tgt_x, tgt_y], dtype="float32"),
                    local_strength,
                    float(item["warmth"]),
                    tuple(item["rgb"]),
                    float(item["cone"]),
                    float(item["source_radius"]),
                    True,
                )
            )
        if not manual_lights and (global_strength <= 0.001 or count <= 0):
            return image

        yy, xx = np.mgrid[0:height, 0:width].astype("float32")
        nx = xx / max(1.0, float(width - 1))
        ny = yy / max(1.0, float(height - 1))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        normal_strength = max(0.0, min(1.0, float(getattr(profile, "light_normal_strength", 0.0) or 0.0)))
        relief_strength = max(0.0, min(3.0, float(getattr(profile, "dark_relief_strength", 0.0) or 0.0)))

        mud = None
        mud_edge = None
        if dirt_mask is not None:
            mud = np.asarray(dirt_mask, dtype="float32")
            if mud.shape[:2] != (height, width):
                mud = cv2.resize(mud, (width, height), interpolation=cv2.INTER_LINEAR)
            mud = cv2.GaussianBlur(np.clip(mud, 0.0, 1.0), (0, 0), sigmaX=0.55, sigmaY=0.55)
            mud_grad_x = cv2.Sobel(mud, cv2.CV_32F, 1, 0, ksize=3)
            mud_grad_y = cv2.Sobel(mud, cv2.CV_32F, 0, 1, ksize=3)
            mud_edge = cv2.GaussianBlur(np.clip(np.abs(mud_grad_x) + np.abs(mud_grad_y), 0.0, 1.0), (0, 0), sigmaX=0.45, sigmaY=0.45)

        sign_height = _surface_float_mask(base_surface, "symbol_height", width, height)
        if sign_height is None:
            sign_height = _build_symbol_contour_height_map(image, _contour_detection_sensitivity(profile))
        if sign_height is None:
            sign_height = np.clip((0.66 - gray) / 0.52, 0.0, 1.0)
        sign_height = np.clip(sign_height, 0.0, 1.0) ** 1.18
        contour_profile_height = _surface_float_mask(base_surface, "contour_profile", width, height)
        contour_profile_normal_x = _surface_float_mask(base_surface, "contour_profile_normal_x", width, height, signed=True)
        contour_profile_normal_y = _surface_float_mask(base_surface, "contour_profile_normal_y", width, height, signed=True)
        contour_profile_normal_energy = _surface_float_mask(base_surface, "contour_profile_normal_energy", width, height)
        if (
            contour_profile_height is None
            or contour_profile_normal_x is None
            or contour_profile_normal_y is None
            or contour_profile_normal_energy is None
        ):
            contour_profile = _build_contour_profile_surface(sign_height, profile)
            contour_profile_height = contour_profile.get("height") if contour_profile else None
            contour_profile_normal_x = contour_profile.get("normal_x") if contour_profile else None
            contour_profile_normal_y = contour_profile.get("normal_y") if contour_profile else None
            contour_profile_normal_energy = contour_profile.get("normal_energy") if contour_profile else None
        height_map = sign_height * (0.10 + 1.18 * relief_strength)
        if contour_profile_height is not None:
            height_map = np.clip(
                height_map * 0.42 + contour_profile_height * (0.22 + 0.90 * relief_strength),
                0.0,
                1.0,
            )
        if mud is not None:
            height_map = np.clip(height_map + mud * (0.72 + 0.82 * wet_mud_gloss), 0.0, 1.0)
        height_map = cv2.GaussianBlur(height_map.astype("float32"), (0, 0), sigmaX=0.46, sigmaY=0.46)
        height_grad_x = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
        height_grad_y = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)
        grad_abs = np.abs(height_grad_x) + np.abs(height_grad_y)
        grad_scale = float(np.percentile(grad_abs, 99.2) or 0.0)
        if grad_scale > 0.0001:
            height_grad_x = height_grad_x / grad_scale
            height_grad_y = height_grad_y / grad_scale
            grad_abs = np.clip(grad_abs / grad_scale, 0.0, 1.0)
        else:
            grad_abs = np.clip(grad_abs, 0.0, 1.0)
        if (
            isinstance(contour_profile_normal_x, np.ndarray)
            and isinstance(contour_profile_normal_y, np.ndarray)
            and contour_profile_normal_x.shape == (height, width)
            and contour_profile_normal_y.shape == (height, width)
        ):
            profile_weight = np.clip(contour_profile_height if contour_profile_height is not None else 0.0, 0.0, 1.0)
            height_grad_x = np.clip(
                height_grad_x * (1.0 - profile_weight * 0.68) + contour_profile_normal_x * profile_weight * 0.68,
                -1.0,
                1.0,
            )
            height_grad_y = np.clip(
                height_grad_y * (1.0 - profile_weight * 0.68) + contour_profile_normal_y * profile_weight * 0.68,
                -1.0,
                1.0,
            )
            if isinstance(contour_profile_normal_energy, np.ndarray) and contour_profile_normal_energy.shape == (height, width):
                grad_abs = np.clip(np.maximum(grad_abs, contour_profile_normal_energy * profile_weight), 0.0, 1.0)
        symbol_gate = cv2.GaussianBlur(np.clip(sign_height, 0.0, 1.0).astype("float32"), (0, 0), sigmaX=0.32, sigmaY=0.32)
        if contour_profile_height is not None:
            symbol_gate = np.clip(np.maximum(symbol_gate, contour_profile_height * 0.72), 0.0, 1.0)
        relief_field = np.zeros((height, width), dtype="float32")
        shadow_field = np.zeros((height, width), dtype="float32")

        procedural_lights = []
        if not manual_lights:
            base_sources = [
                (-0.28, 1.18),
                (1.28, 1.14),
                (0.50, 1.36),
                (-0.18, 0.38),
                (1.18, 0.34),
                (0.50, -0.18),
            ]
            for index in range(count):
                src_x, src_y = base_sources[index % len(base_sources)]
                src = np.array([src_x + _rng_float(rng, -0.06, 0.06), src_y + _rng_float(rng, -0.05, 0.05)], dtype="float32")
                target = np.array([0.50 + _rng_float(rng, -0.08, 0.08), 0.46 + _rng_float(rng, -0.08, 0.07)], dtype="float32")
                procedural_lights.append((src, target, global_strength, base_warmth, _headlight_rgb(1, base_warmth), 0.45, 0.08, False))
        active_lights = manual_lights or procedural_lights
        strength = max((float(item[2]) for item in active_lights), default=0.0)
        if strength <= 0.001:
            return image
        bounce_light_x, bounce_light_y, _fallback_bounce_strength = _scene_or_fallback_light_direction(profile, height, width)
        bounce_direction_x = 0.0
        bounce_direction_y = 0.0
        bounce_direction_weight = 0.0
        light_field = np.zeros((height, width), dtype="float32")
        hotspot_field = np.zeros((height, width), dtype="float32")
        color_weight = np.zeros((height, width), dtype="float32")
        color_accum = np.zeros((height, width, 3), dtype="float32")
        cold = np.array([255.0, 240.0, 214.0], dtype="float32")
        warm = np.array([180.0, 225.0, 255.0], dtype="float32")
        scene_lights = _combined_scene_headlight_fields(profile, height, width)
        scene_geometry_drives_manual_lights = bool(scene_lights) and bool(manual_lights)

        for src, target_center, local_strength, local_warmth, local_rgb, local_cone, local_source_radius, stable in active_lights:
            delta = target_center - src
            norm = float(np.linalg.norm(delta) or 0.0)
            source_width = max(0.0, min(2.5, float(local_source_radius)))
            target_width = max(0.02, min(2.5, float(local_cone)))
            if not stable:
                target_width *= _rng_float(rng, 0.86, 1.18)
                source_width *= _rng_float(rng, 0.86, 1.14)
            local_color = np.array(
                [
                    float(local_rgb[2]) * 255.0,
                    float(local_rgb[1]) * 255.0,
                    float(local_rgb[0]) * 255.0,
                ],
                dtype="float32",
            )
            if stable and scene_geometry_drives_manual_lights:
                # Explicit R1/R2/R3 lights are now driven by the 3D scene.
                # The old 2D beam ignored source Z and could visibly start
                # from a projected point instead of the actual reflector.
                continue
            if norm <= 0.012:
                center = (src + target_center) * 0.5
                rel_x = nx - float(center[0])
                rel_y = (ny - float(center[1])) * norm_y_metric
                radial = np.sqrt(rel_x * rel_x + rel_y * rel_y)
                outer_radius = max(0.008, max(source_width, target_width))
                inner_radius = max(0.006, min(source_width, target_width))
                beam = np.exp(-((radial / outer_radius) ** 2)).astype("float32")
                core = np.exp(-((radial / inner_radius) ** 2)).astype("float32") * 0.42
                beam = np.maximum(beam, core)
                beam *= local_strength * (1.0 if stable else _rng_float(rng, 0.58, 1.18))
                light_field = np.maximum(light_field, beam)

                hot_sigma = max(0.012, outer_radius * (0.26 + 0.22 * local_strength))
                hot = np.exp(-((radial / hot_sigma) ** 2)).astype("float32")
                hot *= local_strength * (0.62 if stable else _rng_float(rng, 0.42, 1.0))
                hotspot_field = np.maximum(hotspot_field, hot)
                local_source = np.clip(beam * (0.56 + 0.70 * local_strength) + hot * (0.32 + 0.90 * local_strength), 0.0, 1.0)
                color_accum += local_source[:, :, None] * local_color
                color_weight += local_source
                continue

            direction = delta / norm
            bounce_direction_x += float(direction[0]) * max(0.0, float(local_strength))
            bounce_direction_y += float(direction[1]) * max(0.0, float(local_strength))
            bounce_direction_weight += max(0.0, float(local_strength))
            perp = np.array([-float(direction[1]), float(direction[0])], dtype="float32")

            rel_x = nx - float(src[0])
            rel_y = ny - float(src[1])
            along = rel_x * float(direction[0]) + rel_y * float(direction[1])
            side = rel_x * float(perp[0]) + rel_y * float(perp[1])
            beam_length = max(0.16, norm * (0.72 if stable else _rng_float(rng, 0.58, 0.92)) + target_width * 1.20)
            center_along = norm * (0.50 if stable else _rng_float(rng, 0.45, 0.62))
            progress = np.clip(along / max(0.001, norm), 0.0, 1.30)
            width_progress = np.clip(progress, 0.0, 1.0)
            local_width = np.maximum(0.006, source_width * (1.0 - width_progress) + target_width * width_progress)
            beam = np.exp(-((side / local_width) ** 2 + ((along - center_along) / beam_length) ** 2)).astype("float32")
            near_fade = max(0.018, norm * 0.055)
            beam *= np.clip((along + near_fade) / near_fade, 0.0, 1.0)
            far_fade = max(0.028, target_width + norm * 0.12)
            beam *= np.clip((norm + far_fade - along) / far_fade, 0.0, 1.0)
            beam *= local_strength * (1.0 if stable else _rng_float(rng, 0.58, 1.18))
            light_field = np.maximum(light_field, beam)
            local_relief = (height_grad_x * float(direction[0])) + (height_grad_y * float(direction[1]))
            relief_field = np.maximum(relief_field, np.clip(local_relief, 0.0, 1.0) * beam)
            shadow_field = np.maximum(shadow_field, np.clip(-local_relief, 0.0, 1.0) * beam)

            hot_x = float(src[0]) + float(direction[0]) * norm * (0.78 if stable else _rng_float(rng, 0.62, 0.92))
            hot_y = float(src[1]) + float(direction[1]) * norm * (0.78 if stable else _rng_float(rng, 0.62, 0.92))
            hot_sigma_x = (0.055 if stable else _rng_float(rng, 0.035, 0.090)) * (1.0 + 0.55 * local_strength)
            hot_sigma_y = (0.036 if stable else _rng_float(rng, 0.020, 0.060)) * (1.0 + 0.40 * local_strength)
            hot = np.exp(-(((nx - hot_x) / hot_sigma_x) ** 2 + ((ny - hot_y) / hot_sigma_y) ** 2)).astype("float32")
            hot *= local_strength * (1.0 if stable else _rng_float(rng, 0.42, 1.0))
            hotspot_field = np.maximum(hotspot_field, hot)
            local_source = np.clip(beam * (0.56 + 0.70 * local_strength) + hot * (0.32 + 0.90 * local_strength), 0.0, 1.0)
            color_accum += local_source[:, :, None] * local_color
            color_weight += local_source

        if bounce_direction_weight > 0.0001:
            bounce_direction_len = math.hypot(bounce_direction_x, bounce_direction_y)
            if bounce_direction_len > 0.0001:
                bounce_light_x = bounce_direction_x / bounce_direction_len
                bounce_light_y = bounce_direction_y / bounce_direction_len

        light_field = cv2.GaussianBlur(np.clip(light_field, 0.0, 1.0), (0, 0), sigmaX=1.2 + 3.0 * strength, sigmaY=1.2 + 3.0 * strength)
        hotspot_field = cv2.GaussianBlur(np.clip(hotspot_field, 0.0, 1.0), (0, 0), sigmaX=0.8 + 1.6 * strength, sigmaY=0.8 + 1.6 * strength)
        color_weight = cv2.GaussianBlur(np.clip(color_weight, 0.0, 1.0), (0, 0), sigmaX=1.0 + 2.2 * strength, sigmaY=1.0 + 2.2 * strength)
        for channel in range(3):
            color_accum[:, :, channel] = cv2.GaussianBlur(color_accum[:, :, channel], (0, 0), sigmaX=1.0 + 2.2 * strength, sigmaY=1.0 + 2.2 * strength)
        field = np.clip(light_field * (0.56 + 0.70 * strength) + hotspot_field * (0.32 + 0.90 * strength), 0.0, 1.0)

        scene_specular = None
        if scene_lights:
            scene_field = cv2.GaussianBlur(
                np.clip(np.asarray(scene_lights.get("field"), dtype="float32"), 0.0, 1.0),
                (0, 0),
                sigmaX=0.8 + 2.0 * strength,
                sigmaY=0.8 + 2.0 * strength,
            )
            scene_hotspot = cv2.GaussianBlur(
                np.clip(np.asarray(scene_lights.get("hotspot"), dtype="float32"), 0.0, 1.0),
                (0, 0),
                sigmaX=0.55 + 1.20 * strength,
                sigmaY=0.55 + 1.20 * strength,
            )
            light_field = np.maximum(light_field, scene_field)
            hotspot_field = np.maximum(hotspot_field, scene_hotspot)
            try:
                scene_specular = cv2.GaussianBlur(
                    np.clip(np.asarray(scene_lights.get("specular"), dtype="float32"), 0.0, 1.0),
                    (0, 0),
                    sigmaX=0.45 + 0.95 * strength,
                    sigmaY=0.45 + 0.95 * strength,
                )
                if scene_specular.shape == (height, width):
                    hotspot_field = np.maximum(hotspot_field, scene_specular * (0.60 + 0.70 * strength))
                else:
                    scene_specular = None
            except Exception:
                scene_specular = None
            field = np.clip(
                np.maximum(field, scene_field * (0.64 + 0.52 * strength) + scene_hotspot * (0.22 + 0.58 * strength)),
                0.0,
                1.0,
            )
            if manual_lights:
                scene_color_source = np.clip(scene_field * 0.78 + scene_hotspot * 0.54, 0.0, 1.0)
                if scene_specular is not None:
                    scene_color_source = np.maximum(scene_color_source, np.clip(scene_specular * 0.62, 0.0, 1.0))
                try:
                    raw_scene_color_accum = np.asarray(scene_lights.get("color_accum"), dtype="float32")
                    raw_scene_color_weight = np.asarray(
                        scene_lights.get("color_weight_raw", scene_lights.get("color_weight")),
                        dtype="float32",
                    )
                    if raw_scene_color_accum.shape != (height, width, 3) or raw_scene_color_weight.shape != (height, width):
                        raise ValueError("scene color map shape mismatch")
                    scene_color_sigma = 0.8 + 2.0 * strength
                    scene_color_weight = cv2.GaussianBlur(
                        np.maximum(raw_scene_color_weight, 0.0),
                        (0, 0),
                        sigmaX=scene_color_sigma,
                        sigmaY=scene_color_sigma,
                    )
                    scene_color_accum = raw_scene_color_accum.copy()
                    for channel in range(3):
                        scene_color_accum[:, :, channel] = cv2.GaussianBlur(
                            scene_color_accum[:, :, channel],
                            (0, 0),
                            sigmaX=scene_color_sigma,
                            sigmaY=scene_color_sigma,
                        )
                    scene_color = scene_color_accum / np.maximum(scene_color_weight[:, :, None], 0.0001)
                    valid_color = scene_color_weight > 0.001
                    scene_color_source = np.where(valid_color, scene_color_source, 0.0)
                    color_accum += scene_color_source[:, :, None] * np.clip(scene_color, 0.0, 255.0)
                    color_weight += scene_color_source
                except Exception:
                    scene_color = np.zeros(3, dtype="float32")
                    scene_color_weight = 0.0
                    for _src, _target_center, local_strength, _local_warmth, local_rgb, _local_cone, _local_source_radius, _stable in manual_lights:
                        local_weight = max(0.0, float(local_strength))
                        scene_color += np.array(
                            [
                                float(local_rgb[2]) * 255.0,
                                float(local_rgb[1]) * 255.0,
                                float(local_rgb[0]) * 255.0,
                            ],
                            dtype="float32",
                        ) * local_weight
                        scene_color_weight += local_weight
                    if scene_color_weight > 0.0001:
                        scene_color /= scene_color_weight
                        color_accum += scene_color_source[:, :, None] * scene_color
                        color_weight += scene_color_source
            scene_relief = (
                height_grad_x * np.asarray(scene_lights.get("ray_x"), dtype="float32")
                + height_grad_y * np.asarray(scene_lights.get("ray_y"), dtype="float32")
            )
            relief_field = np.maximum(relief_field, np.clip(scene_relief, 0.0, 1.0) * scene_field)
            shadow_field = np.maximum(shadow_field, np.clip(-scene_relief, 0.0, 1.0) * scene_field)

        bright_mask = np.clip((gray - 0.28) / 0.62, 0.0, 1.0)
        ink_mask = cv2.GaussianBlur(np.clip(symbol_gate, 0.0, 1.0), (0, 0), sigmaX=0.34, sigmaY=0.34)
        reflective_surface = cv2.GaussianBlur(np.clip(1.0 - ink_mask * 0.92, 0.0, 1.0), (0, 0), sigmaX=0.72, sigmaY=0.72)
        reflective_surface = np.clip(
            reflective_surface * (0.76 + 0.24 * np.clip((gray + 0.18) / 0.88, 0.0, 1.0)),
            0.0,
            1.0,
        )
        retro_mask = cv2.GaussianBlur(
            np.clip(bright_mask * 0.38 + reflective_surface * 0.76, 0.0, 1.0),
            (0, 0),
            sigmaX=0.7,
            sigmaY=0.7,
        )
        fallback_color = cold * (1.0 - base_warmth) + warm * base_warmth
        light_color = color_accum / np.maximum(color_weight[:, :, None], 0.0001)
        light_color = np.where(color_weight[:, :, None] > 0.001, light_color, fallback_color)

        out = image.astype("float32")
        night_strength = _clamp01(_profile_float("night_strength", 0.0))
        night_response = 0.35 + 0.65 * night_strength
        overglare = np.clip((strength - 0.86) / 0.14, 0.0, 1.0)
        beam_exposure = np.clip(
            (light_field * 0.78 + hotspot_field * 0.42)[:, :, None]
            * (0.24 + 0.78 * strength)
            * night_response,
            0.0,
            0.92,
        )
        if float(np.max(beam_exposure) or 0.0) > 0.001:
            # In a night scene the reflector must locally recover exposure,
            # otherwise the later light pass only tints an already crushed
            # image and the result looks like a weak filter.
            material_response = np.clip(0.14 + reflective_surface * (1.08 - 0.26 * overglare), 0.0, 1.0)
            out = out + (light_color - out) * beam_exposure * material_response[:, :, None]
            retro_exposure = np.clip(
                (field * retro_mask * reflective_surface * (0.30 + 1.05 * strength) * night_response)[:, :, None],
                0.0,
                0.84,
            )
            out = out + (255.0 - out) * retro_exposure * (0.34 + 0.44 * strength)
            if scene_specular is not None:
                scene_specular_exposure = np.clip(
                    scene_specular
                    * reflective_surface
                    * night_response
                    * (0.26 + 0.92 * strength + 0.30 * wet_mud_gloss),
                    0.0,
                    0.78,
                )
                out = out + (light_color - out) * scene_specular_exposure[:, :, None]
                out = out + (255.0 - out) * cv2.GaussianBlur(scene_specular_exposure, (0, 0), sigmaX=0.75, sigmaY=0.75)[:, :, None] * 0.34
            ink_contrast = np.clip(
                field
                * ink_mask
                * night_response
                * (0.18 + 0.42 * (1.0 - overglare)),
                0.0,
                0.62,
            )
            out = out * (1.0 - ink_contrast[:, :, None])
            rim_light = np.clip(grad_abs * field * reflective_surface * (0.04 + 0.18 * normal_strength), 0.0, 0.34)
            out = out + light_color * rim_light[:, :, None]

        medium_strength = max(rain_strength, edge_mist_strength * 0.72)
        tyndall_gain = medium_strength * strength * tyndall_strength
        if tyndall_gain > 0.001:
            min_dim = float(max(1, min(width, height)))
            aerosol = np.clip(light_field * 0.72 + hotspot_field * 0.28, 0.0, 1.0)
            aerosol = cv2.GaussianBlur(
                aerosol,
                (0, 0),
                sigmaX=max(1.0, 1.6 + min_dim * (0.006 + 0.018 * medium_strength)),
                sigmaY=max(0.8, 1.1 + min_dim * (0.004 + 0.014 * medium_strength)),
            )
            noise_h = max(6, min(96, height // 28 or 6))
            noise_w = max(6, min(96, width // 28 or 6))
            noise_rng = np.random.default_rng(_rng_int(rng, 1, 2_147_483_646))
            particles = noise_rng.random((noise_h, noise_w)).astype("float32")
            particles = cv2.resize(particles, (width, height), interpolation=cv2.INTER_LINEAR)
            particles = cv2.GaussianBlur(
                particles,
                (0, 0),
                sigmaX=0.8 + 2.2 * medium_strength,
                sigmaY=0.8 + 2.2 * medium_strength,
            )
            particle_gate = np.clip(0.72 + particles * (0.24 + 0.22 * medium_strength), 0.0, 1.18)
            tyndall = np.clip(
                aerosol
                * particle_gate
                * tyndall_gain
                * (0.28 + 0.46 * medium_strength),
                0.0,
                0.48,
            )
            tyndall = cv2.GaussianBlur(
                tyndall,
                (0, 0),
                sigmaX=0.9 + 2.4 * tyndall_gain,
                sigmaY=0.7 + 1.9 * tyndall_gain,
            )
            tyndall_color = np.clip(
                light_color * 0.82 + np.array([245.0, 248.0, 255.0], dtype="float32") * 0.18,
                0.0,
                255.0,
            )
            veil = tyndall[:, :, None]
            contrast_loss = veil * (0.05 + 0.12 * tyndall_gain)
            out = out * (1.0 - contrast_loss) + np.array([186.0, 193.0, 205.0], dtype="float32") * contrast_loss
            out = out + (tyndall_color - out) * veil * (0.28 + 0.42 * tyndall_strength)
        ambient_material = np.clip(0.10 + reflective_surface * 0.90 + ink_mask * overglare * 0.36, 0.0, 1.0)
        ambient_gain = (
            field[:, :, None]
            * (0.10 + 0.22 * night_response + 0.18 * (1.0 - retro_mask[:, :, None]))
            * ambient_material[:, :, None]
        )
        retro_gain = field[:, :, None] * retro_mask[:, :, None] * reflective_surface[:, :, None] * (0.30 + 0.58 * normal_strength + 0.36 * night_strength)
        out = out + (light_color - out) * np.clip(ambient_gain + retro_gain, 0.0, 0.86)
        bloom = cv2.GaussianBlur(np.clip(field * retro_mask, 0.0, 1.0), (0, 0), sigmaX=2.0 + 5.5 * strength, sigmaY=1.2 + 3.5 * strength)
        out = out + (255.0 - out) * np.clip(bloom[:, :, None] * strength * (0.30 + 0.18 * night_strength), 0.0, 0.54)
        if night_strength > 0.001:
            final_ink_contrast = np.clip(
                field
                * ink_mask
                * night_response
                * (0.22 + 0.52 * (1.0 - overglare)),
                0.0,
                0.72,
            )
            out = out * (1.0 - final_ink_contrast[:, :, None])
        relief_gain = strength * (0.10 + 0.42 * normal_strength + 0.96 * relief_strength + 0.36 * wet_mud_gloss)
        if relief_gain > 0.001:
            relief_field = cv2.GaussianBlur(np.clip(relief_field, 0.0, 1.0), (0, 0), sigmaX=0.32 + 0.55 * relief_gain, sigmaY=0.32 + 0.55 * relief_gain)
            shadow_field = cv2.GaussianBlur(np.clip(shadow_field, 0.0, 1.0), (0, 0), sigmaX=0.55 + 0.70 * relief_gain, sigmaY=0.55 + 0.70 * relief_gain)
            edge_gate = np.clip((symbol_gate * 0.34) + (height_map * 0.26) + (grad_abs * 1.18), 0.0, 1.0)
            lit = np.clip(relief_field * edge_gate * (0.78 + 1.05 * field), 0.0, 1.0)
            bounce_depth = max(1, min(4, int(float(getattr(profile, "relief_bounce_depth", 1) or 1))))
            bounce_strength = max(0.0, min(1.0, float(getattr(profile, "relief_bounce_strength", 0.28) if getattr(profile, "relief_bounce_strength", None) is not None else 0.28)))
            if bounce_depth > 1 and bounce_strength > 0.001:
                bounce_field = _build_relief_bounce_light(
                    np.clip(relief_field * edge_gate, 0.0, 1.0),
                    edge_gate,
                    bounce_light_x,
                    bounce_light_y,
                    bounce_depth,
                    bounce_strength,
                )
                if bounce_field is not None:
                    lit = np.clip(lit + bounce_field * (0.22 + 0.64 * field), 0.0, 1.0)
            shaded = np.clip(shadow_field * edge_gate * (0.64 + 0.86 * field), 0.0, 1.0)
            out = out + light_color * lit[:, :, None] * (0.14 + 0.68 * relief_gain)
            out = out - out * shaded[:, :, None] * (0.12 + 0.52 * relief_gain)

        if wet_mud_gloss > 0.001 and mud is not None and mud_edge is not None:
            wetness = max(0.0, min(1.0, float(getattr(profile, "dirt_flow_humidity", 0.45) if getattr(profile, "dirt_flow_humidity", None) is not None else 0.45)))
            gloss_source = np.clip((field * 0.72 + hotspot_field * 0.95) * mud, 0.0, 1.0)
            gloss_shape = np.clip(0.36 + 1.18 * mud_edge + 0.72 * relief_field, 0.0, 2.2)
            gloss = np.clip(
                gloss_source
                * gloss_shape
                * (0.48 + 0.82 * wetness)
                * wet_mud_gloss
                * (0.75 + 0.95 * wet_mud_gloss),
                0.0,
                1.0,
            )
            gloss = cv2.GaussianBlur(gloss, (0, 0), sigmaX=0.28 + 1.15 * wet_mud_gloss, sigmaY=0.28 + 0.82 * wet_mud_gloss)
            color_gloss = np.clip(gloss[:, :, None] * (0.86 + 0.72 * wet_mud_gloss), 0.0, 0.86)
            white_gloss = np.clip((gloss * (0.42 + 0.78 * mud_edge + 0.36 * hotspot_field))[:, :, None] * wet_mud_gloss, 0.0, 0.62)
            out = out + (light_color - out) * color_gloss
            out = out + (255.0 - out) * white_gloss

        return np.clip(out, 0, 255).astype("uint8")
    except Exception:
        return image


def _has_active_traffic_headlight(profile: AugmentationProfile) -> bool:
    try:
        return any(
            max(0.0, min(1.0, float(getattr(profile, name, 0.0) or 0.0))) > 0.001
            for name in (
                "traffic_headlight_strength",
                "traffic_headlight_2_strength",
                "traffic_headlight_3_strength",
            )
        )
    except Exception:
        return False

def _apply_camera_glare_effect(image, profile: AugmentationProfile, rng) -> object:
    if np is None or cv2 is None or profile.light_normal_strength <= 0:
        return image

    try:
        normal = max(0.0, min(1.0, float(profile.light_normal_strength or 0.0)))
        light = max(
            max(0.0, min(1.0, float(getattr(profile, "night_light_strength", 0.0) or 0.0))),
            max(0.0, min(1.0, float(getattr(profile, "traffic_headlight_strength", 0.0) or 0.0))),
            max(0.0, min(1.0, float(getattr(profile, "traffic_headlight_2_strength", 0.0) or 0.0))),
            max(0.0, min(1.0, float(getattr(profile, "traffic_headlight_3_strength", 0.0) or 0.0))),
        )
        rain = max(0.0, min(1.0, float(profile.rain_strength or 0.0)))
        if light <= 0.001 and rain <= 0.001:
            return image
        strength = normal * (0.28 + 0.72 * light) * (0.70 + 0.30 * rain)
        if strength <= 0.002:
            return image

        height, width = image.shape[:2]
        if height < 12 or width < 12:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        bright_mask = np.clip((gray - 0.44) / 0.56, 0.0, 1.0)
        broad = cv2.GaussianBlur(
            bright_mask,
            (0, 0),
            sigmaX=max(2.0, min(width, height) * (0.035 + 0.085 * strength)),
            sigmaY=max(2.0, min(width, height) * (0.035 + 0.085 * strength)),
        )
        fine = cv2.GaussianBlur(bright_mask, (0, 0), sigmaX=1.2 + 2.5 * strength, sigmaY=1.2 + 2.5 * strength)
        glare = np.clip((broad * (0.62 + 0.28 * rain)) + (fine * 0.22), 0.0, 1.0)

        light_x, light_y, scene_beam_strength = _scene_or_fallback_light_direction(profile, height, width)
        offset_x = light_x * width * 0.10 * normal
        offset_y = light_y * height * 0.10 * normal
        center_x = width * 0.5 + offset_x
        center_y = height * 0.5 + offset_y
        yy, xx = np.mgrid[0:height, 0:width].astype("float32")
        radial = ((xx - center_x) ** 2) / ((width * (0.36 + 0.24 * normal)) ** 2)
        radial += ((yy - center_y) ** 2) / ((height * (0.30 + 0.20 * normal)) ** 2)
        veil = np.exp(-radial).astype("float32") * (0.18 + 0.38 * strength)
        glare = np.clip(glare * (0.26 + 0.74 * strength + 0.18 * scene_beam_strength) + veil, 0.0, 1.0)

        tint = np.array([238.0, 245.0, 255.0], dtype="float32")
        out = image.astype("float32")
        contrast_loss = glare[:, :, None] * (0.05 + 0.16 * strength)
        out = out * (1.0 - contrast_loss) + np.array([190.0, 198.0, 210.0], dtype="float32") * contrast_loss
        out = out + (tint - out) * glare[:, :, None] * (0.28 + 0.42 * strength)
        return np.clip(out, 0, 255).astype("uint8")
    except Exception:
        return image


def _apply_dirt_streak_effect(image, profile: AugmentationProfile, rng) -> object:
    if np is None or cv2 is None or profile.dirt_streak_strength <= 0:
        return image

    try:
        strength = max(0.0, min(1.0, float(profile.dirt_streak_strength)))
        height, width = image.shape[:2]
        if height < 12 or width < 12:
            return image

        overlay = image.astype("float32")
        mask = np.zeros((height, width), dtype="float32")
        streaks = int(max(1, (width / 95.0) * (2.0 + 8.0 * strength)))
        for _ in range(streaks):
            x = _rng_int(rng, 0, max(1, width - 1))
            y = _rng_int(rng, 0, max(1, int(height * 0.55)))
            length = _rng_int(rng, int(height * (0.18 + 0.18 * strength)), int(height * (0.38 + 0.42 * strength)))
            thickness = _rng_int(rng, 1, max(2, int(2 + 5 * strength)))
            drift = _rng_int(rng, -max(2, int(width * 0.035)), max(2, int(width * 0.035)))
            mid_y = min(height - 1, y + int(length * 0.45))
            end_y = min(height - 1, y + length)
            points = np.array(
                [
                    [x, y],
                    [max(0, min(width - 1, x + int(drift * 0.35))), mid_y],
                    [max(0, min(width - 1, x + drift)), end_y],
                ],
                dtype=np.int32,
            )
            cv2.polylines(mask, [points], False, 1.0, thickness=thickness, lineType=cv2.LINE_AA)
            if _rng_int(rng, 0, 100) < int(35 + 35 * strength):
                drop_y = min(height - 1, y + _rng_int(rng, int(length * 0.08), max(int(length * 0.35), int(length * 0.08) + 1)))
                cv2.circle(mask, (x, drop_y), max(1, thickness), 0.65, -1, lineType=cv2.LINE_AA)

        kernel = max(3, int(3 + 8 * strength))
        if kernel % 2 == 0:
            kernel += 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
        max_value = float(mask.max() or 0.0)
        if max_value > 0:
            mask = mask / max_value

        dirt_color = np.array([42.0, 55.0, 75.0], dtype="float32")  # BGR: muted brown-gray.
        alpha = mask[:, :, None] * (0.10 + 0.38 * strength)
        overlay = overlay * (1.0 - alpha) + dirt_color * alpha
        return np.clip(overlay, 0, 255).astype("uint8")
    except Exception:
        return image


def _apply_physical_dirt_flow_effect(
    image,
    profile: AugmentationProfile,
    rng,
    *,
    return_mask: bool = False,
    base_surface: dict | None = None,
) -> object:
    if np is None or cv2 is None:
        return (image, None) if return_mask else image

    try:
        legacy_strength = max(0.0, min(1.0, float(profile.dirt_flow_strength or 0.0)))
        points = max(0, min(MAX_DIRT_FLOW_POINTS, int(profile.dirt_flow_points or 0)))
        if points <= 0 and legacy_strength <= 0:
            return (image, None) if return_mask else image
        if points <= 0:
            points = max(1, int(round(24 * (0.20 + 0.80 * legacy_strength))))

        mass_min = max(0.05, min(2.8, float(getattr(profile, "dirt_flow_mass_min", 0.18) or 0.18)))
        mass_max = max(0.05, min(2.8, float(getattr(profile, "dirt_flow_mass_max", 1.0) or 1.0)))
        if mass_min > mass_max:
            mass_min, mass_max = mass_max, mass_min
        splash_scale = max(0.0, min(1.0, float(profile.dirt_flow_splash_scale if profile.dirt_flow_splash_scale is not None else 0.45)))
        trail_length = max(0.0, min(1.0, float(profile.dirt_flow_trail_length if profile.dirt_flow_trail_length is not None else 0.55)))
        humidity = max(0.0, min(1.0, float(profile.dirt_flow_humidity)))
        legacy_stickiness = max(0.0, min(1.0, float(getattr(profile, "dirt_flow_stickiness", 0.45) if getattr(profile, "dirt_flow_stickiness", None) is not None else 0.45)))
        stickiness_min = max(0.0, min(1.0, float(getattr(profile, "dirt_flow_stickiness_min", legacy_stickiness) if getattr(profile, "dirt_flow_stickiness_min", None) is not None else legacy_stickiness)))
        stickiness_max = max(0.0, min(1.0, float(getattr(profile, "dirt_flow_stickiness_max", legacy_stickiness) if getattr(profile, "dirt_flow_stickiness_max", None) is not None else legacy_stickiness)))
        if stickiness_min > stickiness_max:
            stickiness_min, stickiness_max = stickiness_max, stickiness_min
        angle = math.radians(float(profile.dirt_flow_air_angle or 0.0))
        wind_strength = max(0.0, min(1.0, float(profile.dirt_flow_wind_strength or 0.0)))
        vehicle_speed = max(0.0, min(1.0, float(profile.vehicle_speed if profile.vehicle_speed is not None else 0.0)))
        rain_strength = max(0.0, min(1.0, float(profile.rain_strength or 0.0)))
        gravity_strength = 1.0
        opacity_min = max(0.0, min(1.0, float(profile.dirt_flow_opacity_min or 0.0)))
        opacity_max = max(0.0, min(1.0, float(profile.dirt_flow_opacity_max or 0.0)))
        if opacity_min > opacity_max:
            opacity_min, opacity_max = opacity_max, opacity_min

        height, width = image.shape[:2]
        if height < 12 or width < 12:
            return (image, None) if return_mask else image

        stop_on_dark_contour = bool(getattr(profile, "dirt_flow_stop_on_dark_contour", False))
        stop_mask = None
        stop_normal_x = None
        stop_normal_y = None
        if stop_on_dark_contour or trail_length > 0.001:
            try:
                kernel = np.ones((3, 3), dtype="uint8")
                stable_height = _surface_float_mask(base_surface, "symbol_height", width, height)
                stable_edge = _surface_float_mask(base_surface, "contour_curve", width, height)
                stable_normal_x = _surface_float_mask(base_surface, "normal_x", width, height, signed=True)
                stable_normal_y = _surface_float_mask(base_surface, "normal_y", width, height, signed=True)
                if stable_height is not None:
                    edge_seed = stable_edge if stable_edge is not None else _build_smooth_symbol_contour_edge_mask(stable_height)
                    if edge_seed is not None:
                        dark_neighborhood = cv2.dilate((stable_height > 0.08).astype("uint8"), kernel, iterations=1) > 0
                        contour_neighborhood = cv2.dilate((edge_seed > 0.035).astype("uint8"), kernel, iterations=1) > 0
                        stop_mask = cv2.dilate((dark_neighborhood & contour_neighborhood).astype("uint8"), kernel, iterations=1) > 0
                        if int(np.count_nonzero(stop_mask)) < max(6, int(width * height * 0.0005)):
                            stop_mask = cv2.dilate((edge_seed > 0.030).astype("uint8"), kernel, iterations=1) > 0
                        if stable_normal_x is not None and stable_normal_y is not None:
                            stop_normal_x = stable_normal_x
                            stop_normal_y = stable_normal_y
                if stop_mask is None:
                    source_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                    source_gray = source_gray.astype("uint8", copy=False)
                    blurred_gray = cv2.GaussianBlur(source_gray, (3, 3), 0)
                    dark_threshold = max(20.0, min(175.0, float(np.percentile(blurred_gray, 38)) * 0.96))
                    dark_region = blurred_gray <= dark_threshold
                    median_gray = float(np.median(blurred_gray) or 0.0)
                    lower = int(max(10.0, min(110.0, median_gray * 0.58)))
                    upper = int(max(float(lower + 24), min(220.0, median_gray * 1.28)))
                    contour_edges = cv2.Canny(blurred_gray, lower, upper)
                    if int(np.count_nonzero(contour_edges)) < max(8, int(width * height * 0.0008)):
                        contour_edges = cv2.Canny(blurred_gray, 18, 82)
                    dark_neighborhood = cv2.dilate(dark_region.astype("uint8"), kernel, iterations=1) > 0
                    contour_neighborhood = cv2.dilate((contour_edges > 0).astype("uint8"), kernel, iterations=1) > 0
                    stop_mask = cv2.dilate((dark_neighborhood & contour_neighborhood).astype("uint8"), kernel, iterations=1) > 0
                    if int(np.count_nonzero(stop_mask)) < max(6, int(width * height * 0.0005)):
                        stop_mask = cv2.dilate((contour_edges > 0).astype("uint8"), kernel, iterations=1) > 0
                    grad_x = cv2.Sobel(blurred_gray.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(blurred_gray.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)
                    grad_norm = np.sqrt(grad_x * grad_x + grad_y * grad_y) + 1e-6
                    stop_normal_x = grad_x / grad_norm
                    stop_normal_y = grad_y / grad_norm
            except Exception:
                stop_mask = None
                stop_normal_x = None
                stop_normal_y = None

        alpha_mask = np.zeros((height, width), dtype="float32")
        gravity = np.array([0.0, 1.0], dtype="float32") * (0.18 + 1.40 * gravity_strength)
        airflow = np.array([math.cos(angle) * (0.35 + 1.85 * wind_strength), 0.0], dtype="float32")
        initial_force = gravity * 0.35 + airflow * (0.65 + 0.45 * wind_strength)
        initial_norm = float(np.linalg.norm(initial_force) or 1.0)
        initial_force = initial_force / initial_norm
        gravity_norm = gravity / float(np.linalg.norm(gravity) or 1.0)
        flow_power = 1.0 + 0.80 * humidity + 0.85 * wind_strength

        def contour_hit_on_segment(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int] | None:
            if stop_mask is None:
                return None
            sample_count = max(2, int(math.ceil(max(abs(x1 - x0), abs(y1 - y0)))))
            sample_count = min(64, sample_count)
            for sample in range(1, sample_count + 1):
                ratio = sample / float(sample_count)
                sx = max(0, min(width - 1, int(round(x0 + (x1 - x0) * ratio))))
                sy = max(0, min(height - 1, int(round(y0 + (y1 - y0) * ratio))))
                if bool(stop_mask[sy, sx]):
                    return sx, sy
            return None

        def slide_along_contour(hit_x: int, hit_y: int, force: np.ndarray, distance: float) -> tuple[float, float]:
            if stop_normal_x is None or stop_normal_y is None:
                return float(hit_x), float(hit_y)
            try:
                nx = float(stop_normal_x[hit_y, hit_x])
                ny = float(stop_normal_y[hit_y, hit_x])
                tangent = np.array([-ny, nx], dtype="float32")
                tangent_norm = float(np.linalg.norm(tangent) or 0.0)
                if tangent_norm <= 0.0001:
                    return float(hit_x), float(hit_y)
                tangent = tangent / tangent_norm
                if float(np.dot(tangent, force)) < 0:
                    tangent = -tangent
                slide_distance = max(1.0, min(max(width, height) * 0.06, float(distance) * (0.38 + 0.44 * humidity)))
                return (
                    max(0.0, min(float(width - 1), float(hit_x) + float(tangent[0]) * slide_distance)),
                    max(0.0, min(float(height - 1), float(hit_y) + float(tangent[1]) * slide_distance)),
                )
            except Exception:
                return float(hit_x), float(hit_y)

        active_points = max(1, int(points))
        base_radius = max(2.0, min(width, height) * (0.005 + 0.030 * splash_scale))
        center = np.array([width * 0.5, height * 0.52], dtype="float32")
        point_seeds = [_rng_int(rng, 1, 2_147_483_646) for _ in range(active_points)]
        for point_seed in point_seeds:
            rng = random.Random(point_seed)
            mass = _rng_float(rng, mass_min, mass_max)
            if stickiness_max > stickiness_min:
                bucket_count = 5
                raw_bucket = _rng_int(rng, 0, bucket_count - 1)
                stickiness = stickiness_min + (stickiness_max - stickiness_min) * (raw_bucket / max(1, bucket_count - 1))
            else:
                stickiness = stickiness_min
            mass_norm = max(0.0, min(1.0, (mass - 0.05) / 2.75))
            mass_common = max(0.0, min(1.0, mass))
            mass_heavy_boost = max(0.0, min(1.0, (mass - 1.0) / 1.8))
            opacity = _rng_float(rng, opacity_min, opacity_max)
            impact_energy = (vehicle_speed ** 1.35) * (0.35 + 0.65 * mass_norm) * (0.50 + 0.50 * splash_scale)
            impact_force = max(0.0, min(1.0, impact_energy * (0.72 + 0.48 * mass_heavy_boost)))
            dry_adhesion = (1.0 - humidity) * stickiness
            cohesive_slide = humidity * stickiness
            wet_runoff = humidity * (1.0 - stickiness) * (1.0 - 0.45 * mass_norm)
            dry_dust_runoff = (1.0 - humidity) * (1.0 - stickiness) * (1.0 - mass_norm)
            start_x = _rng_int(rng, 0, max(1, width - 1))
            start_y = _rng_int(rng, 0, max(1, int(height * 0.62)))
            mass_trail_factor = max(
                0.10,
                (0.24 + 0.76 * (mass_common ** 1.10)) * (1.0 + 0.55 * mass_heavy_boost),
            )
            mass_trail_factor *= 0.42 + 0.95 * cohesive_slide + 0.70 * wet_runoff + 0.32 * dry_dust_runoff
            mass_trail_factor *= 1.0 - 0.62 * dry_adhesion
            mass_trail_factor = max(0.05, mass_trail_factor)
            if trail_length <= 0.001 or humidity <= 0.001:
                length = 0
            else:
                trail_intent = 0.25 + 1.85 * trail_length
                length = int((height * (0.08 + 0.62 * humidity)) * mass_trail_factor * trail_intent * flow_power)
                min_length = int(
                    round(
                        height
                        * (0.018 + 0.090 * trail_length)
                        * (0.30 + 0.70 * mass_common)
                        * (0.55 + 0.45 * humidity)
                    )
                )
                max_length = int(
                    math.hypot(width, height)
                    * (0.18 + 0.82 * min(1.0, mass_trail_factor))
                    * (0.35 + 0.65 * trail_length)
                )
                length = max(min_length, min(max_length, length))
            thickness = max(1, int(round((1 + mass_norm * (0.75 + 4.6 * splash_scale + 2.6 * humidity + 2.4 * impact_force)) * (1.0 + 0.90 * dry_adhesion))))

            impact_mask = np.zeros((height, width), dtype="float32")
            radius_gain = 1.0 + 2.65 * impact_force + 1.35 * (vehicle_speed ** 2.0) * (0.30 + 0.70 * mass_norm)
            radius_x = max(2, int(round(base_radius * (0.45 + 2.85 * mass_norm) * (1.0 + 0.55 * dry_adhesion + 0.28 * stickiness * mass_norm) * radius_gain)))
            radius_y = max(2, int(round(radius_x * _rng_float(rng, 0.55, 1.30) * (1.0 + 0.22 * dry_adhesion + 0.55 * impact_force))))
            impact_opacity = min(1.0, opacity * (0.86 + 0.38 * dry_adhesion + 0.22 * stickiness + 0.70 * impact_force))
            vertex_count = _rng_int(rng, 9, 19)
            vertices: list[list[int]] = []
            angle_offset = _rng_float(rng, 0.0, math.tau)
            for idx in range(vertex_count):
                local_angle = angle_offset + (math.tau * idx / vertex_count) + _rng_float(rng, -0.18, 0.18)
                local_radius = _rng_float(rng, 0.52, 1.32)
                x = int(round(start_x + math.cos(local_angle) * radius_x * local_radius))
                y = int(round(start_y + math.sin(local_angle) * radius_y * local_radius))
                vertices.append([max(0, min(width - 1, x)), max(0, min(height - 1, y))])
            if len(vertices) >= 3:
                cv2.fillPoly(impact_mask, [np.array(vertices, dtype=np.int32)], impact_opacity, lineType=cv2.LINE_AA)

            lobes = max(1, int(round(1 + 10 * mass_norm * splash_scale + 8 * impact_force)))
            for _lobe in range(lobes):
                lobe_angle = _rng_float(rng, 0.0, math.tau)
                lobe_dist = _rng_float(rng, 0.05, 0.95 + 0.48 * impact_force) * max(radius_x, radius_y)
                lobe_x = int(round(start_x + math.cos(lobe_angle) * lobe_dist))
                lobe_y = int(round(start_y + math.sin(lobe_angle) * lobe_dist))
                lobe_radius = max(1, int(round(_rng_float(rng, 0.18, 0.55 + 0.35 * impact_force) * max(radius_x, radius_y))))
                cv2.circle(
                    impact_mask,
                    (max(0, min(width - 1, lobe_x)), max(0, min(height - 1, lobe_y))),
                    lobe_radius,
                    impact_opacity * _rng_float(rng, 0.35, 0.90),
                    -1,
                    lineType=cv2.LINE_AA,
                )

            holes = max(0, int(round(1 + 6 * (1.0 - humidity) * mass_norm)))
            for _hole in range(holes):
                hole_angle = _rng_float(rng, 0.0, math.tau)
                hole_dist = _rng_float(rng, 0.0, 0.75) * max(radius_x, radius_y)
                hole_x = int(round(start_x + math.cos(hole_angle) * hole_dist))
                hole_y = int(round(start_y + math.sin(hole_angle) * hole_dist))
                hole_radius = max(1, int(round(_rng_float(rng, 0.08, 0.22) * max(radius_x, radius_y))))
                cv2.circle(
                    impact_mask,
                    (max(0, min(width - 1, hole_x)), max(0, min(height - 1, hole_y))),
                    hole_radius,
                    0.0,
                    -1,
                    lineType=cv2.LINE_AA,
                )

            impact_blur = max(3, int(3 + 4 * humidity))
            if impact_blur % 2 == 0:
                impact_blur += 1
            impact_mask = cv2.GaussianBlur(impact_mask, (impact_blur, impact_blur), 0)
            alpha_mask = np.maximum(alpha_mask, np.clip(impact_mask, 0.0, 1.0))

            current_x = float(start_x)
            current_y = float(start_y)
            trail_points: list[list[int]] = [[int(current_x), int(current_y)]]

            if length > 1:
                segments = max(4, min(34, int(length / 7)))
                for step in range(1, segments + 1):
                    decay = 1.0 - (step / float(segments + 1))
                    progress = step / float(segments)
                    gravity_takeover = min(1.0, progress ** (0.72 + 0.55 * (1.0 - humidity)))
                    segment_force = initial_force * (1.0 - gravity_takeover) + gravity_norm * gravity_takeover
                    segment_norm = float(np.linalg.norm(segment_force) or 1.0)
                    segment_force = segment_force / segment_norm
                    wobble = _rng_float(rng, -1.0, 1.0) * (0.35 + 1.20 * humidity) * (0.35 + decay)
                    step_len = (length / segments) * (0.82 + 0.36 * decay)
                    next_x = current_x + float(segment_force[0]) * step_len + wobble * (1.0 - 0.45 * gravity_takeover)
                    next_y = current_y + float(segment_force[1]) * step_len
                    contour_hit = contour_hit_on_segment(current_x, current_y, next_x, next_y)
                    if contour_hit is not None:
                        hit_x, hit_y = contour_hit
                        if stop_on_dark_contour:
                            trail_points.append([hit_x, hit_y])
                            break
                        next_x, next_y = slide_along_contour(hit_x, hit_y, segment_force, step_len)
                    clamped_x = max(0, min(width - 1, int(round(next_x))))
                    clamped_y = max(0, min(height - 1, int(round(next_y))))
                    trail_points.append([clamped_x, clamped_y])
                    if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height:
                        break
                    current_x = next_x
                    current_y = next_y

            if len(trail_points) >= 2:
                trail_mask = np.zeros((height, width), dtype="float32")
                water_channel = max(0.0, min(1.0, humidity * trail_length * (0.56 + 0.44 * stickiness)))
                stream_width_scale = max(0.14, 1.0 - 0.78 * water_channel + 0.20 * dry_adhesion)
                trail_alpha = min(
                    1.0,
                    opacity
                    * (0.54 + 0.64 * humidity)
                    * (0.38 + 0.78 * mass_common)
                    * (0.72 + 0.62 * stickiness + 0.34 * wet_runoff),
                )
                for idx in range(1, len(trail_points)):
                    p0 = tuple(trail_points[idx - 1])
                    p1 = tuple(trail_points[idx])
                    fade = 1.0 - ((idx - 1) / max(1.0, len(trail_points) - 1.0))
                    segment_thickness = max(
                        1,
                        int(round(thickness * stream_width_scale * (0.34 + 0.78 * fade))),
                    )
                    cv2.line(
                        trail_mask,
                        p0,
                        p1,
                        trail_alpha * (0.18 + 0.82 * fade),
                        thickness=segment_thickness,
                        lineType=cv2.LINE_AA,
                    )
                for idx, point in enumerate(trail_points):
                    fade = 1.0 - (idx / max(1.0, len(trail_points) - 1.0))
                    radius = max(
                        1,
                        int(round(thickness * stream_width_scale * (0.28 + 0.55 * humidity) * (0.24 + fade))),
                    )
                    cv2.circle(trail_mask, tuple(point), radius, trail_alpha * (0.12 + 0.88 * fade), -1, lineType=cv2.LINE_AA)
                end_radius = max(1, int(round(thickness * stream_width_scale * (0.45 + 0.65 * humidity))))
                cv2.circle(trail_mask, tuple(trail_points[-1]), end_radius, trail_alpha * (0.22 + 0.35 * humidity), -1, lineType=cv2.LINE_AA)
                trail_blur = max(1, int(1 + 5 * (1.0 - water_channel) + 2 * (1.0 - humidity)))
                if trail_blur % 2 == 0:
                    trail_blur += 1
                trail_mask = cv2.GaussianBlur(trail_mask, (trail_blur, trail_blur), 0)
                alpha_mask = np.maximum(alpha_mask, np.clip(trail_mask, 0.0, 1.0))

            lower_runoff = max(0.0, min(1.0, (0.72 * wet_runoff + 0.38 * dry_dust_runoff) * (1.0 - 0.55 * mass_norm)))
            if lower_runoff > 0.035:
                runoff_mask = np.zeros((height, width), dtype="float32")
                runoff_len = int(height * (0.08 + 0.40 * trail_length) * lower_runoff * (0.85 + 0.35 * (1.0 - mass_norm)))
                runoff_len = max(4, min(height, runoff_len))
                drift = math.cos(angle) * wind_strength * (0.15 + 0.40 * lower_runoff)
                end_x = max(0, min(width - 1, int(round(start_x + drift * runoff_len))))
                end_y = max(0, min(height - 1, start_y + runoff_len))
                runoff_alpha = min(1.0, opacity * (0.16 + 0.42 * lower_runoff))
                runoff_thickness = max(1, int(round(thickness * (0.32 + 0.58 * lower_runoff))))
                cv2.line(
                    runoff_mask,
                    (start_x, start_y),
                    (end_x, end_y),
                    runoff_alpha,
                    thickness=runoff_thickness,
                    lineType=cv2.LINE_AA,
                )
                bottom_start = max(0, min(height - 1, int(round(height * (0.52 + 0.25 * (1.0 - lower_runoff))))))
                if bottom_start < height - 1:
                    gradient = np.linspace(0.0, 1.0, height - bottom_start, dtype="float32")[:, None]
                    x0 = max(0, start_x - int(radius_x * (1.5 + lower_runoff)))
                    x1 = min(width, start_x + int(radius_x * (1.5 + lower_runoff)) + 1)
                    runoff_mask[bottom_start:height, x0:x1] = np.maximum(
                        runoff_mask[bottom_start:height, x0:x1],
                        gradient * runoff_alpha * 0.36,
                    )
                blur = max(3, int(3 + 8 * lower_runoff))
                if blur % 2 == 0:
                    blur += 1
                runoff_mask = cv2.GaussianBlur(runoff_mask, (blur, blur), 0)
                alpha_mask = np.maximum(alpha_mask, np.clip(runoff_mask, 0.0, 1.0))

            if vehicle_speed > 0.015:
                airflow_mask = np.zeros((height, width), dtype="float32")
                impact_point = np.array([float(start_x), float(start_y)], dtype="float32")
                radial = impact_point - center
                radial_norm = float(np.linalg.norm(radial) or 0.0)
                if radial_norm < max(4.0, min(width, height) * 0.035):
                    radial = np.array([_rng_float(rng, -0.35, 0.35), -1.0], dtype="float32")
                    radial_norm = float(np.linalg.norm(radial) or 1.0)
                radial = radial / radial_norm
                wind_bias = np.array([math.cos(angle) * wind_strength, 0.0], dtype="float32")
                flow_dir = radial + wind_bias * (0.08 + 0.18 * vehicle_speed)
                flow_norm = float(np.linalg.norm(flow_dir) or 1.0)
                flow_dir = flow_dir / flow_norm
                flow_angle = math.atan2(float(flow_dir[1]), float(flow_dir[0]))
                fluid_spread = max(0.0, min(1.0, (1.0 - stickiness) * (0.45 + 0.55 * humidity)))
                speed_splash = vehicle_speed * (0.50 + 0.50 * splash_scale) * (0.62 + 0.38 * mass_norm) * (0.65 + 0.85 * fluid_spread + 0.60 * impact_force)
                speed_core_alpha = min(1.0, opacity * (0.24 + 0.88 * vehicle_speed) * (0.58 + 0.42 * mass_norm + 0.34 * impact_force))
                # High vehicle speed is interpreted as impact energy: dense mud balls flatten and can cover characters.
                occluding_blob_count = max(1, int(round(1 + 8 * impact_force + 4 * speed_splash)))
                for _blob in range(occluding_blob_count):
                    distance = _rng_float(rng, 0.0, max(2.0, max(radius_x, radius_y) * (0.35 + 1.40 * impact_force)))
                    lateral = _rng_float(rng, -max(radius_x, radius_y), max(radius_x, radius_y)) * (0.12 + 0.48 * impact_force)
                    blob_x = int(round(start_x + float(flow_dir[0]) * distance - float(flow_dir[1]) * lateral))
                    blob_y = int(round(start_y + float(flow_dir[1]) * distance + float(flow_dir[0]) * lateral))
                    if not (0 <= blob_x < width and 0 <= blob_y < height):
                        continue
                    blob_major = max(1, int(round(max(radius_x, radius_y) * _rng_float(rng, 0.20, 0.82 + 0.78 * impact_force))))
                    blob_minor = max(1, int(round(blob_major * _rng_float(rng, 0.26, 0.70) * (1.0 - 0.24 * fluid_spread))))
                    blob_angle = math.degrees(flow_angle + _rng_float(rng, -0.42, 0.42))
                    cv2.ellipse(
                        airflow_mask,
                        (blob_x, blob_y),
                        (blob_major, blob_minor),
                        blob_angle,
                        0,
                        360,
                        speed_core_alpha * _rng_float(rng, 0.42, 0.98),
                        -1,
                        lineType=cv2.LINE_AA,
                    )
                ray_count = max(1 + int(round(2 * vehicle_speed)), int(round(2 + 12 * speed_splash + 10 * impact_force)))
                spread = math.radians(7.0 + 34.0 * (1.0 - vehicle_speed) + 18.0 * splash_scale + 22.0 * fluid_spread)
                speed_alpha = opacity * (0.14 + 0.84 * vehicle_speed) * (0.58 + 0.42 * mass_norm) * (0.70 + 0.54 * fluid_spread)
                speed_alpha = max(speed_alpha, 0.06 * vehicle_speed * (0.35 + opacity), speed_core_alpha * (0.36 + 0.32 * impact_force))
                speed_length = int(min(width, height) * (0.05 + 0.50 * vehicle_speed) * (0.82 + 0.18 * trail_length) * (0.48 + 0.52 * mass_norm + 0.35 * impact_force) * (0.72 + 0.72 * fluid_spread))
                speed_length = max(int(4 + 8 * vehicle_speed), min(int(min(width, height) * 0.80), speed_length))
                for _ray in range(ray_count):
                    local_angle = flow_angle + _rng_float(rng, -spread, spread)
                    local_dir = np.array([math.cos(local_angle), math.sin(local_angle)], dtype="float32")
                    local_length = speed_length * _rng_float(rng, 0.45, 1.18)
                    end_x = int(round(start_x + float(local_dir[0]) * local_length))
                    end_y = int(round(start_y + float(local_dir[1]) * local_length))
                    end_x = max(0, min(width - 1, end_x))
                    end_y = max(0, min(height - 1, end_y))
                    ray_thickness = max(1, int(round(thickness * _rng_float(rng, 0.24, 0.78) * (1.0 - 0.35 * fluid_spread))))
                    cv2.line(
                        airflow_mask,
                        (start_x, start_y),
                        (end_x, end_y),
                        speed_alpha * _rng_float(rng, 0.32, 0.95),
                        thickness=ray_thickness,
                        lineType=cv2.LINE_AA,
                    )
                    if _rng_float(rng, 0.0, 1.0) < 0.55:
                        cv2.circle(
                            airflow_mask,
                            (end_x, end_y),
                            max(1, int(round(ray_thickness * _rng_float(rng, 0.6, 1.4)))),
                            speed_alpha * _rng_float(rng, 0.12, 0.38),
                            -1,
                            lineType=cv2.LINE_AA,
                        )
                splashlet_count = max(
                    int(round(1 + 5 * vehicle_speed)),
                    int(round(18 * speed_splash * (0.35 + splash_scale) + 20 * impact_force)),
                )
                for _splashlet in range(splashlet_count):
                    local_angle = flow_angle + _rng_float(rng, -spread * 1.35, spread * 1.35)
                    distance = _rng_float(rng, max(1.0, radius_x * 0.20), max(2.0, speed_length * (0.62 + 0.28 * impact_force)))
                    dot_x = int(round(start_x + math.cos(local_angle) * distance))
                    dot_y = int(round(start_y + math.sin(local_angle) * distance))
                    if not (0 <= dot_x < width and 0 <= dot_y < height):
                        continue
                    dot_radius = max(1, int(round(_rng_float(rng, 0.25, 1.35 + 1.10 * impact_force) * thickness * (0.55 + splash_scale + 0.65 * impact_force))))
                    cv2.circle(
                        airflow_mask,
                        (dot_x, dot_y),
                        dot_radius,
                        speed_alpha * _rng_float(rng, 0.18, 0.62 + 0.32 * impact_force),
                        -1,
                        lineType=cv2.LINE_AA,
                    )
                airflow_blur = max(3, int(1 + 5 * (1.0 - vehicle_speed + humidity * 0.45)))
                if airflow_blur % 2 == 0:
                    airflow_blur += 1
                airflow_mask = cv2.GaussianBlur(airflow_mask, (airflow_blur, airflow_blur), 0)
                alpha_mask = np.maximum(alpha_mask, np.clip(airflow_mask, 0.0, 1.0))

        wash_strength = rain_strength * vehicle_speed
        if wash_strength > 0.015:
            rain_dir = np.array([math.cos(angle) * wind_strength * (0.35 + 0.75 * (1.0 - profile.rain_drop_size)), 1.0], dtype="float32")
            rain_norm = float(np.linalg.norm(rain_dir) or 1.0)
            rain_dir = rain_dir / rain_norm
            washed = np.zeros_like(alpha_mask)
            steps = max(2, int(round(2 + 7 * wash_strength)))
            for step in range(1, steps + 1):
                decay = 1.0 - (step / float(steps + 1))
                distance = step * (2.0 + 9.0 * wash_strength)
                transform = np.float32([[1, 0, float(rain_dir[0]) * distance], [0, 1, float(rain_dir[1]) * distance]])
                shifted = cv2.warpAffine(
                    alpha_mask,
                    transform,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                washed = np.maximum(washed, shifted * (0.10 + 0.55 * decay) * wash_strength)
            alpha_mask = np.clip((alpha_mask * (1.0 - 0.42 * wash_strength)) + washed, 0.0, 1.0)

        dirt_color = np.array([32.0, 45.0, 68.0], dtype="float32")
        alpha = np.clip(alpha_mask, 0.0, 1.0)[:, :, None]
        out = image.astype("float32")
        out = out * (1.0 - alpha) + dirt_color * alpha
        result = np.clip(out, 0, 255).astype("uint8")
        return (result, np.clip(alpha_mask, 0.0, 1.0)) if return_mask else result
    except Exception:
        return (image, None) if return_mask else image


def _build_relief_bounce_light(
    source_map,
    receiver_edge_map,
    light_x: float,
    light_y: float,
    depth: int,
    strength: float,
) -> object:
    if np is None or cv2 is None:
        return None
    try:
        depth = max(1, min(4, int(float(depth or 1))))
        strength = max(0.0, min(1.0, float(strength or 0.0)))
        if depth <= 1 or strength <= 0.001:
            return None
        source = np.clip(np.asarray(source_map, dtype="float32"), 0.0, 1.0)
        receivers = np.clip(np.asarray(receiver_edge_map, dtype="float32"), 0.0, 1.0)
        if source.shape != receivers.shape or source.size <= 0:
            return None
        if float(np.max(source) or 0.0) <= 0.001 or float(np.max(receivers) or 0.0) <= 0.001:
            return None

        receiver_halo = cv2.dilate(
            receivers,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        )
        receiver_halo = cv2.GaussianBlur(np.clip(receiver_halo, 0.0, 1.0), (0, 0), sigmaX=1.35, sigmaY=1.35)
        receiver_halo = np.clip(np.maximum(receivers, receiver_halo * 0.92), 0.0, 1.0)

        current = cv2.GaussianBlur(source, (0, 0), sigmaX=0.36, sigmaY=0.36)
        accumulated = np.zeros_like(current, dtype="float32")
        height, width = source.shape[:2]
        surface_scale = max(1.0, min(4.0, float(max(width, height)) / 95.0))
        for bounce_index in range(2, depth + 1):
            reach = (1.65 + bounce_index * 2.85) * surface_scale
            attenuation = strength * (0.78 ** max(0, bounce_index - 2))
            direction_len = math.hypot(float(light_x), float(light_y))
            if direction_len > 0.0001:
                dir_x = float(light_x) / direction_len
                dir_y = float(light_y) / direction_len
            else:
                dir_x = 0.0
                dir_y = 0.0
            tangent_x = -dir_y
            tangent_y = dir_x
            parallel_gain = min(1.0, direction_len)
            transform = np.float32([[1, 0, float(light_x) * reach], [0, 1, float(light_y) * reach]])
            directional = cv2.warpAffine(
                current,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            lateral = np.zeros_like(current, dtype="float32")
            if parallel_gain > 0.001:
                lateral_reach = reach * (0.48 + 0.18 * bounce_index) * parallel_gain
                for side in (-1.0, 1.0):
                    side_transform = np.float32(
                        [
                            [1, 0, tangent_x * lateral_reach * side + dir_x * reach * 0.20],
                            [0, 1, tangent_y * lateral_reach * side + dir_y * reach * 0.20],
                        ]
                    )
                    lateral = np.maximum(
                        lateral,
                        cv2.warpAffine(
                            current,
                            side_transform,
                            (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        ),
                    )
            diffuse = cv2.GaussianBlur(
                current,
                (0, 0),
                sigmaX=0.70 + bounce_index * 0.58,
                sigmaY=0.70 + bounce_index * 0.58,
            )
            candidate = np.clip((directional * 0.58 + lateral * 0.42 + diffuse * 0.28) * receiver_halo, 0.0, 1.0)
            if float(np.max(candidate) or 0.0) <= 0.001:
                break
            accumulated = np.clip(accumulated + candidate * attenuation, 0.0, 1.70)
            current = cv2.GaussianBlur(np.clip(candidate + current * 0.18, 0.0, 1.0), (0, 0), sigmaX=0.30, sigmaY=0.30)
        if float(np.max(accumulated) or 0.0) <= 0.001:
            return None
        return cv2.GaussianBlur(
            np.clip(accumulated, 0.0, 1.0),
            (0, 0),
            sigmaX=0.38 + 0.08 * depth,
            sigmaY=0.38 + 0.08 * depth,
        )
    except Exception:
        return None


def _apply_dark_relief_effect(image, profile: AugmentationProfile, extra_height_mask=None, *, base_surface: dict | None = None) -> object:
    if np is None or cv2 is None or profile.dark_relief_strength <= 0:
        return image

    try:
        strength = max(0.0, min(3.0, float(profile.dark_relief_strength)))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        symbol_curve_base = _surface_float_mask(base_surface, "symbol_height", width, height)
        if symbol_curve_base is None:
            symbol_curve_base = _build_symbol_contour_height_map(image, _contour_detection_sensitivity(profile))

        surface = _build_contour_blanket_surface(
            image,
            profile,
            None,
            relief_mask=extra_height_mask,
            include_water=False,
            base_surface=base_surface,
        )
        soft_mask = surface.get("height") if surface else None
        if soft_mask is None:
            # Dark text/stains become a soft height map; bright regions stay flat.
            dark_mask = symbol_curve_base
            if dark_mask is None:
                dark_mask = 1.0 - (gray.astype("float32") / 255.0)
                dark_mask = np.clip((dark_mask - 0.18) / 0.58, 0.0, 1.0)
                symbol_curve_base = dark_mask
            soft_mask = cv2.GaussianBlur(dark_mask, (0, 0), sigmaX=1.0, sigmaY=1.0)
        else:
            soft_mask = np.clip(np.asarray(soft_mask, dtype="float32"), 0.0, 1.0)
        surface_normal_x = surface.get("normal_x") if surface else None
        surface_normal_y = surface.get("normal_y") if surface else None
        surface_normal_energy = surface.get("normal_energy") if surface else None
        mud_mask = None
        mud_presence = 0.0
        mud_mass = 0.0
        mud_stickiness = 0.0
        mud_gloss = 0.0
        if extra_height_mask is not None:
            mud_mask = np.asarray(extra_height_mask, dtype="float32")
            if mud_mask.shape[:2] != soft_mask.shape[:2]:
                mud_mask = cv2.resize(mud_mask, (soft_mask.shape[1], soft_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            mud_mask = np.clip(mud_mask, 0.0, 1.0)
            mud_mask = cv2.GaussianBlur(mud_mask, (0, 0), sigmaX=0.55, sigmaY=0.55)
            mud_presence = max(float(np.percentile(mud_mask, 92.0) or 0.0), float(mud_mask.max() or 0.0) * 0.45)
            mass_min = max(0.05, min(2.8, float(getattr(profile, "dirt_flow_mass_min", 0.18) or 0.18)))
            mass_max = max(0.05, min(2.8, float(getattr(profile, "dirt_flow_mass_max", 1.0) or 1.0)))
            mud_mass = max(0.0, min(1.0, (((mass_min + mass_max) * 0.5) - 0.05) / 2.75))
            mud_stickiness = max(0.0, min(1.0, float(getattr(profile, "dirt_flow_stickiness", 0.45) if getattr(profile, "dirt_flow_stickiness", None) is not None else 0.45)))
            mud_gloss = max(0.0, min(1.0, float(getattr(profile, "wet_mud_gloss_strength", 0.0) or 0.0)))
            mud_height_gain = 1.18 + 0.72 * mud_mass + 0.40 * mud_stickiness + 0.28 * mud_gloss
            soft_mask = np.clip((soft_mask * 0.64) + (mud_mask * mud_height_gain), 0.0, 1.0)

        light_context = _active_relief_light_context(profile, soft_mask.shape[0], soft_mask.shape[1])
        if not light_context:
            return image

        grad_x = cv2.Sobel(soft_mask, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(soft_mask, cv2.CV_32F, 0, 1, ksize=3)
        if (
            isinstance(surface_normal_x, np.ndarray)
            and isinstance(surface_normal_y, np.ndarray)
            and surface_normal_x.shape == soft_mask.shape
            and surface_normal_y.shape == soft_mask.shape
        ):
            grad_x = np.clip(surface_normal_x.astype("float32"), -1.0, 1.0)
            grad_y = np.clip(surface_normal_y.astype("float32"), -1.0, 1.0)
        light_x = float(light_context.get("x", 0.0) or 0.0)
        light_y = float(light_context.get("y", 0.0) or 0.0)
        light_strength = max(0.0, min(1.0, float(light_context.get("strength", 0.0) or 0.0)))
        shadow_z_gain = max(0.42, min(7.2, float(light_context.get("shadow_z_gain", 1.0) or 1.0)))
        shadow_blur_gain = max(0.55, min(2.45, float(light_context.get("shadow_blur_gain", 1.0) or 1.0)))
        shadow_crispness = max(0.18, min(0.88, float(light_context.get("shadow_crispness", 0.35) or 0.35)))
        light_gate = light_context.get("light_gate")
        if isinstance(light_gate, np.ndarray) and light_gate.shape == soft_mask.shape:
            light_gate = np.clip(light_gate.astype("float32"), 0.0, 1.0)
        else:
            light_gate = np.ones_like(soft_mask, dtype="float32")
        light = (grad_x * light_x) + (grad_y * light_y)
        max_abs = float(np.max(np.abs(light)) or 0.0)
        if max_abs > 0:
            light = light / max_abs

        edge_mask = np.clip(np.abs(grad_x) + np.abs(grad_y), 0.0, 1.0)
        if isinstance(surface_normal_energy, np.ndarray) and surface_normal_energy.shape == soft_mask.shape:
            edge_mask = np.clip(np.maximum(edge_mask, surface_normal_energy.astype("float32")), 0.0, 1.0)
        edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), sigmaX=0.75, sigmaY=0.75)
        smooth_symbol_edge = _build_smooth_symbol_contour_edge_mask(symbol_curve_base if symbol_curve_base is not None else soft_mask)
        if smooth_symbol_edge is not None:
            edge_mask = np.clip(np.maximum(smooth_symbol_edge, edge_mask * 0.24), 0.0, 1.0)
        shadow_edge_mask = smooth_symbol_edge if smooth_symbol_edge is not None else edge_mask
        directional_shadow_edge = _build_directional_symbol_shadow_edge_mask(
            symbol_curve_base if symbol_curve_base is not None else soft_mask,
            light_x,
            light_y,
            fallback_edge=shadow_edge_mask,
        )
        if directional_shadow_edge is not None:
            shadow_edge_mask = directional_shadow_edge
        if mud_mask is not None:
            mud_grad_x = cv2.Sobel(mud_mask, cv2.CV_32F, 1, 0, ksize=3)
            mud_grad_y = cv2.Sobel(mud_mask, cv2.CV_32F, 0, 1, ksize=3)
            mud_edge = np.clip(np.abs(mud_grad_x) + np.abs(mud_grad_y), 0.0, 1.0)
            mud_edge = cv2.GaussianBlur(mud_edge, (0, 0), sigmaX=0.45, sigmaY=0.45)
            edge_mask = np.clip(edge_mask + (mud_edge * 1.35), 0.0, 1.0)
            shadow_edge_mask = np.clip(shadow_edge_mask + mud_edge * 0.42, 0.0, 1.0)
        bump_boost = 1.0
        if mud_mask is not None:
            bump_boost += mud_presence * (0.48 + 0.62 * mud_mass + 0.44 * mud_stickiness + 0.34 * mud_gloss)
        relief_mask = np.clip((soft_mask * 0.45) + (edge_mask * 1.55), 0.0, 1.0)
        visible_strength = strength * light_strength
        visible_map = light_gate[:, :, None] * visible_strength
        front_light = np.clip(light, 0.0, 1.0)[:, :, None] * (150.0 * bump_boost) * relief_mask[:, :, None] * visible_map
        edge_highlight = np.clip(light, 0.0, 1.0) * edge_mask * np.clip(soft_mask + edge_mask, 0.0, 1.0) * light_gate
        edge_highlight = cv2.GaussianBlur(edge_highlight, (0, 0), sigmaX=0.35 + 0.45 * visible_strength, sigmaY=0.35 + 0.45 * visible_strength)
        edge_highlight = edge_highlight[:, :, None] * (38.0 * visible_strength * bump_boost)

        mud_shadow_gain = 0.0
        if mud_mask is not None:
            mud_shadow_gain = mud_presence * (0.44 + 0.72 * mud_mass) * (0.70 + 0.45 * mud_stickiness)
        relief_height_gain = 0.72 + 0.38 * min(1.0, strength / 3.0)
        shadow_reach = (
            1.0 + 4.4 * visible_strength + (2.0 + 4.8 * visible_strength) * mud_shadow_gain
        ) * shadow_z_gain * relief_height_gain
        # The shadow is cast away from the headlight source, so it follows the
        # light travel vector instead of pointing back toward the reflector.
        shadow_dx = int(round(light_x * shadow_reach))
        shadow_dy = int(round(light_y * shadow_reach))
        shadow_source = np.clip(
            (
                shadow_edge_mask * (0.86 + 0.10 * min(3.0, visible_strength))
                + soft_mask * (0.035 + 0.12 * mud_shadow_gain)
            )
            * light_gate,
            0.0,
            1.0,
        )
        shadow_source = cv2.dilate(
            shadow_source,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        shadow_source = cv2.GaussianBlur(shadow_source, (0, 0), sigmaX=0.30, sigmaY=0.30)
        shadow_span = max(abs(float(shadow_dx)), abs(float(shadow_dy)), float(shadow_reach))
        sweep_steps = max(2, min(56, int(math.ceil(shadow_span * 2.35))))
        swept_shadow = np.zeros_like(shadow_source, dtype="float32")
        terminal_shadow = np.zeros_like(shadow_source, dtype="float32")
        for step in range(1, sweep_steps + 1):
            ratio = float(step) / float(sweep_steps)
            transform = np.float32([[1, 0, light_x * shadow_reach * ratio], [0, 1, light_y * shadow_reach * ratio]])
            shifted = cv2.warpAffine(
                shadow_source,
                transform,
                (edge_mask.shape[1], edge_mask.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            swept_shadow = np.maximum(swept_shadow, shifted * (0.92 + 0.08 * ratio))
            if step == sweep_steps:
                terminal_shadow = shifted
        shadow_blur = (
            0.72
            + 0.88 * visible_strength
            + 0.95 * mud_shadow_gain
            + 0.10 * shadow_z_gain
        ) * shadow_blur_gain
        filled_shadow = cv2.GaussianBlur(swept_shadow, (0, 0), sigmaX=max(0.36, shadow_blur * 0.38), sigmaY=max(0.36, shadow_blur * 0.38))
        soft_shadow = cv2.GaussianBlur(swept_shadow, (0, 0), sigmaX=shadow_blur, sigmaY=shadow_blur)
        terminal_shadow = cv2.GaussianBlur(terminal_shadow, (0, 0), sigmaX=max(0.42, shadow_blur * 0.52), sigmaY=max(0.42, shadow_blur * 0.52))
        destination_gate = cv2.GaussianBlur(light_gate, (0, 0), sigmaX=0.70 + 0.08 * shadow_blur, sigmaY=0.70 + 0.08 * shadow_blur)
        destination_gate = np.clip(destination_gate, 0.0, 1.0)
        cast_shadow = np.clip(
            (
                filled_shadow * (0.78 + 0.20 * shadow_crispness)
                + soft_shadow * (0.26 + 0.18 * (1.0 - shadow_crispness))
                + terminal_shadow * 0.06
            )
            * destination_gate,
            0.0,
            1.0,
        )
        plate_floor_payload = _estimate_plate_shadow_floor(image, symbol_curve_base if symbol_curve_base is not None else soft_mask)
        if plate_floor_payload is not None:
            plate_shadow_floor, plate_receiver_gate = plate_floor_payload
            receiver_gate = cv2.GaussianBlur(
                np.clip(plate_receiver_gate, 0.0, 1.0),
                (0, 0),
                sigmaX=0.48,
                sigmaY=0.48,
            )
            cast_shadow = np.clip(cast_shadow * receiver_gate, 0.0, 1.0)
        else:
            plate_shadow_floor = None
        contour_shadow = cast_shadow[:, :, None] * (68.0 * visible_strength * bump_boost)
        contact_shadow = cv2.GaussianBlur(
            np.clip(shadow_edge_mask * soft_mask * light_gate, 0.0, 1.0),
            (0, 0),
            sigmaX=0.82 + 0.88 * visible_strength,
            sigmaY=0.82 + 0.88 * visible_strength,
        )
        contact_shadow = contact_shadow[:, :, None] * (8.0 * visible_strength * bump_boost)
        bounce_depth = max(1, min(4, int(float(getattr(profile, "relief_bounce_depth", 1) or 1))))
        bounce_strength = max(0.0, min(1.0, float(getattr(profile, "relief_bounce_strength", 0.28) if getattr(profile, "relief_bounce_strength", None) is not None else 0.28)))
        secondary_light = None
        if bounce_depth > 1 and bounce_strength > 0.001:
            bounce_source = np.clip(
                np.clip(light, 0.0, 1.0)
                * relief_mask
                * np.clip(edge_mask * 0.72 + soft_mask * 0.28, 0.0, 1.0)
                * light_gate,
                0.0,
                1.0,
            )
            receiver_bounce_map = np.clip(np.maximum(shadow_edge_mask, edge_mask * 0.62), 0.0, 1.0)
            secondary_light = _build_relief_bounce_light(
                bounce_source,
                receiver_bounce_map,
                light_x,
                light_y,
                bounce_depth,
                bounce_strength,
            )
        if secondary_light is not None:
            secondary_gain = (
                44.0
                + 36.0 * bounce_strength
                + 10.0 * max(0, bounce_depth - 2)
            ) * visible_strength * bump_boost
            secondary_light = secondary_light[:, :, None] * secondary_gain

        lit_out = image.astype("float32") + front_light + edge_highlight
        shadowed = lit_out - contour_shadow - contact_shadow
        if plate_shadow_floor is not None:
            # Cast shadow attenuates light on the plate; it should not paint a
            # dark mark that drops below the local plate material.
            shadow_presence = np.clip(cast_shadow[:, :, None] + contact_shadow / 255.0, 0.0, 1.0)
            floor = np.minimum(lit_out, plate_shadow_floor)
            shadowed = np.where(shadow_presence > 0.001, np.maximum(shadowed, floor), shadowed)
        out = shadowed
        if secondary_light is not None:
            out = out + secondary_light
        return np.clip(out, 0, 255).astype("uint8")
    except Exception:
        return image


def _apply_overhang_shadow_effect(image, profile: AugmentationProfile, rng) -> object:
    """Simulate a soft top shadow cast by a bodywork lip above the plate."""
    if np is None or cv2 is None:
        return image

    try:
        strength = max(0.0, min(1.0, float(getattr(profile, "overhang_shadow_strength", 0.0) or 0.0)))
        if strength <= 0.001:
            return image
        height, width = image.shape[:2]
        if height < 8 or width < 8:
            return image

        light_x, light_y, _scene_beam_strength = _scene_or_fallback_light_direction(profile, height, width)
        # Image-space positive Y means light travels downward from a source
        # above the plate. If the scene light is neutral or points sideways,
        # keep the slider useful by falling back to a soft top lip shadow.
        top_light = max(0.0, light_y)
        if top_light <= 0.015:
            top_light = 0.55

        max_depth = max(0.0, min(0.60, float(getattr(profile, "overhang_shadow_depth", 0.60) if getattr(profile, "overhang_shadow_depth", None) is not None else 0.60)))
        manual_skew = max(-1.0, min(1.0, float(getattr(profile, "overhang_shadow_skew", 0.0) or 0.0)))
        depth_ratio = max_depth * (0.18 + 0.82 * top_light)
        if depth_ratio <= 0.002:
            return image

        yy, xx = np.mgrid[0:height, 0:width].astype("float32")
        centered_x = (xx / max(1.0, float(width - 1))) - 0.5
        side_light = light_x
        random_tilt = _rng_float(rng, -0.08, 0.08)
        tilt = (side_light * 0.12 + manual_skew * 0.42 + random_tilt) * height * depth_ratio
        wave_phase = _rng_float(rng, 0.0, math.tau)
        wave_amp = height * (0.006 + 0.026 * depth_ratio) * _rng_float(rng, 0.35, 1.0)
        boundary = (
            height * depth_ratio
            + centered_x * tilt
            + np.sin((xx / max(1.0, width)) * math.tau * _rng_float(rng, 0.55, 1.25) + wave_phase) * wave_amp
        )
        softness = max(2.0, height * (0.025 + 0.075 * (1.0 - top_light) + 0.045 * strength))
        shadow = np.clip((boundary + softness - yy) / max(1.0, softness * 2.0), 0.0, 1.0)
        shadow = shadow * shadow * (3.0 - 2.0 * shadow)
        shadow = cv2.GaussianBlur(shadow.astype("float32"), (0, 0), sigmaX=0.65 + 2.4 * strength, sigmaY=0.85 + 2.8 * strength)

        darken = shadow[:, :, None] * (0.28 + 0.68 * strength) * (0.45 + 0.55 * top_light)
        darken = np.clip(darken, 0.0, 0.86)
        cool = np.array([0.94, 0.91, 0.87], dtype="float32")
        out = image.astype("float32")
        out = out * (1.0 - darken)
        out = out * (1.0 - 0.10 * darken) + (out * cool) * (0.10 * darken)
        return np.clip(out, 0, 255).astype("uint8")
    except Exception:
        return image


def _apply_plate_horizontal_curve_effect(image, profile: AugmentationProfile) -> object:
    """Bend the plate crop as if it was screwed onto a rounded car body.

    The existing reflectance model only changed light distribution. This warp
    makes the horizontal plate contour visibly bow by simulating a shallow
    cylindrical bend around the vertical axis.
    """
    if np is None or cv2 is None:
        return image
    try:
        strength = max(0.0, min(2.0, float(getattr(profile, "plate_reflect_curve_strength", 0.0) or 0.0)))
        if strength <= 0.001:
            return image
        height, width = image.shape[:2]
        if height < 10 or width < 20:
            return image

        gain = min(1.0, strength / 2.0)
        yy, xx = np.mgrid[0:height, 0:width].astype("float32")
        center_x = (float(width) - 1.0) * 0.5
        center_y = (float(height) - 1.0) * 0.5
        nx = (xx - center_x) / max(1.0, center_x)

        # Convex bend toward the camera: the middle of the plate is visually a
        # little closer, so the top/bottom contours bow and characters get a
        # small horizontal perspective squeeze near the sides.
        center_bulge = np.clip(1.0 - nx * nx, 0.0, 1.0)
        vertical_scale = 1.0 + center_bulge * (0.045 + 0.105 * gain) * gain
        horizontal_scale = 1.0 + center_bulge * (0.020 + 0.055 * gain) * gain
        side_roll = (nx * np.abs(nx)) * (0.010 + 0.026 * gain) * float(width) * gain

        map_x = center_x + ((xx - center_x) - side_roll) / horizontal_scale
        map_y = center_y + (yy - center_y) / vertical_scale
        warped = cv2.remap(
            image,
            map_x.astype("float32"),
            map_y.astype("float32"),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # A small side falloff makes the geometry legible even without a strong
        # reflector: curved edges receive less frontal light than the center.
        side_falloff = np.clip((np.abs(nx) ** 1.75) * (0.035 + 0.085 * gain), 0.0, 0.16)
        center_lift = np.clip(center_bulge * (0.010 + 0.035 * gain), 0.0, 0.06)
        out = warped.astype("float32")
        if out.ndim == 2:
            out = out * (1.0 - side_falloff)
            out = out + (255.0 - out) * center_lift
        else:
            out = out * (1.0 - side_falloff[:, :, None])
            out = out + (255.0 - out) * center_lift[:, :, None]
        return np.clip(out, 0, 255).astype("uint8")
    except Exception:
        return image


def _apply_plate_reflectance_effect(image, profile: AugmentationProfile, rng) -> object:
    """Approximate uneven reflectance of a slightly curved license plate."""
    if np is None or cv2 is None:
        return image

    try:
        gradient_strength = max(0.0, min(2.0, float(getattr(profile, "plate_reflect_gradient_strength", 0.0) or 0.0)))
        glare_strength = max(0.0, min(2.0, float(getattr(profile, "plate_reflect_glare_strength", 0.0) or 0.0)))
        curve_strength = max(0.0, min(2.0, float(getattr(profile, "plate_reflect_curve_strength", 0.0) or 0.0)))
        if gradient_strength <= 0.001 and glare_strength <= 0.001 and curve_strength <= 0.001:
            return image

        height, width = image.shape[:2]
        if height < 8 or width < 8:
            return image

        light_x, light_y, scene_beam_strength = _scene_or_fallback_light_direction(profile, height, width)
        angle = math.atan2(-light_y, light_x)
        scene_lights = _combined_scene_headlight_fields(profile, height, width)
        scene_specular = None
        if scene_lights:
            try:
                local_scene_specular = np.clip(np.asarray(scene_lights.get("specular"), dtype="float32"), 0.0, 1.0)
                if local_scene_specular.shape == (height, width):
                    scene_specular = cv2.GaussianBlur(
                        local_scene_specular,
                        (0, 0),
                        sigmaX=0.55 + 1.45 * min(1.0, glare_strength / 2.0),
                        sigmaY=0.45 + 1.05 * min(1.0, glare_strength / 2.0),
                    )
            except Exception:
                scene_specular = None
        camera_axis = max(
            max(0.0, min(1.0, float(getattr(profile, "light_normal_strength", 0.0) or 0.0))),
            scene_beam_strength * 0.28,
        )

        yy, xx = np.mgrid[0:height, 0:width].astype("float32")
        nx = (xx / max(1.0, float(width - 1))) * 2.0 - 1.0
        ny = (yy / max(1.0, float(height - 1))) * 2.0 - 1.0
        axis = nx * light_x + ny * light_y
        axis_extent = float(np.max(np.abs(axis)) or 1.0)
        axis = axis / axis_extent

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        bright_mask = np.clip((gray - 0.36) / 0.54, 0.0, 1.0)
        bright_mask = cv2.GaussianBlur(bright_mask, (0, 0), sigmaX=0.75, sigmaY=0.75)
        surface_mask = np.clip(0.35 + 0.65 * bright_mask, 0.0, 1.0)

        dark_side = np.clip(-axis, 0.0, 1.0)
        light_side = np.clip(axis, 0.0, 1.0)
        out = image.astype("float32")
        if gradient_strength > 0.001:
            gradient_gain = min(1.0, gradient_strength / 2.0)
            darken = dark_side[:, :, None] * surface_mask[:, :, None] * (0.12 + 0.46 * gradient_gain)
            brighten = light_side[:, :, None] * bright_mask[:, :, None] * (0.10 + 0.38 * gradient_gain) * (0.50 + 0.70 * camera_axis)
            out = out * (1.0 - darken)
            out = out + (255.0 - out) * np.clip(brighten, 0.0, 0.58)

        if curve_strength > 0.001:
            curve_gain = min(1.0, curve_strength / 2.0)
            horizontal_curve = np.clip(nx * nx, 0.0, 1.0)
            ridge_center = np.clip(light_x * 0.34 + curve_gain * 0.12 * math.sin(angle), -0.72, 0.72)
            ridge_sigma = max(0.11, 0.32 - 0.14 * curve_gain)
            curved_ridge = np.exp(-((nx - ridge_center) ** 2) / (2.0 * ridge_sigma * ridge_sigma)).astype("float32")
            edge_falloff = np.clip(horizontal_curve * surface_mask, 0.0, 1.0)
            ridge_light = np.clip(curved_ridge * bright_mask * (0.16 + 0.48 * curve_gain) * (0.45 + 0.75 * camera_axis), 0.0, 0.72)
            curve_shadow = np.clip(edge_falloff * (0.05 + 0.20 * curve_gain) * (0.55 + 0.45 * abs(light_x)), 0.0, 0.32)
            curve_shadow *= 1.0 - 0.42 * min(1.0, abs(float(light_x)))
            out = out * (1.0 - curve_shadow[:, :, None])
            out = out + (255.0 - out) * ridge_light[:, :, None]
            plate_surface = _plate_curvature_surface(profile, height, width)
            if plate_surface:
                try:
                    curve_normal_x = np.asarray(plate_surface.get("normal_x"), dtype="float32")
                    curve_normal_z = np.asarray(plate_surface.get("normal_z"), dtype="float32")
                    if curve_normal_x.shape == (height, width) and curve_normal_z.shape == (height, width):
                        # Scene direction is the light travel vector
                        # source->surface. Surface illumination needs the
                        # opposite vector, pointing from the plate to source.
                        side_response = curve_normal_x * float(-light_x)
                        frontal_response = curve_normal_z * (0.18 + 0.34 * camera_axis)
                        curved_incidence = np.clip(side_response + frontal_response, -1.0, 1.0)
                        curved_shadow = np.clip(-curved_incidence, 0.0, 1.0)
                        curved_light = np.clip(curved_incidence, 0.0, 1.0)
                        curved_shadow = cv2.GaussianBlur(
                            curved_shadow.astype("float32"),
                            (0, 0),
                            sigmaX=0.80 + 2.0 * curve_gain,
                            sigmaY=0.55 + 1.2 * curve_gain,
                        )
                        curved_light = cv2.GaussianBlur(
                            curved_light.astype("float32"),
                            (0, 0),
                            sigmaX=0.70 + 1.5 * curve_gain,
                            sigmaY=0.45 + 0.9 * curve_gain,
                        )
                        side_strength = min(1.0, abs(float(light_x)) * (0.45 + 0.65 * max(scene_beam_strength, gradient_strength / 2.0, glare_strength / 2.0)))
                        if side_strength > 0.001:
                            shade = np.clip(curved_shadow * surface_mask * side_strength * (0.12 + 0.56 * curve_gain), 0.0, 0.58)
                            lift = np.clip(curved_light * (0.35 + 0.65 * bright_mask) * side_strength * (0.12 + 0.48 * curve_gain), 0.0, 0.48)
                            out = out * (1.0 - shade[:, :, None])
                            out = out + (255.0 - out) * lift[:, :, None]
                except Exception:
                    pass

        if glare_strength > 0.001:
            # A shallow camera angle turns bright plate areas into a broad reflector.
            glare_gain = min(1.0, glare_strength / 2.0)
            curve_gain = min(1.0, curve_strength / 2.0)
            acute_factor = 0.30 + 1.15 * camera_axis
            curve = curve_gain * (0.72 * (nx * nx - 0.34) + 0.24 * np.sin(nx * math.pi * 1.25 + _rng_float(rng, -0.45, 0.45)))
            skew = light_x * (0.34 + 0.18 * curve_gain)
            band_center = (-0.22 + 0.44 * math.sin(angle)) + curve + skew * nx
            sigma = 0.20 - 0.10 * camera_axis + 0.05 * (1.0 - glare_gain) + 0.05 * curve_gain
            sigma = max(0.045, sigma)
            band = np.exp(-((ny - band_center) ** 2) / (2.0 * sigma * sigma)).astype("float32")
            broad_curve = np.exp(-((ny - curve * 0.65) ** 2) / (2.0 * (sigma * 2.55) ** 2)).astype("float32")
            specular = np.clip((band * (0.92 + 1.25 * curve_gain) + broad_curve * (0.28 + 0.32 * curve_gain)) * bright_mask, 0.0, 1.0)
            if scene_specular is not None:
                specular = np.maximum(specular, np.clip(scene_specular * bright_mask * (0.55 + 0.95 * glare_gain), 0.0, 1.0))
            specular = cv2.GaussianBlur(specular, (0, 0), sigmaX=1.0 + 3.8 * curve_gain, sigmaY=0.55 + 2.1 * curve_gain)
            specular *= (0.70 + 1.15 * glare_gain) * acute_factor
            tint = np.array([242.0, 248.0, 255.0], dtype="float32")
            out = out + (tint - out) * np.clip(specular[:, :, None] * 1.05, 0.0, 0.92)
            veil = cv2.GaussianBlur(specular, (0, 0), sigmaX=3.0 + 7.0 * glare_gain, sigmaY=1.4 + 4.2 * glare_gain)
            out = out + (255.0 - out) * np.clip(veil[:, :, None] * (0.16 + 0.38 * camera_axis), 0.0, 0.52)

        return np.clip(out, 0, 255).astype("uint8")
    except Exception:
        return image


def _apply_image_postprocess(image, profile: AugmentationProfile, rng=None, *, return_debug: bool = False) -> object:
    profile = (profile or AugmentationProfile()).normalized()
    rng = rng or random.Random(profile.seed)
    base_seed = int(rng.randint(1, 2_147_483_647))

    def effect_rng(offset: int) -> random.Random:
        return random.Random((base_seed + int(offset)) % 2_147_483_647 or 1)

    def profile_float(name: str, default: float = 0.0) -> float:
        try:
            value = getattr(profile, name, default)
            if value is None:
                value = default
            return float(value)
        except Exception:
            return float(default)

    needs_stable_contours = bool(
        profile_float("dark_relief_strength") > 0.001
        or profile_float("water_film_strength") > 0.001
        or profile_float("water_film_lens_strength") > 0.001
        or profile_float("traffic_headlight_strength") > 0.001
        or profile_float("traffic_headlight_2_strength") > 0.001
        or profile_float("traffic_headlight_3_strength") > 0.001
        or profile_float("rain_edge_mist_strength") > 0.001
        or profile_float("dirt_flow_strength") > 0.001
        or int(profile_float("dirt_flow_points", 0.0)) > 0
        or bool(getattr(profile, "dirt_flow_stop_on_dark_contour", False))
    )
    out = _apply_plate_horizontal_curve_effect(image, profile)
    stable_contour_surface = _build_stable_contour_geometry(out, profile) if needs_stable_contours else {}
    needs_relief_mask = bool(
        float(getattr(profile, "dark_relief_strength", 0.0) or 0.0) > 0.001
        or float(getattr(profile, "wet_mud_gloss_strength", 0.0) or 0.0) > 0.001
        or float(getattr(profile, "water_film_strength", 0.0) or 0.0) > 0.001
    )
    rain_strength = max(0.0, min(1.0, profile_float("rain_strength", 0.0)))
    rain_lens_strength = max(0.0, min(1.0, profile_float("rain_lens_strength", 0.0)))
    # Keep the same contract in the full pipeline: the UI slider controls the
    # post-lighting lens pass directly, instead of being hidden behind an
    # automatic rain-size baseline.
    post_rain_lens_strength = rain_lens_strength if rain_strength > 0.001 else 0.0
    needs_post_rain_lens = bool(rain_strength > 0.001 and post_rain_lens_strength > 0.001)
    needs_rain_mask = bool(needs_relief_mask or needs_post_rain_lens)
    rain_edge_debug_mask = build_rain_edge_debug_mask(out, profile) if return_debug else None
    out = _apply_night_effect(out, profile, base_surface=stable_contour_surface)
    out = _apply_overexposure_effect(out, profile, effect_rng(101))
    out = _apply_dirt_streak_effect(out, profile, effect_rng(211))
    dirt_result = _apply_physical_dirt_flow_effect(
        out,
        profile,
        effect_rng(307),
        return_mask=needs_relief_mask,
        base_surface=stable_contour_surface,
    )
    if needs_relief_mask:
        out, dirt_bump_mask = dirt_result
    else:
        out = dirt_result
        dirt_bump_mask = None
    out = _apply_coarse_noise(out, profile, effect_rng(401))
    rain_result = _apply_rain_effect(
        out,
        profile,
        effect_rng(503),
        return_mask=needs_rain_mask,
        base_surface=stable_contour_surface,
        apply_lens=not needs_post_rain_lens,
    )
    if needs_rain_mask:
        out, rain_bump_mask = rain_result
    else:
        out = rain_result
        rain_bump_mask = None
    relief_mask = None
    if dirt_bump_mask is not None and rain_bump_mask is not None:
        relief_mask = np.clip(np.maximum(dirt_bump_mask, rain_bump_mask * 0.72), 0.0, 1.0)
    elif dirt_bump_mask is not None:
        relief_mask = dirt_bump_mask
    elif rain_bump_mask is not None:
        relief_mask = np.clip(rain_bump_mask * 0.72, 0.0, 1.0)
    out = _apply_dark_relief_effect(out, profile, relief_mask, base_surface=stable_contour_surface)
    water_result = _apply_water_film_effect(
        out,
        profile,
        effect_rng(557),
        relief_mask,
        return_mask=needs_relief_mask,
        base_surface=stable_contour_surface,
    )
    if needs_relief_mask:
        out, water_film_mask = water_result
        if water_film_mask is not None:
            if relief_mask is not None:
                try:
                    relief_mask = np.clip(np.maximum(relief_mask, water_film_mask * 0.64), 0.0, 1.0)
                except Exception:
                    relief_mask = water_film_mask
            else:
                relief_mask = water_film_mask
    else:
        out = water_result
    # Headlights/reflections add scene lighting after the material relief has
    # already created its local contour shadows and highlights.
    out = _apply_plate_reflectance_effect(out, profile, effect_rng(575))
    out = _apply_wet_reflection_effect(out, profile, effect_rng(601))
    out = _apply_traffic_headlight_effect(out, profile, effect_rng(625), relief_mask, base_surface=stable_contour_surface)
    out = _apply_overhang_shadow_effect(out, profile, effect_rng(655))
    if needs_post_rain_lens and rain_bump_mask is not None:
        out = _apply_rain_lens_distortion(out, rain_bump_mask, post_rain_lens_strength, profile)
    out = _apply_camera_glare_effect(out, profile, effect_rng(709))
    if return_debug:
        return out, {"rain_edge_debug_mask": rain_edge_debug_mask}
    return out


def _apply_albumentations_image_effects(image, profile: AugmentationProfile, *, seed: int | None = None) -> object:
    try:
        A = _load_albumentations()
        transform = A.Compose(_build_image_effect_transforms(A, profile))
        if seed is not None:
            _seed_transform(transform, int(seed))
        result = transform(image=image)
        return _apply_direct_image_tuning(result.get("image", image), profile)
    except Exception:
        return image


def _apply_direct_image_tuning(image, profile: AugmentationProfile) -> object:
    """Apply deterministic scene sliders so preview changes match user input."""
    if np is None:
        return image
    try:
        brightness = max(0.0, min(0.25, float(getattr(profile, "brightness_limit", 0.0) or 0.0)))
        contrast = max(0.0, min(0.25, float(getattr(profile, "contrast_limit", 0.0) or 0.0)))
        saturation = max(0.0, min(3.0, float(getattr(profile, "saturation_limit", 1.0) if getattr(profile, "saturation_limit", None) is not None else 1.0)))
        if brightness <= 0.001 and contrast <= 0.001 and abs(saturation - 1.0) <= 0.001:
            return image

        tuned = image.astype("float32", copy=True)
        if contrast > 0.001:
            tuned = (tuned - 127.5) * (1.0 + 2.0 * contrast) + 127.5
        if brightness > 0.001:
            tuned = tuned + 255.0 * brightness
        tuned = np.clip(tuned, 0, 255).astype("uint8")

        if abs(saturation - 1.0) > 0.001 and cv2 is not None:
            if saturation < 1.0:
                gray = cv2.cvtColor(tuned, cv2.COLOR_BGR2GRAY).astype("float32")
                tuned = np.clip(
                    gray[:, :, None] * (1.0 - saturation) + tuned.astype("float32") * saturation,
                    0,
                    255,
                ).astype("uint8")
            else:
                hsv = cv2.cvtColor(tuned, cv2.COLOR_BGR2HSV).astype("float32")
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
                tuned = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
        return tuned
    except Exception:
        return image


def _build_geometry_transforms(A, profile: AugmentationProfile) -> list:
    transforms = []
    rotation_angle = float(profile.rotation_limit or 0.0)
    if abs(rotation_angle) > 0.001 or profile.translate_limit > 0 or profile.scale_limit > 0:
        try:
            transforms.append(
                A.Affine(
                    rotate=(rotation_angle, rotation_angle),
                    translate_percent=(-profile.translate_limit, profile.translate_limit),
                    scale=(1.0 - profile.scale_limit, 1.0 + profile.scale_limit),
                    shear=(0.0, 0.0),
                    p=1.0,
                )
            )
        except TypeError:
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=profile.translate_limit,
                    scale_limit=profile.scale_limit,
                    rotate_limit=(rotation_angle, rotation_angle),
                    p=1.0,
                )
            )

    if not transforms:
        transforms.append(A.NoOp(p=1.0))
    return transforms


def _build_image_effect_transforms(A, profile: AugmentationProfile) -> list:
    transforms = []
    if profile.flare_strength > 0 and hasattr(A, "RandomSunFlare"):
        strength = max(0.0, min(1.0, float(profile.flare_strength)))
        try:
            transforms.append(
                A.RandomSunFlare(
                    flare_roi=(0.0, 0.0, 1.0, 0.65),
                    src_radius=int(70 + 260 * strength),
                    src_color=(255, 245, 215),
                    num_flare_circles_range=(2, max(3, int(3 + 5 * strength))),
                    method="overlay",
                    p=max(0.10, min(0.85, 0.18 + 0.62 * strength)),
                )
            )
        except TypeError:
            transforms.append(A.RandomSunFlare(p=max(0.10, min(0.85, 0.18 + 0.62 * strength))))

    blur_strength = max(0.0, min(1.0, float(getattr(profile, "blur_strength", 0.0) or 0.0)))
    if blur_strength <= 0.001 and bool(getattr(profile, "blur_enabled", False)):
        blur_strength = 0.18
    if blur_strength > 0.001:
        blur_limit = 3 + int(round(8.0 * blur_strength))
        if blur_limit % 2 == 0:
            blur_limit += 1
        blur_limit = max(3, min(11, blur_limit))
        sigma_limit = (0.1, max(0.2, 0.25 + 2.2 * blur_strength))
        try:
            transforms.append(A.GaussianBlur(blur_limit=(3, blur_limit), sigma_limit=sigma_limit, p=1.0))
        except Exception:
            try:
                transforms.append(A.Blur(blur_limit=(3, blur_limit), p=1.0))
            except Exception:
                transforms.append(A.Blur(blur_limit=blur_limit, p=1.0))

    if not transforms:
        transforms.append(A.NoOp(p=1.0))
    return transforms


def _build_transform(profile: AugmentationProfile, *, has_keypoints: bool):
    A = _load_albumentations()
    transforms = _build_geometry_transforms(A, profile)

    bbox_kwargs = {
        "format": "yolo",
        "label_fields": ["class_labels"],
        "min_visibility": 0.0,
    }
    try:
        bbox_params = A.BboxParams(**bbox_kwargs, clip=True, filter_invalid_bboxes=False)
    except TypeError:
        try:
            bbox_params = A.BboxParams(**bbox_kwargs, clip=True)
        except TypeError:
            bbox_params = A.BboxParams(**bbox_kwargs)

    keypoint_params = None
    if has_keypoints:
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)

    return A.Compose(
        transforms,
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
    )


def preview_augmentation_image(image_path: Path, profile: AugmentationProfile) -> tuple[bool, str, dict]:
    """Return RGB original and augmented arrays for the modal preview."""
    if not CV2_AVAILABLE or cv2 is None:
        return False, "OpenCV jest niedostępny, więc podgląd augmentacji nie może zostać wykonany.", {}
    if not is_albumentations_available():
        return False, "Albumentations nie jest zainstalowane. Użyj przycisku doinstalowania w modalu.", {}

    image_path = Path(image_path)
    if not image_path.exists() or not image_path.is_file():
        return False, "Nie znaleziono obrazu do podglądu augmentacji.", {}

    image = cv2.imread(str(image_path))
    if image is None:
        return False, "Nie udało się odczytać obrazu do podglądu augmentacji.", {}

    profile = (profile or AugmentationProfile()).normalized()
    pipeline_description = describe_augmentation_components(profile)
    preview_note = ""
    try:
        height, width = image.shape[:2]
        pixel_count = int(width) * int(height)
        if pixel_count > MAX_AUGMENTATION_PREVIEW_PIXELS:
            scale = math.sqrt(MAX_AUGMENTATION_PREVIEW_PIXELS / float(pixel_count))
            resized_w = max(4, int(round(width * scale)))
            resized_h = max(4, int(round(height * scale)))
            image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
            preview_note = f" Podgląd w skali {scale * 100.0:.0f}% chroni RAM."
    except Exception:
        preview_note = ""
    try:
        A = _load_albumentations()
        geometry_transform = A.Compose(_build_geometry_transforms(A, profile))
        geometry_seed = _stable_preview_seed(image_path, profile)
        _seed_transform(geometry_transform, geometry_seed)
        augmented = geometry_transform(image=image)
        augmented_image = augmented.get("image", image)
        augmented_image = _apply_albumentations_image_effects(
            augmented_image,
            profile,
            seed=geometry_seed,
        )
        augmented_image, debug_payload = _apply_image_postprocess(
            augmented_image,
            profile,
            random.Random(geometry_seed),
            return_debug=True,
        )
    except Exception as exc:
        return False, f"Nie udało się przygotować podglądu augmentacji: {exc}", {}

    try:
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
    except Exception:
        original_rgb = image
        augmented_rgb = augmented_image

    return True, f"Podgląd augmentacji gotowy.{preview_note}", {
        "source": str(image_path),
        "original_rgb": original_rgb,
        "augmented_rgb": augmented_rgb,
        "augmentation_pipeline": pipeline_description,
        "rain_edge_debug_mask": debug_payload.get("rain_edge_debug_mask") if isinstance(debug_payload, dict) else None,
    }


def _objects_to_albumentations_payload(
    objects: Iterable[YoloObject],
    *,
    width: int,
    height: int,
) -> tuple[list[list[float]], list[int], list[tuple[float, float]], list[list[float | None]]]:
    bboxes: list[list[float]] = []
    class_labels: list[int] = []
    keypoints: list[tuple[float, float]] = []
    visibility_groups: list[list[float | None]] = []

    for obj in objects:
        bboxes.append([_clamp01(value) for value in obj.bbox])
        class_labels.append(int(obj.class_id))
        visibilities: list[float | None] = []
        for kp_x, kp_y, visibility in obj.keypoints:
            keypoints.append((_clamp01(kp_x) * width, _clamp01(kp_y) * height))
            visibilities.append(visibility)
        visibility_groups.append(visibilities)
    return bboxes, class_labels, keypoints, visibility_groups


def _format_augmented_labels(
    *,
    bboxes: list,
    class_labels: list,
    keypoints: list,
    visibility_groups: list[list[float | None]],
    width: int,
    height: int,
    kpt_count: int,
    kpt_dim: int,
) -> list[str]:
    lines: list[str] = []
    keypoint_index = 0
    for obj_idx, bbox in enumerate(bboxes):
        if len(bbox) < 4:
            continue
        try:
            class_id = int(class_labels[obj_idx])
        except Exception:
            class_id = 0

        values: list[str] = [str(class_id)]
        values.extend(_format_float(float(value)) for value in bbox[:4])

        if kpt_count > 0 and kpt_dim >= 2:
            visibilities = visibility_groups[obj_idx] if obj_idx < len(visibility_groups) else []
            for kp_idx in range(kpt_count):
                if keypoint_index >= len(keypoints):
                    return []
                kp_x, kp_y = keypoints[keypoint_index][:2]
                keypoint_index += 1
                values.append(_format_float(float(kp_x) / max(1, width)))
                values.append(_format_float(float(kp_y) / max(1, height)))
                if kpt_dim >= 3:
                    visibility = visibilities[kp_idx] if kp_idx < len(visibilities) else 2.0
                    try:
                        values.append(f"{float(visibility):.0f}")
                    except Exception:
                        values.append("2")

        lines.append(" ".join(values))
    return lines


def _split_variant_stem(stem: str) -> tuple[str, int, int] | None:
    """Return semantic prefix and numeric variant from names like ABC_DEF_001."""
    prefix, sep, variant = str(stem or "").rpartition("_")
    if not sep or not prefix or not variant.isdigit():
        return None
    return prefix, int(variant), len(variant)


def _collect_reserved_dataset_stems(dataset_dir: Path) -> set[str]:
    """Collect all image/label stems so generated variants never replace existing samples."""
    reserved: set[str] = set()
    dataset_dir = Path(dataset_dir)
    image_root = dataset_dir / "images"
    if image_root.exists():
        for image_path in image_root.rglob("*"):
            try:
                if image_path.is_file() and image_path.suffix.lower() in _image_extensions():
                    reserved.add(image_path.stem)
            except Exception:
                continue
    label_root = dataset_dir / "labels"
    if label_root.exists():
        for label_path in label_root.rglob("*.txt"):
            try:
                if label_path.is_file():
                    reserved.add(label_path.stem)
            except Exception:
                continue
    return reserved


def _unique_augmented_paths(
    image_path: Path,
    label_dir: Path,
    index: int,
    reserved_stems: set[str] | None = None,
) -> tuple[Path, Path]:
    image_dir = image_path.parent
    suffix = image_path.suffix or ".jpg"
    stem = image_path.stem
    reserved_stems = reserved_stems if reserved_stems is not None else set()

    def is_candidate_free(candidate_stem: str, candidate_img: Path, candidate_lbl: Path) -> bool:
        if candidate_stem in reserved_stems:
            return False
        return not candidate_img.exists() and not candidate_lbl.exists()

    def reserve(candidate_stem: str, candidate_img: Path, candidate_lbl: Path) -> tuple[Path, Path]:
        reserved_stems.add(candidate_stem)
        return candidate_img, candidate_lbl

    parsed_variant = _split_variant_stem(stem)
    if parsed_variant is not None:
        prefix, source_variant, width = parsed_variant
        counter = max(source_variant + 1, int(index))
        while True:
            candidate_stem = f"{prefix}_{counter:0{max(width, len(str(counter)))}d}"
            candidate_img = image_dir / f"{candidate_stem}{suffix}"
            candidate_lbl = label_dir / f"{candidate_stem}.txt"
            if is_candidate_free(candidate_stem, candidate_img, candidate_lbl):
                return reserve(candidate_stem, candidate_img, candidate_lbl)
            counter += 1

    counter = int(index)
    while True:
        candidate_name = f"{stem}__aug_{counter:04d}{suffix}"
        candidate_stem = Path(candidate_name).stem
        candidate_img = image_dir / candidate_name
        candidate_lbl = label_dir / f"{candidate_stem}.txt"
        if is_candidate_free(candidate_stem, candidate_img, candidate_lbl):
            return reserve(candidate_stem, candidate_img, candidate_lbl)
        counter += 1


def _write_json_file(path: Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_manifest(dataset_dir: Path, payload: dict) -> None:
    _write_json_file(Path(dataset_dir) / "augmentation_manifest.json", payload)


def _build_augmentation_profile_payload(
    dataset_dir: Path,
    profile: AugmentationProfile,
    stats: dict,
    generated_files: list[dict],
) -> dict:
    randomness_mode = normalize_augmentation_randomness_mode(getattr(profile, "randomness_mode", "realistic"))
    return {
        "schema": "augmentation_profile_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "policy": "train_only_no_crop_no_flip_no_strong_deformation",
        "task_target": _profile_task_target(profile),
        "randomness": {
            "mode": randomness_mode,
            "label": describe_augmentation_randomness_mode(randomness_mode),
            "seed": int(getattr(profile, "seed", 42) or 42),
            "manual": describe_manual_randomness_config(
                getattr(profile, "manual_randomness", {}),
                _profile_task_target(profile),
            ),
        },
        "pipeline": describe_augmentation_components(profile),
        "profile_jitter": {
            "enabled": randomness_mode != "fixed",
            "mode": f"per_sample_{randomness_mode}",
            "base_profile_is_expected_value": True,
        },
        "profile": asdict(profile),
        "stats": dict(stats),
        "generated_files": generated_files,
    }


def _format_augmentation_profile_report(payload: dict) -> str:
    stats = payload.get("stats", {}) if isinstance(payload.get("stats"), dict) else {}
    randomness = payload.get("randomness", {}) if isinstance(payload.get("randomness"), dict) else {}
    lines = [
        "Raport profilu augmentacji",
        f"Utworzono: {payload.get('created_at', '')}",
        f"Dataset: {payload.get('dataset_dir', '')}",
        f"Tor: {payload.get('task_target', '')}",
        f"Tryb losowosci: {randomness.get('label', randomness.get('mode', ''))}",
        f"Seed: {randomness.get('seed', '')}",
        "",
        "Wynik serii:",
        f"- train przed: {stats.get('train_before', 0)}",
        f"- wygenerowano: {stats.get('generated', 0)}",
        f"- pominieto: {stats.get('skipped', 0)}",
        f"- train po: {stats.get('train_after', 0)}",
        "",
        "Reczne odchylki:",
    ]
    manual = randomness.get("manual", {}) if isinstance(randomness.get("manual"), dict) else {}
    for group in manual.get("groups", []) if isinstance(manual.get("groups"), list) else []:
        state = "wlaczone" if group.get("enabled") else "wylaczone"
        amount = _percent(group.get("amount", 0.0), 0.0)
        lines.append(f"- {group.get('label', group.get('key', ''))}: {state}, {amount:.0f}%")
        active_fields = [
            f"{field.get('label', field.get('key', ''))} { _percent(field.get('amount', 0.0), 0.0):.0f}%"
            for field in group.get("fields", [])
            if isinstance(field, dict) and field.get("enabled")
        ]
        if active_fields:
            lines.append(f"  Parametry: {', '.join(active_fields)}")
    return "\n".join(lines).rstrip() + "\n"


def _write_augmentation_profile_files(dataset_dir: Path, payload: dict) -> None:
    dataset_dir = Path(dataset_dir)
    _write_json_file(dataset_dir / "augmentation_profile.json", payload)
    (dataset_dir / "augmentation_report.txt").write_text(
        _format_augmentation_profile_report(payload),
        encoding="utf-8",
    )


def _safe_preset_stem(name: object, fallback: str = "preset") -> str:
    raw = str(name or "").strip().lower()
    safe_chars: list[str] = []
    for char in raw:
        if ("a" <= char <= "z") or ("0" <= char <= "9") or char in {"_", "-"}:
            safe_chars.append(char)
        elif char.isspace() or char in {".", ",", ";", ":", "/", "\\"}:
            safe_chars.append("_")
    stem = "".join(safe_chars).strip("_-")
    return stem or fallback


def get_augmentation_presets_dir(target: str | None = None) -> Path:
    presets_dir = CONFIG.get_presets_dir("augmentation", target) if target else CONFIG.get_presets_dir("augmentation")
    presets_dir.mkdir(parents=True, exist_ok=True)
    return presets_dir


def build_augmentation_preset_payload(
    profile: AugmentationProfile,
    *,
    name: str = "",
    description: str = "",
    source_dataset_dir: Path | None = None,
) -> dict:
    profile = (profile or AugmentationProfile()).normalized()
    randomness_mode = normalize_augmentation_randomness_mode(getattr(profile, "randomness_mode", "realistic"))
    return {
        "schema": "augmentation_preset_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "name": str(name or "").strip(),
        "description": str(description or "").strip(),
        "task_target": _profile_task_target(profile),
        "source_dataset_dir": str(source_dataset_dir or ""),
        "randomness": {
            "mode": randomness_mode,
            "label": describe_augmentation_randomness_mode(randomness_mode),
            "seed": int(getattr(profile, "seed", 42) or 42),
            "manual": describe_manual_randomness_config(
                getattr(profile, "manual_randomness", {}),
                _profile_task_target(profile),
            ),
        },
        "pipeline": describe_augmentation_components(profile),
        "profile": asdict(profile),
    }


def save_augmentation_preset(
    profile: AugmentationProfile,
    name: str,
    *,
    description: str = "",
    target: str | None = None,
    source_dataset_dir: Path | None = None,
) -> Path:
    profile = (profile or AugmentationProfile(task_target=target or "")).normalized()
    if target:
        profile = replace(profile, task_target=_normalize_task_target_value(target)).normalized()
    presets_dir = get_augmentation_presets_dir(_profile_task_target(profile))
    stem = _safe_preset_stem(name, fallback=f"aug_{_profile_task_target(profile)}")
    path = presets_dir / f"{stem}.json"
    payload = build_augmentation_preset_payload(
        profile,
        name=name or stem,
        description=description,
        source_dataset_dir=source_dataset_dir,
    )
    _write_json_file(path, payload)
    return path


def load_augmentation_preset(path: Path, *, target: str | None = None) -> AugmentationProfile:
    payload = safe_load_yaml(Path(path)) if str(path).lower().endswith((".yaml", ".yml")) else None
    if payload is None:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    return augmentation_profile_from_mapping(payload, target=target)


def iter_augmentation_preset_files(target: str | None = None) -> list[Path]:
    dirs = [get_augmentation_presets_dir(target) if target else get_augmentation_presets_dir()]
    if not target:
        dirs.extend([
            get_augmentation_presets_dir("char"),
            get_augmentation_presets_dir("plate"),
        ])
    files: list[Path] = []
    seen: set[str] = set()
    for presets_dir in dirs:
        for path in sorted(presets_dir.glob("*.json")):
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def augment_yolo_dataset_train_split(
    dataset_dir: Path,
    profile: AugmentationProfile,
    progress_callback: Callable[[int, int, str], None] | None = None,
    balance_plan: Any = None,
    max_augmented_variants_per_source: int | None = None,
) -> tuple[bool, str, dict]:
    """Create additional augmented samples in images/train and labels/train."""
    dataset_dir = Path(dataset_dir)
    profile = (profile or AugmentationProfile()).normalized()
    randomness_mode = normalize_augmentation_randomness_mode(getattr(profile, "randomness_mode", "realistic"))
    profile_jitter_label = f"per_sample_{randomness_mode}"
    stats = {
        "enabled": bool(profile.enabled),
        "generated": 0,
        "skipped": 0,
        "train_before": 0,
        "train_after": 0,
        "sample_pool": 0,
        "profile_jitter": profile_jitter_label,
        "balance_plan_enabled": False,
        "max_augmented_variants_per_source": None,
        "generated_by_source": {},
        "requested": int(profile.extra_count or 0),
        "attempts": 0,
        "completion_ok": False,
        "completion_status": "PENDING",
        "stop_reason": "",
    }

    if not profile.enabled or profile.extra_count <= 0:
        stats["completion_ok"] = True
        stats["completion_status"] = "SKIPPED"
        return True, "Augmentacja train pominięta.", stats

    if not CV2_AVAILABLE or cv2 is None:
        stats["completion_status"] = "FAILED"
        return False, "OpenCV jest niedostępny, więc augmentacja obrazów nie może zostać wykonana.", stats

    if not is_albumentations_available():
        stats["completion_status"] = "FAILED"
        return False, "Albumentations nie jest zainstalowane. Zainstaluj pakiet albumentations albo wyłącz augmentację.", stats

    config = _load_dataset_config(dataset_dir)
    kpt_count, kpt_dim = _parse_kpt_shape(config)
    has_keypoints = kpt_count > 0 and kpt_dim >= 2
    items = _discover_train_items(dataset_dir)
    stats["train_before"] = len(items)
    if not items:
        stats["completion_status"] = "FAILED"
        return False, "Brak obrazów z etykietami w train, nie ma czego augmentować.", stats

    rng = random.Random(profile.seed)
    pool, source_limits, initial_generated_by_source, balance_stats = _build_balance_augmented_source_state(
        dataset_dir,
        items,
        balance_plan=balance_plan,
        max_augmented_variants_per_source=max_augmented_variants_per_source,
    )
    stats.update(balance_stats)
    pool = _select_augmentation_sample_pool(
        rng,
        pool,
        profile.sample_size,
        balance_plan_enabled=bool(stats.get("balance_plan_enabled")),
    )
    stats["sample_pool"] = len(pool)
    if not pool:
        stats["completion_status"] = "FAILED"
        return False, "Losowa próbka train jest pusta.", stats

    try:
        _build_transform(profile, has_keypoints=has_keypoints)
    except Exception as exc:
        stats["completion_status"] = "FAILED"
        return False, f"Nie udało się przygotować pipeline Albumentations: {exc}", stats

    label_dir = dataset_dir / "labels" / "train"
    generated_files: list[dict] = []
    generated_by_source: dict[str, int] = {}
    reserved_stems = _collect_reserved_dataset_stems(dataset_dir)

    max_attempts = max(profile.extra_count * 10, profile.extra_count + len(pool) * 2)
    attempts = 0
    while stats["generated"] < profile.extra_count and attempts < max_attempts:
        attempts += 1
        if callable(progress_callback):
            progress_callback(stats["generated"], profile.extra_count, "augmentacja")

        selected_record = _select_balance_record(
            rng,
            pool,
            generated_by_source=generated_by_source,
            initial_generated_by_source=initial_generated_by_source,
            source_limits=source_limits,
        )
        if selected_record is None:
            stats["stop_reason"] = (
                "source_reuse_safety_limit"
                if bool(stats.get("balance_plan_enabled"))
                else "wyczerpano limit kandydatów albo kopii z jednego źródła"
            )
            break
        image_path = Path(selected_record["image_path"])
        label_path = Path(selected_record["label_path"])
        source_key = str(selected_record.get("source_key") or label_path.stem)
        sample_seed = int(rng.randint(1, 2_147_483_647))
        sample_profile = _jitter_augmentation_profile(profile, random.Random(sample_seed))
        image = cv2.imread(str(image_path))
        if image is None:
            stats["skipped"] += 1
            continue

        height, width = image.shape[:2]
        objects = _parse_yolo_label(label_path, kpt_count=kpt_count, kpt_dim=kpt_dim)
        if not objects:
            stats["skipped"] += 1
            continue

        if has_keypoints and any(len(obj.keypoints) != kpt_count for obj in objects):
            stats["skipped"] += 1
            continue

        bboxes, class_labels, keypoints, visibility_groups = _objects_to_albumentations_payload(
            objects,
            width=width,
            height=height,
        )

        try:
            transform = _build_transform(sample_profile, has_keypoints=has_keypoints)
            _seed_transform(transform, sample_seed)
            augmented = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
                keypoints=keypoints if has_keypoints else [],
            )
        except Exception as exc:
            logger.debug(f"Augmentacja pominięta dla {image_path.name}: {exc}")
            stats["skipped"] += 1
            continue

        augmented_bboxes = list(augmented.get("bboxes") or [])
        augmented_labels = list(augmented.get("class_labels") or [])
        augmented_keypoints = list(augmented.get("keypoints") or [])
        if len(augmented_bboxes) != len(objects) or len(augmented_labels) != len(objects):
            stats["skipped"] += 1
            continue
        if has_keypoints and len(augmented_keypoints) != len(keypoints):
            stats["skipped"] += 1
            continue

        label_lines = _format_augmented_labels(
            bboxes=augmented_bboxes,
            class_labels=augmented_labels,
            keypoints=augmented_keypoints,
            visibility_groups=visibility_groups,
            width=width,
            height=height,
            kpt_count=kpt_count,
            kpt_dim=kpt_dim,
        )
        if not label_lines:
            stats["skipped"] += 1
            continue

        out_img, out_lbl = _unique_augmented_paths(image_path, label_dir, stats["generated"] + 1, reserved_stems)
        augmented_image = _apply_albumentations_image_effects(
            augmented["image"],
            sample_profile,
            seed=(sample_seed + 31337) % 2_147_483_647,
        )
        augmented_image = _apply_image_postprocess(
            augmented_image,
            sample_profile,
            random.Random((sample_seed + 7919) % 2_147_483_647 or 1),
        )
        if not cv2.imwrite(str(out_img), augmented_image):
            stats["skipped"] += 1
            continue
        out_lbl.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        stats["generated"] += 1
        generated_by_source[source_key] = int(generated_by_source.get(source_key, 0) or 0) + 1
        generated_files.append(
            {
                "image": str(out_img.relative_to(dataset_dir)),
                "label": str(out_lbl.relative_to(dataset_dir)),
                "source_image": str(image_path.relative_to(dataset_dir)),
                "source_label": str(label_path.relative_to(dataset_dir)),
                "source_key": source_key,
                "augmentation_seed": sample_seed,
                "randomness_mode": randomness_mode,
            }
        )

    stats["attempts"] = attempts
    if stats["generated"] < profile.extra_count and not str(stats.get("stop_reason") or "").strip():
        stats["stop_reason"] = "wyczerpano limit prób bez uzyskania żądanej liczby obrazów"
    stats["train_after"] = stats["train_before"] + stats["generated"]
    stats["generated_by_source"] = dict(sorted(generated_by_source.items()))
    stats["total_generated_variants_by_source"] = {
        key: int(initial_generated_by_source.get(key, 0) or 0) + int(generated_by_source.get(key, 0) or 0)
        for key in sorted(set(initial_generated_by_source) | set(generated_by_source))
    }
    profile_payload = _build_augmentation_profile_payload(dataset_dir, profile, stats, generated_files)
    _write_manifest(dataset_dir, profile_payload)
    _write_augmentation_profile_files(dataset_dir, profile_payload)

    if callable(progress_callback):
        progress_callback(stats["generated"], profile.extra_count, "augmentacja zakończona")

    ok, message = _finalize_train_augmentation_result(stats, profile.extra_count)
    stats["completion_ok"] = bool(ok)
    if ok:
        stats["completion_status"] = "COMPLETED"
    elif int(stats.get("generated", 0) or 0) > 0:
        stats["completion_status"] = "PARTIAL"
    else:
        stats["completion_status"] = "NO_OUTPUT"
    return ok, message, stats
