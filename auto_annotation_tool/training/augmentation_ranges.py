"""Explicit manual deviations, separate from the legacy percentage sampler."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import math
import random

RANGE_SCHEMA = "augmentation_manual_ranges_v2"
GROUP_LABELS = {
    "geometry": "Geometria", "color": "Kolor i tonalność", "weather": "Pogoda",
    "wind": "Wiatr", "illumination": "Światło", "material": "Powierzchnia",
    "relief": "Relief i kontury", "dirt": "Błoto i zabrudzenia", "sensor": "Kamera i optyka",
}


@dataclass(frozen=True)
class RangeSpec:
    key: str
    label: str
    group: str
    fields: tuple[str, ...]
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    targets: tuple[str, ...] = ("plate", "char")
    requires: tuple[str, ...] = ()
    advanced: bool = False
    integer: bool = False
    angle: bool = False
    unit: str = ""


def _spec(key, label, group, **kwargs):
    return RangeSpec(key, label, group, kwargs.pop("fields", (key,)), **kwargs)


SPECS = [
    _spec("rotation_limit", "Limit obrotu", "geometry", minimum=-15, maximum=15, step=.1, targets=("plate",), unit="°"),
    _spec("translate_limit", "Limit przesunięcia", "geometry", maximum=.03, step=.001, targets=("plate",)),
    _spec("scale_limit", "Limit zmiany skali", "geometry", maximum=.08, step=.001, targets=("plate",)),
    _spec("brightness_limit", "Jasność", "color", maximum=.25),
    _spec("contrast_limit", "Kontrast", "color", maximum=.25),
    _spec("saturation_limit", "Nasycenie", "color", maximum=3),
    _spec("rain_strength", "Gęstość deszczu", "weather", requires=("rain_strength",)),
    _spec("rain_drop_size", "Rozmiar kropli · min–max", "weather", fields=("rain_drop_size_min", "rain_drop_size_max"), requires=("rain_strength",)),
    _spec("rain_alpha", "Widoczność kropli", "weather", requires=("rain_strength",)),
    _spec("rain_edge_mist_strength", "Rozbryzg na konturach", "weather", requires=("rain_strength", "rain_edge_mist_strength")),
    _spec("rain_edge_mist_radius", "Zasięg rozbryzgu", "weather", requires=("rain_strength", "rain_edge_mist_strength")),
    _spec("rain_lens_strength", "Soczewka kropli", "weather", requires=("rain_strength", "rain_lens_strength"), advanced=True),
    _spec("rain_vector_field_strength", "Pole deszczu", "weather", requires=("rain_strength", "rain_vector_field_strength"), advanced=True),
    _spec("rain_vortex_strength", "Wiry deszczu", "weather", requires=("rain_strength", "rain_vortex_strength"), advanced=True),
    _spec("tyndall_strength", "Efekt Tyndalla", "weather", targets=("char",), requires=("rain_strength", "tyndall_strength"), advanced=True),
    _spec("dirt_flow_wind_strength", "Siła wiatru", "wind"),
    _spec("dirt_flow_air_angle", "Kierunek wiatru", "wind", minimum=-180, maximum=180, step=1, angle=True, unit="°"),
    _spec("night_strength", "Przyciemnienie sceny", "illumination", requires=("night_strength",)),
    _spec("night_light_strength", "Światło bazowe", "illumination", requires=("night_light_strength",)),
    _spec("night_light_warmth", "Temperatura światła", "illumination", requires=("night_light_strength",)),
    _spec("scene_camera_z", "Odległość kamery", "illumination", minimum=.35, maximum=4, targets=("char",), advanced=True),
    _spec("relief_bounce_depth", "Liczba odbić światła", "illumination", minimum=1, maximum=4, step=1, integer=True, targets=("char",), requires=("dark_relief_strength", "night_light_strength"), advanced=True),
    _spec("relief_bounce_strength", "Siła odbić światła", "illumination", targets=("char",), requires=("dark_relief_strength", "night_light_strength"), advanced=True),
    _spec("overhang_shadow_strength", "Cień daszka", "illumination", targets=("char",), requires=("overhang_shadow_strength",), advanced=True),
    _spec("overhang_shadow_depth", "Zasięg cienia", "illumination", maximum=.6, targets=("char",), requires=("overhang_shadow_strength",), advanced=True),
    _spec("overhang_shadow_skew", "Skos cienia", "illumination", minimum=-1, targets=("char",), requires=("overhang_shadow_strength",), advanced=True),
    _spec("water_film_strength", "Film wodny", "material", targets=("char",), requires=("water_film_strength",)),
    *[_spec(key, label, "material", targets=("char",), requires=("water_film_strength",), advanced=True) for key, label in (
        ("water_film_unevenness", "Nierówność filmu"), ("water_film_lens_strength", "Soczewka filmu"),
        ("water_film_contour_response", "Reakcja filmu na kontury"), ("water_film_gloss_strength", "Połysk filmu"))],
    *[_spec(key, label, "material", maximum=2, targets=("char",), requires=(key,)) for key, label in (
        ("plate_reflect_gradient_strength", "Cieniowanie tablicy"), ("plate_reflect_glare_strength", "Odblask tablicy"),
        ("plate_reflect_curve_strength", "Krzywizna tablicy"))],
    _spec("dark_relief_strength", "Wypukłość konturów", "relief", maximum=3, targets=("char",), requires=("dark_relief_strength",)),
    _spec("wet_mud_gloss_strength", "Połysk błota", "dirt", targets=("char",), requires=("wet_mud_gloss_strength",)),
    _spec("dirt_flow_points", "Liczba smug błota", "dirt", maximum=2000, step=1, integer=True, targets=("char",)),
    _spec("dirt_flow_mass", "Masa błota · min–max", "dirt", fields=("dirt_flow_mass_min", "dirt_flow_mass_max"), minimum=.05, maximum=2.8, targets=("char",)),
    _spec("dirt_flow_stickiness_range", "Lepkość · min–max", "dirt", fields=("dirt_flow_stickiness_min", "dirt_flow_stickiness_max"), targets=("char",)),
    _spec("dirt_flow_opacity", "Krycie · min–max", "dirt", fields=("dirt_flow_opacity_min", "dirt_flow_opacity_max"), targets=("char",)),
    *[_spec(key, label, "dirt", targets=("char",), advanced=True) for key, label in (
        ("dirt_flow_trail_length", "Długość smug"), ("dirt_flow_humidity", "Wilgotność błota"), ("vehicle_speed", "Ruch pojazdu"))],
    _spec("noise_strength", "Szum", "sensor", maximum=.08, step=.001, requires=("noise_strength",)),
    _spec("noise_grain_size", "Ziarno szumu", "sensor", minimum=1, maximum=12, step=1, integer=True, requires=("noise_strength",), advanced=True),
    _spec("blur_strength", "Rozmycie", "sensor", requires=("blur_strength",)),
    *[_spec(key, label, "sensor", requires=(key,), advanced=True) for key, label in (
        ("night_iso_noise_strength", "Szum ISO"), ("night_bloom_strength", "Poświata"),
        ("flare_strength", "Flara kamery"), ("overexposure_strength", "Prześwietlenie"))],
]
for _index in (1, 2, 3):
    _prefix = "traffic_headlight" if _index == 1 else f"traffic_headlight_{_index}"
    _power = f"{_prefix}_strength"
    SPECS.append(_spec(_power, f"Reflektor R{_index} · moc", "illumination", targets=("char",), requires=(_power,), advanced=True))
    for _axis in ("x", "y", "z"):
        SPECS.append(_spec(f"{_prefix}_source_world_{_axis}", f"R{_index} · położenie {_axis.upper()}", "illumination",
                           minimum=0 if _axis == "z" else -5, maximum=8 if _axis == "z" else 5,
                           targets=("char",), requires=(_power,), advanced=True))
    SPECS.append(_spec(f"traffic_headlight_{_index}_cone", f"R{_index} · stożek światła", "illumination",
                       minimum=.02, maximum=2.5, targets=("char",), requires=(_power,), advanced=True))
    SPECS.append(_spec(f"traffic_headlight_{_index}_source_radius", f"R{_index} · promień źródła", "illumination",
                       maximum=2.5, targets=("char",), requires=(_power,), advanced=True))
SPEC_BY_KEY = {spec.key: spec for spec in SPECS}


def is_range_config(value):
    return isinstance(value, dict) and value.get("schema") == RANGE_SCHEMA


def _number(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def normalize_range_config(value, target, normalize_legacy):
    legacy = value.get("legacy", {})
    if is_range_config(legacy):
        legacy = {}
    overrides = {}
    raw = value.get("overrides", {})
    if isinstance(raw, dict):
        for key, state in raw.items():
            spec = SPEC_BY_KEY.get(key)
            if spec is None or target not in spec.targets or not isinstance(state, dict):
                continue
            ceiling = 180 if spec.angle else spec.maximum - spec.minimum
            delta = max(0.0, min(ceiling, _number(state.get("delta"))))
            if spec.integer:
                delta = float(round(delta))
            overrides[key] = {"enabled": bool(state.get("enabled")) and delta > 0, "delta": delta}
    return {"schema": RANGE_SCHEMA, "target": target, "legacy": normalize_legacy(legacy, target), "overrides": overrides}


def unavailable_reason(spec, profile):
    target = getattr(profile, "task_target", "char")
    if target not in spec.targets:
        return "Dostępne tylko dla " + ("wycinka tablicy (MZ)." if spec.targets == ("char",) else "pełnej sceny (MT).")
    if any(_number(getattr(profile, key, 0)) < spec.minimum for key in spec.fields):
        return "Położenie jest automatyczne. Najpierw ustaw je w profilu."
    if any(_number(getattr(profile, key, 0)) <= .001 for key in spec.requires):
        return "Efekt wyłączony w profilu. Włącz go w ustawieniach efektów."
    if spec.group in {"dirt", "wind"}:
        dirt = target == "char" and (getattr(profile, "dirt_flow_points", 0) > 0 or getattr(profile, "dirt_flow_strength", 0) > .001)
        rain = getattr(profile, "rain_strength", 0) > .001
        if not dirt and (spec.group == "dirt" or not rain):
            return "Włącz zabrudzenia w profilu." if spec.group == "dirt" else "Wiatr wymaga włączonego deszczu lub zabrudzeń."
    if spec.angle and getattr(profile, "dirt_flow_wind_strength", 0) <= .001:
        return "Wiatr jest wyłączony w profilu."
    if spec.key.startswith("traffic_headlight"):
        index = 2 if "headlight_2" in spec.key else 3 if "headlight_3" in spec.key else 1
        if int(getattr(profile, "traffic_headlight_count", 0)) < index:
            return f"Reflektor R{index} jest wyłączony w profilu."
    return ""


def range_bounds(spec, profile, delta):
    """Scalar bounds, or a common offset for a range pair / circular angle."""
    values = [_number(getattr(profile, key)) for key in spec.fields]
    if spec.angle:
        return -delta, delta
    if len(values) > 1:
        return max(-delta, spec.minimum - min(values)), min(delta, spec.maximum - max(values))
    return max(spec.minimum, values[0] - delta), min(spec.maximum, values[0] + delta)


def sample_explicit_ranges(base, legacy_sample, config, seed):
    updates = {}
    for key, state in config.get("overrides", {}).items():
        spec = SPEC_BY_KEY[key]
        # An override always replaces the legacy choice, even when the effect is off.
        updates.update({field: getattr(base, field) for field in spec.fields})
        if not state["enabled"] or unavailable_reason(spec, base):
            continue
        low, high = range_bounds(spec, base, state["delta"])
        if high <= low:
            sampled = low
        else:
            digest = hashlib.sha256(f"{seed}:{key}".encode("ascii")).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
            center = 0 if spec.angle or len(spec.fields) > 1 else float(getattr(base, key))
            sampled = rng.triangular(low, high, max(low, min(high, center)))
        if spec.integer:
            sampled = max(math.ceil(low), min(math.floor(high), round(sampled)))
        for field in spec.fields:
            value = float(getattr(base, field)) + sampled if len(spec.fields) > 1 or spec.angle else sampled
            if spec.angle:
                value = (value + 180) % 360 - 180
            updates[field] = int(value) if spec.integer else value
    return replace(legacy_sample, **updates, manual_randomness=config).normalized()


def legacy_amount(config, field):
    for group in config.get("groups", {}).values():
        state = group.get("fields", {}).get(field, {})
        if group.get("enabled") and state.get("enabled"):
            return _number(group.get("amount")) * _number(state.get("amount")) / 100
    return 0.0


class RangeDraft:
    """A detached edit transaction. Untouched legacy presets remain byte-equivalent data."""
    def __init__(self, profile):
        self.profile = profile.normalized()
        self.original = deepcopy(self.profile.manual_randomness)
        self.legacy = deepcopy(self.original.get("legacy", {}) if is_range_config(self.original) else self.original)
        self.overrides = deepcopy(self.original.get("overrides", {}) if is_range_config(self.original) else {})

    def state(self, spec):
        if spec.key in self.overrides:
            state = self.overrides[spec.key]
            return ("range" if state["enabled"] else "fixed"), state["delta"]
        amounts = [legacy_amount(self.legacy, field) for field in spec.fields]
        return ("legacy" if any(amounts) else "fixed"), self.default_delta(spec)

    def default_delta(self, spec):
        value = max(abs(float(getattr(self.profile, key))) for key in spec.fields)
        delta = max(spec.step, min((spec.maximum - spec.minimum) / 4, value * .1))
        return float(max(1, round(delta))) if spec.integer else delta

    def set(self, spec, enabled, delta):
        if unavailable_reason(spec, self.profile):
            raise ValueError(unavailable_reason(spec, self.profile))
        try:
            value = float(str(delta).replace(",", "."))
        except (TypeError, ValueError):
            raise ValueError("Podaj liczbę; możesz użyć przecinka jako separatora.") from None
        ceiling = 180 if spec.angle else spec.maximum - spec.minimum
        if not math.isfinite(value) or not 0 <= value <= ceiling:
            raise ValueError(f"Podaj odchylenie od 0 do {ceiling:g}.")
        if spec.integer and value != round(value):
            raise ValueError("Ten parametr wymaga liczby całkowitej.")
        self.overrides[spec.key] = {"enabled": bool(enabled) and value > 0, "delta": value}

    @property
    def changed(self):
        initial = self.original.get("overrides", {}) if is_range_config(self.original) else {}
        return self.overrides != initial

    def export(self):
        if not self.changed:
            return deepcopy(self.original)
        return {"schema": RANGE_SCHEMA, "target": self.profile.task_target,
                "legacy": deepcopy(self.legacy), "overrides": deepcopy(self.overrides)}
