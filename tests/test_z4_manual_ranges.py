from dataclasses import asdict, replace
import json
import random

import numpy as np
import pytest

from auto_annotation_tool.training import dataset_augmentation as aug
from auto_annotation_tool.training.augmentation_ranges import (
    RANGE_SCHEMA, SPECS, SPEC_BY_KEY, RangeDraft, unavailable_reason,
)


def profile(**kwargs):
    values = dict(task_target="plate", randomness_mode="manual", rain_strength=.6,
                  rain_edge_mist_strength=.5, night_light_strength=0, dark_relief_strength=0,
                  tyndall_strength=0, seed=42)
    values.update(kwargs)
    return aug.AugmentationProfile(**values).normalized()


def explicit(base, **deviations):
    draft = RangeDraft(base)
    for key, delta in deviations.items():
        draft.set(SPEC_BY_KEY[key], delta > 0, delta)
    return replace(base, manual_randomness=draft.export()).normalized()


def test_scalar_stays_in_declared_range_and_is_repeatable():
    p = explicit(profile(), rain_strength=.1)
    first = [aug._jitter_augmentation_profile(p, random.Random(i)).rain_strength for i in range(100)]
    assert min(first) >= .5 and max(first) <= .7
    assert len(set(first)) > 90
    assert first == [aug._jitter_augmentation_profile(p, random.Random(i)).rain_strength for i in range(100)]
    changed = explicit(p, rain_edge_mist_strength=.2)
    assert first == [aug._jitter_augmentation_profile(changed, random.Random(i)).rain_strength for i in range(100)]


@pytest.mark.parametrize("target", ["plate", "char"])
def test_opening_and_applying_untouched_legacy_does_not_migrate(target):
    config = aug.default_manual_randomness_config(target)
    config["groups"]["weather"].update(enabled=True, amount=50)
    config["groups"]["weather"]["fields"]["rain_strength"].update(enabled=True, amount=50)
    base = profile(task_target=target, manual_randomness=config)
    draft = RangeDraft(base)
    assert draft.state(SPEC_BY_KEY["rain_strength"])[0] == "legacy"
    assert draft.export() == base.manual_randomness
    assert not draft.changed
    wrapped = replace(base, manual_randomness={"schema": RANGE_SCHEMA, "legacy": config, "overrides": {}})
    for seed in range(10):
        old = asdict(aug._jitter_augmentation_profile(base, random.Random(seed)))
        new = asdict(aug._jitter_augmentation_profile(wrapped, random.Random(seed)))
        old.pop("manual_randomness"); new.pop("manual_randomness")
        assert old == new


def test_editing_one_field_preserves_all_other_legacy_samples():
    config = aug.default_manual_randomness_config("plate")
    for group in config["groups"].values():
        group.update(enabled=True, amount=50)
        for field in group["fields"].values():
            field.update(enabled=True, amount=70)
    base = profile(manual_randomness=config, noise_strength=.02)
    changed = explicit(base, rain_strength=.1)
    for seed in range(10):
        old = asdict(aug._jitter_augmentation_profile(base, random.Random(seed)))
        new = asdict(aug._jitter_augmentation_profile(changed, random.Random(seed)))
        for key in ("manual_randomness", "rain_strength"):
            old.pop(key); new.pop(key)
        assert old == new


@pytest.mark.parametrize("key,fields,kwargs", [
    ("rain_drop_size", ("rain_drop_size_min", "rain_drop_size_max"), dict()),
    ("dirt_flow_mass", ("dirt_flow_mass_min", "dirt_flow_mass_max"), dict(task_target="char", dirt_flow_points=20)),
    ("dirt_flow_opacity", ("dirt_flow_opacity_min", "dirt_flow_opacity_max"), dict(task_target="char", dirt_flow_points=20)),
    ("dirt_flow_stickiness_range", ("dirt_flow_stickiness_min", "dirt_flow_stickiness_max"), dict(task_target="char", dirt_flow_points=20)),
])
def test_pair_moves_together_and_never_crosses(key, fields, kwargs):
    base = profile(**kwargs)
    p = explicit(base, **{key: .2})
    spec = SPEC_BY_KEY[key]
    width = getattr(base, fields[1])-getattr(base, fields[0])
    for seed in range(50):
        sample = aug._jitter_augmentation_profile(p, random.Random(seed))
        low, high = (getattr(sample, field) for field in fields)
        assert spec.minimum <= low <= high <= spec.maximum
        assert high-low == pytest.approx(width)


def test_wind_is_available_for_mt_and_angle_wraps():
    p = explicit(profile(dirt_flow_air_angle=179), dirt_flow_air_angle=15, dirt_flow_wind_strength=.1)
    angles = [aug._jitter_augmentation_profile(p, random.Random(i)).dirt_flow_air_angle for i in range(60)]
    assert any(a < -170 for a in angles) and any(a > 170 for a in angles)
    assert all(abs((angle-179+180) % 360-180) <= 15 for angle in angles)


def test_integer_deviation_and_off_toggle():
    p = explicit(profile(task_target="char", dark_relief_strength=.3, night_light_strength=.5, relief_bounce_depth=2), relief_bounce_depth=1)
    values = [aug._jitter_augmentation_profile(p, random.Random(i)).relief_bounce_depth for i in range(50)]
    assert set(values) == {1, 2, 3}
    fixed = explicit(p, relief_bounce_depth=0)
    assert aug._jitter_augmentation_profile(fixed, random.Random(2)).relief_bounce_depth == 2


def test_unavailable_effect_cannot_be_enabled_and_mz_geometry_stays_fixed():
    base = profile(rain_strength=0)
    draft = RangeDraft(base)
    with pytest.raises(ValueError):
        draft.set(SPEC_BY_KEY["rain_strength"], True, .1)
    assert unavailable_reason(SPEC_BY_KEY["water_film_strength"], profile())
    p = profile(task_target="char", manual_randomness={"schema": RANGE_SCHEMA, "overrides": {"rotation_limit": {"enabled": True, "delta": 5}}})
    assert "rotation_limit" not in p.manual_randomness["overrides"]
    assert aug._jitter_augmentation_profile(p, random.Random(1)).rotation_limit == 0


@pytest.mark.parametrize("value", ["", "oops", "nan", "inf", "-1", "1.01"])
def test_invalid_entry_does_not_mutate_draft(value):
    draft = RangeDraft(profile())
    before = draft.export()
    with pytest.raises(ValueError):
        draft.set(SPEC_BY_KEY["rain_strength"], True, value)
    assert draft.export() == before


def test_comma_input_and_preset_roundtrip(tmp_path):
    draft = RangeDraft(profile())
    draft.set(SPEC_BY_KEY["rain_strength"], True, "0,12")
    p = replace(profile(), manual_randomness=draft.export()).normalized()
    payload = aug.build_augmentation_preset_payload(p)
    path = tmp_path / "ranges.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = aug.load_augmentation_preset(path, target="plate")
    assert restored.manual_randomness == p.manual_randomness
    assert aug.describe_manual_randomness_config(p.manual_randomness, "plate")["ranges"][0]["delta"] == .12


def test_catalog_matches_profile_fields_and_bounds():
    base = profile(task_target="char")
    assert len({s.key for s in SPECS}) == len(SPECS)
    for spec in SPECS:
        assert spec.maximum > spec.minimum
        assert all(hasattr(base, field) for field in spec.fields)
    assert SPEC_BY_KEY["dirt_flow_points"].maximum == aug.MAX_DIRT_FLOW_POINTS


def test_preview_uses_generator_sampler_and_stable_seeds(tmp_path, monkeypatch):
    if not aug.is_albumentations_available():
        pytest.skip("Albumentations unavailable")
    monkeypatch.setattr(aug, "is_cuda_available", lambda: False)
    image = np.full((48, 96, 3), 80, dtype=np.uint8)
    image[24:] = 190
    path = tmp_path / "próba_ł.png"
    aug.cv2.imencode(".png", image)[1].tofile(str(path))
    p = explicit(profile(), rain_strength=.1)
    ok, message, first = aug.preview_augmentation_image(path, p, sample_seed=123, max_pixels=10000)
    assert ok, message
    expected = aug._jitter_augmentation_profile(p, random.Random(123))
    assert first["sample_profile"] == asdict(expected)
    ok, message, second = aug.preview_augmentation_image(path, p, sample_seed=123, max_pixels=10000)
    assert ok, message
    assert np.array_equal(first["augmented_rgb"], second["augmented_rgb"])
    assert np.any(first["augmented_rgb"] != first["original_rgb"])


def test_empty_preview_source_returns_a_readable_error(tmp_path):
    if not aug.is_albumentations_available():
        pytest.skip("Albumentations unavailable")
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    ok, message, payload = aug.preview_augmentation_image(path, profile(), sample_seed=123)
    assert not ok and "odczytać" in message and payload == {}


@pytest.mark.parametrize("with_geometry", [False, True])
def test_generation_and_preview_use_the_same_parameters(tmp_path, monkeypatch, with_geometry):
    if not aug.is_albumentations_available():
        pytest.skip("Albumentations unavailable")
    monkeypatch.setattr(aug, "is_cuda_available", lambda: False)
    for rel in ("images/train", "labels/train"):
        (tmp_path / rel).mkdir(parents=True)
    source = tmp_path / "images/train/source.png"
    aug.cv2.imwrite(str(source), np.full((48, 96, 3), 100, dtype=np.uint8))
    (tmp_path / "labels/train/source.txt").write_text("0 0.5 0.5 0.4 0.3\n")
    (tmp_path / "data.yaml").write_text("names: [plate]\nnc: 1\n")
    p = explicit(profile(enabled=True, extra_count=2, sample_size=1), rain_strength=.1)
    if with_geometry:
        p = explicit(replace(p, rotation_limit=3), rotation_limit=1)
    seen = []
    original = aug._apply_mt_scene_postprocess_cuda
    def capture(image, sample, *, seed):
        seen.append((asdict(sample), seed))
        return original(image, sample, seed=seed)
    monkeypatch.setattr(aug, "_apply_mt_scene_postprocess_cuda", capture)
    ok, message, stats = aug.augment_yolo_dataset_train_split(tmp_path, p)
    assert ok, message
    assert stats["generated"] == 2
    payload = json.loads((tmp_path / "augmentation_profile.json").read_text(encoding="utf-8"))
    assert payload["profile"]["manual_randomness"]["schema"] == RANGE_SCHEMA
    generated = list(seen)
    for record, (sample, post_seed) in zip(payload["generated_files"], generated):
        seed = record["augmentation_seed"]
        ok, message, preview = aug.preview_augmentation_image(source, p, sample_seed=seed)
        assert ok, message
        assert preview["sample_profile"] == sample
        assert post_seed == ((seed + 7919) % 2_147_483_647 or 1)
        generated_image = aug.cv2.imread(str(tmp_path / record["image"]))
        assert np.array_equal(preview["augmented_rgb"], aug.cv2.cvtColor(generated_image, aug.cv2.COLOR_BGR2RGB))
