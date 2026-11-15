from dataclasses import replace
import random
from unittest.mock import Mock

import numpy as np
import pytest

from auto_annotation_tool.training import dataset_augmentation as aug


def scene_profile(**changes):
    defaults = dict(
        task_target="plate", rain_strength=0.65, rain_edge_mist_strength=0.8,
        rain_drop_size_min=0.3, rain_drop_size_max=0.3,
        tyndall_strength=0.0, dark_relief_strength=0.0, night_light_strength=0.0,
        dirt_flow_wind_strength=0.0,
    )
    defaults.update(changes)
    return aug.AugmentationProfile(**defaults).normalized()


def scene_image(kind="horizontal"):
    image = np.full((128, 192, 3), 32, dtype=np.uint8)
    if kind == "horizontal":
        image[64:] = 200
    elif kind == "vertical":
        image[:, 96:] = 200
    return image


@pytest.fixture(params=["cpu", "cuda"])
def render(request):
    if request.param == "cpu":
        return lambda image, profile: aug._apply_image_postprocess(image, profile, random.Random(123))
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")

    def render_cuda(image, profile):
        out, backend = aug._apply_mt_scene_postprocess_cuda(image, profile, seed=123)
        assert backend == "CUDA/Torch scene fastpath"
        return out
    return render_cuda


def test_cuda_eligibility_allows_collision_and_preserves_other_guards(monkeypatch):
    monkeypatch.setattr(aug, "is_cuda_available", lambda: True)
    profile = scene_profile()
    assert aug._mt_scene_cuda_fastpath_eligible(profile)
    assert not aug._mt_scene_cuda_fastpath_eligible(replace(profile, task_target="char"))
    for field in ("tyndall_strength", "water_film_strength", "dark_relief_strength", "wet_reflection_strength"):
        assert not aug._mt_scene_cuda_fastpath_eligible(replace(profile, **{field: 0.5}))
    monkeypatch.setattr(aug, "is_cuda_available", lambda: False)
    assert not aug._mt_scene_cuda_fastpath_eligible(profile)


@pytest.mark.parametrize("kind, expect_collision", [("horizontal", True), ("vertical", False), ("flat", False)])
def test_collision_requires_an_edge_facing_the_rain(render, kind, expect_collision):
    image = scene_image(kind)
    original = image.copy()
    profile = scene_profile()
    enabled = render(image, profile)
    disabled = render(image, replace(profile, rain_edge_mist_strength=0.0))
    assert enabled.shape == image.shape and enabled.dtype == np.uint8
    assert np.array_equal(image, original)
    assert bool(np.any(enabled != disabled)) == expect_collision
    if expect_collision:
        assert np.array_equal(enabled[:32], disabled[:32])
        assert np.array_equal(enabled[96:], disabled[96:])


def test_collision_is_off_when_rain_is_off(render):
    image = scene_image()
    profile = scene_profile(rain_strength=0.0)
    assert np.array_equal(render(image, profile), image)


def test_splash_radius_changes_rendering(render):
    image = scene_image()
    profile = scene_profile(rain_edge_mist_radius=0.05)
    assert np.any(render(image, profile) != render(image, replace(profile, rain_edge_mist_radius=0.9)))


def test_collision_combines_with_lens_noise_and_wind(render):
    image = scene_image()
    profile = scene_profile(rain_lens_strength=0.4, noise_strength=0.1, dirt_flow_wind_strength=0.6)
    out = render(image, profile)
    assert np.any(out != render(image, replace(profile, rain_edge_mist_strength=0.0)))


def test_scene_edge_sensitivity_detects_weaker_edges():
    image = scene_image()
    image[32:64] = 52
    low = aug._build_scene_rain_edge_gate(image, scene_profile(contour_detection_sensitivity=0.0))
    high = aug._build_scene_rain_edge_gate(image, scene_profile(contour_detection_sensitivity=1.0))
    assert np.all(high >= low)
    assert np.any(high[31:33] > low[31:33])


@pytest.mark.parametrize("crosses_edge", [False, True])
def test_cpu_splash_requires_a_drop_crossing_the_edge(monkeypatch, crosses_edge):
    monkeypatch.setattr(aug, "_iter_stream_rain_anchors", lambda *args, **kwargs: [(80, 62 if crosses_edge else 10)])
    profile = scene_profile(rain_drop_size_min=0.8, rain_drop_size_max=0.8)
    image = scene_image()
    on = aug._apply_rain_effect(image, profile, random.Random(123))
    off = aug._apply_rain_effect(image, replace(profile, rain_edge_mist_strength=0), random.Random(123))
    assert bool(np.any(on != off)) == crosses_edge


def test_cpu_scene_and_debug_do_not_use_plate_symbol_geometry(monkeypatch):
    builder = Mock(wraps=aug._build_symbol_contour_height_map)
    monkeypatch.setattr(aug, "_build_symbol_contour_height_map", builder)
    image = scene_image()
    out, debug = aug._apply_image_postprocess(image, scene_profile(), random.Random(123), return_debug=True)
    builder.assert_not_called()
    assert np.any(out != image)
    assert np.any(debug["rain_edge_debug_mask"] > 0)
    aug._apply_image_postprocess(image, scene_profile(task_target="char"), random.Random(123))
    assert builder.call_count > 0


def test_cpu_fallback_keeps_collisions(monkeypatch):
    monkeypatch.setattr(aug, "is_cuda_available", lambda: False)
    image = scene_image()
    profile = scene_profile()
    out, backend = aug._apply_mt_scene_postprocess_cuda(image, profile, seed=123)
    off, _ = aug._apply_mt_scene_postprocess_cuda(image, replace(profile, rain_edge_mist_strength=0), seed=123)
    assert backend == "CPU/OpenCV/NumPy"
    assert np.any(out != off)
