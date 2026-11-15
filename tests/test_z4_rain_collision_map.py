from dataclasses import replace
import random

import numpy as np
import pytest

from auto_annotation_tool.training import dataset_augmentation as aug


def noisy_rain_profile():
    return aug.AugmentationProfile(
        task_target="plate", rain_strength=.65, rain_edge_mist_strength=.7,
        rain_edge_mist_radius=.66, contour_detection_sensitivity=.81,
        dirt_flow_wind_strength=.9, dirt_flow_air_angle=-120,
        noise_strength=.3, night_light_strength=.35,
        dark_relief_strength=0, tyndall_strength=0,
    ).normalized()


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_zero_to_positive_splash_does_not_stop_rain_on_added_noise(backend):
    scene = np.full((128, 192, 3), 100, dtype=np.uint8)
    profile = noisy_rain_profile()
    assert not np.any(aug.build_rain_edge_debug_mask(scene, profile))
    if backend == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        def render(profile):
            result, engine = aug._apply_mt_scene_postprocess_cuda(scene, profile, seed=123)
            assert engine == "CUDA/Torch scene fastpath"
            return result
    else:
        def render(profile):
            return aug._apply_image_postprocess(scene, profile, random.Random(123))
    disabled = render(replace(profile, rain_edge_mist_strength=0))
    enabled = render(profile)
    assert np.any(disabled != scene)
    assert np.array_equal(enabled, disabled)


def test_cuda_collision_geometry_uses_the_scene_before_weather(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    scene = np.full((128, 192, 3), 60, dtype=np.uint8)
    scene[64:] = 200
    original = scene.copy()
    sources = []
    builder = aug._rain_collision_geometry_cuda
    def capture(torch, image, sensitivity):
        sources.append(image.cpu().numpy())
        return builder(torch, image, sensitivity)
    monkeypatch.setattr(aug, "_rain_collision_geometry_cuda", capture)
    _, backend = aug._apply_mt_scene_postprocess_cuda(scene, noisy_rain_profile(), seed=123)
    assert backend == "CUDA/Torch scene fastpath"
    assert len(sources) == 1
    assert np.array_equal(sources[0], original)
    assert np.array_equal(scene, original)


def test_preview_overlay_matches_cuda_collision_map_with_noise_and_night(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not aug.is_albumentations_available():
        pytest.skip("CUDA and Albumentations required")
    scene = np.full((128, 192, 3), 60, dtype=np.uint8)
    scene[64:] = 200
    path = tmp_path / "scene.png"
    aug.cv2.imwrite(str(path), scene)
    gates = []
    clip_paths = aug._clip_rain_paths_at_first_contact
    def capture(*args):
        gates.append(args[-1][0].cpu().numpy())
        return clip_paths(*args)
    monkeypatch.setattr(aug, "_clip_rain_paths_at_first_contact", capture)
    ok, message, preview = aug.preview_augmentation_image(path, noisy_rain_profile(), include_contours=True)
    assert ok, message
    assert "CUDA/Torch scene fastpath" in message
    assert len(gates) == 1
    assert np.array_equal(gates[0] > .05, preview["rain_edge_debug_mask"] > 0)
