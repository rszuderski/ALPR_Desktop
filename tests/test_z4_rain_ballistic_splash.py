from dataclasses import replace
import random

import numpy as np
import pytest

from auto_annotation_tool.training import dataset_augmentation as aug


def splash_profile(**changes):
    fields = dict(task_target="plate", rain_strength=.65, rain_edge_mist_strength=.1,
                  rain_edge_mist_radius=.9, rain_drop_size_min=.9, rain_drop_size_max=.9,
                  rain_alpha=1, dirt_flow_wind_strength=0, night_light_strength=0,
                  dark_relief_strength=0)
    fields.update(changes)
    return aug.AugmentationProfile(**fields).normalized()


def test_fragments_rise_turn_and_fall_with_constant_downward_acceleration():
    time = np.linspace(0, 1.2, 121)
    x, y, free_side = aug._rain_splash_trajectory(0, -1, 1, .9, .9, 1, time)
    assert np.all(np.diff(x) > 0)
    assert np.allclose(np.diff(x, n=2), 0)
    assert np.diff(y)[0] < 0 < np.diff(y)[-1]
    assert np.all(np.diff(y, n=2) > 0)
    assert np.allclose(np.diff(y, n=2), np.diff(y, n=2)[0])
    assert y[50] < -5 and abs(y[100]) < 1e-6
    assert np.all(free_side[:101]) and not np.any(free_side[101:])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cpu_and_torch_render_curved_fragments_and_stop_at_the_next_wall(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    shape = (64, 80)
    gate = np.zeros(shape, dtype=np.float32)
    gate[40] = 1
    gate[:, 50] = 1
    geometry = (gate, np.zeros_like(gate), np.ones_like(gate))
    contacts = [(32, 40, 0, -1, 1, .9, .4)]
    cpu = aug._rain_contact_splash(shape, contacts, .9, geometry)
    unblocked = aug._rain_contact_splash(shape, contacts, .9)
    assert np.any(unblocked[:, 52:] > .01)
    assert not np.any(cpu[:, 52:] > 0)
    assert not np.any(cpu[42:] > 0)
    # Two limbs of the same arc occur at this height (ascending and descending).
    assert np.any(cpu[34:36, 35:40] > .01)
    assert np.any(cpu[34:36, 45:50] > .01)
    data = [torch.tensor([value], device=device, dtype=torch.float32) for value in contacts[0]]
    actual = aug._rain_contact_splash_cuda(
        torch, shape, torch.tensor([True], device=device), *data, .9,
        tuple(torch.from_numpy(value).to(device) for value in geometry),
    ).cpu().numpy()
    assert np.allclose(actual, cpu, atol=.001)
    empty = aug._rain_contact_splash_cuda(torch, shape, torch.tensor([False], device=device), *data, .9)
    assert not bool(empty.any())


@pytest.mark.parametrize("target", ["plate", "char"])
@pytest.mark.parametrize("background", [40, 250])
def test_cpu_splash_strength_preserves_primary_rain(monkeypatch, target, background):
    monkeypatch.setattr(aug, "_iter_stream_rain_anchors", lambda *args, **kwargs: [(40, 60), (80, 16)])
    scene = np.full((96, 96, 3), background, dtype=np.uint8)
    scene[64:] = 120
    profile = splash_profile(task_target=target)
    low, low_mask = aug._apply_rain_effect(scene, profile, random.Random(123), return_mask=True)
    high, high_mask = aug._apply_rain_effect(scene, replace(profile, rain_edge_mist_strength=1), random.Random(123), return_mask=True)
    assert np.any(low_mask > 0)
    assert np.array_equal(low_mask, high_mask)
    # The free-falling drop stays unchanged. At impact, overlapping fragments
    # must increase opacity instead of being erased by the incoming-rain mask.
    assert np.any(low_mask[:40, 70:] > 0)
    assert np.array_equal(low[:40, 70:], high[:40, 70:])
    assert np.any(low[55:65, 30:50] != high[55:65, 30:50])
    assert np.abs(high.astype(float) - scene).sum() > np.abs(low.astype(float) - scene).sum()


@pytest.mark.parametrize("background", [40, 250])
def test_cuda_splash_strength_preserves_primary_rain(monkeypatch, background):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    def fixed_anchor(low, high, size, **kwargs):
        hit = torch.arange(size[0], device=kwargs["device"]) % 2 == 0
        return torch.where(hit, 40 if high == 96 else 60, 80 if high == 96 else 16)
    monkeypatch.setattr(torch, "randint", fixed_anchor)
    masks = []
    blur = aug._torch_gaussian_blur_map
    def capture_blur(*args):
        result = blur(*args)
        masks.append(result.cpu().numpy())
        return result
    monkeypatch.setattr(aug, "_torch_gaussian_blur_map", capture_blur)
    scene = np.full((128, 96, 3), background, dtype=np.uint8)
    scene[64:] = 120
    profile = splash_profile()
    low = aug._apply_mt_scene_rain_cuda(scene, profile, torch, seed=123)
    low_mask = masks[0]
    masks.clear()
    high = aug._apply_mt_scene_rain_cuda(scene, replace(profile, rain_edge_mist_strength=1), torch, seed=123)
    high_mask = masks[0]
    assert np.array_equal(low_mask, high_mask)
    assert np.any(low_mask[:40, 70:] > 0)
    assert np.array_equal(low[:40, 70:], high[:40, 70:])
    assert np.any(low[55:65, 30:50] != high[55:65, 30:50])
    assert np.abs(high.astype(float) - scene).sum() > np.abs(low.astype(float) - scene).sum()


@pytest.mark.parametrize("rain_strength", [0, .2])
def test_splash_strength_does_not_add_global_headlight_fog(rain_strength):
    scene = np.full((96, 192, 3), 90, dtype=np.uint8)
    aug.cv2.putText(scene, "AB 123", (12, 65), aug.cv2.FONT_HERSHEY_SIMPLEX, 1.1, (15, 15, 15), 3)
    profile = splash_profile(task_target="char", rain_strength=rain_strength,
                             traffic_headlight_strength=.9, tyndall_strength=1)
    low = aug._apply_traffic_headlight_effect(scene, profile, random.Random(123))
    high = aug._apply_traffic_headlight_effect(scene, replace(profile, rain_edge_mist_strength=1), random.Random(123))
    assert np.any(low != scene)
    assert np.array_equal(high, low)
    if rain_strength:
        no_tyndall = aug._apply_traffic_headlight_effect(scene, replace(profile, tyndall_strength=0), random.Random(123))
        assert np.any(low != no_tyndall)
