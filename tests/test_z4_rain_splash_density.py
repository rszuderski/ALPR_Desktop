from dataclasses import replace
import random

import numpy as np
import pytest

from auto_annotation_tool.training import dataset_augmentation as aug


def test_independent_impacts_accumulate_opacity_at_the_contact():
    contact = (32, 40, 0, -1, 1, .9, .2)
    single = aug._rain_contact_splash((64, 80), [contact], .66)
    double = aug._rain_contact_splash((64, 80), [contact, contact], .66)
    assert double[40, 32] > single[40, 32]
    assert np.allclose(double, 1 - (1 - single) ** 2, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_independent_impacts_accumulate_opacity(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    def render(count):
        data = [torch.full((count,), value, device=device) for value in (32., 40., 0., -1., 1., .9, .2)]
        return aug._rain_contact_splash_cuda(torch, (64, 80), torch.ones(count, device=device, dtype=torch.bool), *data, .66).cpu().numpy()
    single, double = render(1), render(2)
    assert double[40, 32] > single[40, 32]
    assert np.allclose(double, 1 - (1 - single) ** 2, atol=1e-6)


def test_splash_strength_adds_fragments_without_fading_the_existing_trajectory():
    contacts = [(32, 40, 0, -1, 1, .9, .2)]
    low = aug._rain_contact_splash((64, 80), contacts, .9, strength=.1)
    high = aug._rain_contact_splash((64, 80), contacts, .9, strength=1)
    assert np.any(low[:, :20] > 0)
    assert np.array_equal(low[:, :20], high[:, :20])
    assert np.all(high >= low - 1e-6)
    assert np.count_nonzero(high > .01) > np.count_nonzero(low > .01)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_reflected_droplets_inherit_the_rain_alpha_setting(monkeypatch, backend):
    scene = np.full((128, 96, 3), 40, dtype=np.uint8)
    scene[64:] = 150
    profile = aug.AugmentationProfile(
        task_target="plate", rain_strength=.65, rain_edge_mist_strength=1,
        rain_edge_mist_radius=.66, rain_drop_size_min=.9, rain_drop_size_max=.9,
        dirt_flow_wind_strength=0, night_light_strength=0, dark_relief_strength=0,
    ).normalized()
    recorded = []
    if backend == "cpu":
        monkeypatch.setattr(aug, "_iter_stream_rain_anchors", lambda *args, **kwargs: [(40, 60)])
        splash = aug._rain_contact_splash
        def capture(shape, contacts, *args, **kwargs):
            recorded.append(np.array([c[-1] for c in contacts]))
            return splash(shape, contacts, *args, **kwargs)
        monkeypatch.setattr(aug, "_rain_contact_splash", capture)
        def render(alpha):
            return aug._apply_rain_effect(scene, replace(profile, rain_alpha=alpha), random.Random(123))
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        def fixed_anchor(low, high, size, **kwargs):
            return torch.full(size, 40 if high == 96 else 60, device=kwargs["device"], dtype=torch.int64)
        monkeypatch.setattr(torch, "randint", fixed_anchor)
        splash = aug._rain_contact_splash_cuda
        def capture(torch, shape, hit, hx, hy, nx, ny, incidence, sizes, opacity, *args, **kwargs):
            recorded.append(opacity[hit].cpu().numpy())
            return splash(torch, shape, hit, hx, hy, nx, ny, incidence, sizes, opacity, *args, **kwargs)
        monkeypatch.setattr(aug, "_rain_contact_splash_cuda", capture)
        def render(alpha):
            return aug._apply_mt_scene_rain_cuda(scene, replace(profile, rain_alpha=alpha), torch, seed=123)
    for alpha in (0, .2, .8):
        output = render(alpha)
        if alpha == 0:
            assert np.array_equal(output, scene)
    zero, low, high = recorded
    assert len(low) and np.all(zero == 0)
    assert np.all(high > low)
    assert np.all((high > 0) & (high <= 1))


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_higher_sensitivity_adds_local_splashes_without_lowering_droplet_alpha(monkeypatch, backend):
    scene = np.full((128, 96, 3), 40, dtype=np.uint8)
    scene[48:] = 65
    scene[80:] = 200
    profile = aug.AugmentationProfile(
        task_target="plate", rain_strength=.65, rain_edge_mist_strength=1,
        rain_edge_mist_radius=.66, rain_drop_size_min=.9, rain_drop_size_max=.9,
        rain_alpha=.7, dirt_flow_wind_strength=0, night_light_strength=0, dark_relief_strength=0,
    ).normalized()
    recorded = []
    if backend == "cpu":
        monkeypatch.setattr(aug, "_iter_stream_rain_anchors", lambda *args, **kwargs: [(32, 44), (64, 76)])
        splash = aug._rain_contact_splash
        def capture(shape, contacts, *args, **kwargs):
            layer = splash(shape, contacts, *args, **kwargs)
            recorded.append((np.array([c[-1] for c in contacts]), layer))
            return layer
        monkeypatch.setattr(aug, "_rain_contact_splash", capture)
        def render(sensitivity):
            return aug._apply_rain_effect(scene, replace(profile, contour_detection_sensitivity=sensitivity), random.Random(123))
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        def anchors(low, high, size, **kwargs):
            weak = torch.arange(size[0], device=kwargs["device"]) % 2 == 0
            return torch.where(weak, 32 if high == 96 else 44, 64 if high == 96 else 76)
        monkeypatch.setattr(torch, "randint", anchors)
        splash = aug._rain_contact_splash_cuda
        def capture(torch, shape, hit, hx, hy, nx, ny, incidence, sizes, opacity, *args, **kwargs):
            layer = splash(torch, shape, hit, hx, hy, nx, ny, incidence, sizes, opacity, *args, **kwargs)
            recorded.append((opacity[hit].cpu().numpy(), layer.cpu().numpy()))
            return layer
        monkeypatch.setattr(aug, "_rain_contact_splash_cuda", capture)
        def render(sensitivity):
            return aug._apply_mt_scene_rain_cuda(scene, replace(profile, contour_detection_sensitivity=sensitivity), torch, seed=123)
    render(0)
    render(1)
    (low_alpha, low_layer), (high_alpha, high_layer) = recorded
    assert len(high_alpha) > len(low_alpha) > 0
    assert np.allclose(high_alpha, low_alpha[0], atol=1e-6)
    assert not np.any(low_layer[:50, :50])
    assert np.count_nonzero(high_layer[:50, :50] > .01) > 10
