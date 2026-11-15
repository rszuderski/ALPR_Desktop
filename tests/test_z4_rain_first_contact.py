from dataclasses import replace
import random

import numpy as np
import pytest

from auto_annotation_tool.training import dataset_augmentation as aug


def barrier_geometry():
    gate = np.zeros((20, 24), dtype=np.float32)
    gate[8] = 1
    gate[12] = 1
    nx = np.zeros_like(gate)
    ny = np.ones_like(gate)
    return gate, nx, ny


def test_cpu_stops_at_first_of_two_surfaces_and_normal_faces_incoming_drop():
    geometry = barrier_geometry()
    hit = aug._first_rain_contact(geometry, 10, 2, 0, 1, 16)
    assert hit[:3] == (10, 8, 6)
    assert hit[3:5] == (0, -1)
    reverse = aug._first_rain_contact(geometry, 10, 18, 0, -1, 16)
    assert reverse[:3] == (10, 12, 6)
    assert reverse[3:5] == (0, 1)
    assert aug._first_rain_contact(geometry, 0, 2, 1, 0, 16) is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_first_contact_matches_cpu_and_removes_all_later_samples(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    geometry = barrier_geometry()
    tensors = tuple(torch.from_numpy(value).to(device) for value in geometry)
    anchors = [(10, 2, 0., 1.), (0, 2, 2**-.5, 2**-.5), (0, 2, 1., 0.), (-3, -2, 0., 1.)]
    steps = torch.arange(17, device=device)[None, :]
    ax, ay, dx, dy = [torch.tensor([item[i] for item in anchors], device=device) for i in range(4)]
    px = torch.round(ax[:, None]+dx[:, None]*steps).long()
    py = torch.round(ay[:, None]+dy[:, None]*steps).long()
    valid = (px >= 0) & (px < 24) & (py >= 0) & (py < 20)
    clipped, hit, hx, hy, nx, ny, incidence = aug._clip_rain_paths_at_first_contact(torch, px, py, valid, dx, dy, tensors)
    for i, (x, y, vx, vy) in enumerate(anchors):
        expected = aug._first_rain_contact(geometry, x, y, vx, vy, 16)
        assert bool(hit[i]) == (expected is not None)
        if expected:
            assert (int(hx[i]), int(hy[i])) == expected[:2]
            assert not bool(clipped[i, int(expected[2])+1:].any())
        else:
            assert torch.equal(clipped[i], valid[i])
    assert not bool(clipped[0, py[0] > 8].any())


def test_incidence_uses_actual_local_direction_not_global_rain_direction():
    image = np.full((32, 32, 3), 40, dtype=np.uint8)
    image[:, 16:] = 200
    geometry = aug._rain_collision_geometry(image, .55)
    assert aug._first_rain_contact(geometry, 8, 8, 0, 1, 20) is None
    hit = aug._first_rain_contact(geometry, 8, 8, 1, 0, 20)
    assert hit is not None and hit[0] == 15 and hit[3] < 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grazing_drop_still_stops_at_the_first_contour(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    geometry = barrier_geometry()
    dx, dy = 1 / np.sqrt(1.01), .1 / np.sqrt(1.01)
    contact = aug._first_rain_contact(geometry, 2, 7, dx, dy, 17)
    assert contact is not None and contact[1] == 8
    assert 0 < contact[-1] < .1
    steps = torch.arange(18, device=device)[None, :]
    px = torch.round(2 + dx * steps).long()
    py = torch.round(7 + dy * steps).long()
    valid = torch.ones_like(px, dtype=torch.bool)
    clipped, hit, hx, hy, *_ = aug._clip_rain_paths_at_first_contact(
        torch, px, py, valid, torch.tensor([dx], device=device), torch.tensor([dy], device=device),
        tuple(torch.from_numpy(value).to(device) for value in geometry),
    )
    assert bool(hit[0]) and (int(hx[0]), int(hy[0])) == contact[:2]
    assert not bool(clipped[0, int(contact[2]) + 1:].any())


def test_debug_contours_include_edges_that_local_wind_can_reach():
    image = np.full((32, 32, 3), 40, dtype=np.uint8)
    image[:, 16:] = 200
    profile = aug.AugmentationProfile(task_target="plate", dirt_flow_wind_strength=0).normalized()
    debug = aug._build_scene_rain_edge_gate(image, profile)
    gate, _, _ = aug._rain_collision_geometry(image, aug._contour_detection_sensitivity(profile))
    assert np.any(debug[:, 15:17] > 0)
    assert np.array_equal(debug > 0, gate > .05)


@pytest.mark.parametrize("target", ["plate", "char"])
def test_cpu_render_has_no_primary_streak_after_collision(monkeypatch, target):
    monkeypatch.setattr(aug, "_iter_stream_rain_anchors", lambda *args, **kwargs: [(40, 60)])
    image = np.full((96, 96, 3), 40, dtype=np.uint8)
    image[64:] = 200
    profile = aug.AugmentationProfile(task_target=target, rain_strength=.65, rain_edge_mist_strength=.8,
                                      rain_drop_size_min=.9, rain_drop_size_max=.9, rain_alpha=1,
                                      night_light_strength=0, dark_relief_strength=0, dirt_flow_wind_strength=0).normalized()
    _, free_mask = aug._apply_rain_effect(image, replace(profile, rain_edge_mist_strength=0), random.Random(123), return_mask=True)
    collided, stopped_mask = aug._apply_rain_effect(image, profile, random.Random(123), return_mask=True)
    assert np.any(free_mask[68:] > 0)
    assert not np.any(stopped_mask[68:] > 0)
    assert np.any(stopped_mask[:64] > 0)
    assert np.any(collided[:64] != image[:64])


def test_cuda_render_stops_the_streak_instead_of_only_fading_it(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    def fixed_anchor(low, high, size, **kwargs):
        value = 40 if high == 96 else 60
        return torch.full(size, value, dtype=torch.int64, device=kwargs["device"])
    monkeypatch.setattr(torch, "randint", fixed_anchor)
    image = np.full((128, 96, 3), 40, dtype=np.uint8)
    image[64:] = 200
    profile = aug.AugmentationProfile(task_target="plate", rain_strength=.65, rain_edge_mist_strength=.8,
                                      rain_drop_size_min=.9, rain_drop_size_max=.9, rain_alpha=1,
                                      dirt_flow_wind_strength=0).normalized()
    free = aug._apply_mt_scene_rain_cuda(image, replace(profile, rain_edge_mist_strength=0), torch, seed=123)
    stopped = aug._apply_mt_scene_rain_cuda(image, profile, torch, seed=123)
    assert np.any(free[72:] != image[72:])
    assert np.array_equal(stopped[72:], image[72:])
    assert np.any(stopped[:64] != image[:64])


def test_splash_is_on_the_incoming_side_not_behind_the_surface():
    mask = aug._rain_contact_splash((32, 32), [(16, 16, 0, -1, 1, .9, .4)], .8)
    assert np.any(mask[:16] > 0)
    assert not np.any(mask[18:] > 0)
    assert not np.any(aug._rain_contact_splash((32, 32), [], .8))
