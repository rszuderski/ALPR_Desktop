from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock
import tkinter as tk
from tkinter import ttk

import numpy as np
import pytest

from auto_annotation_tool.gui.z4_augmentation_modal import Step4AugmentationModal
from auto_annotation_tool.training import dataset_augmentation as aug


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


@pytest.mark.parametrize("target", ["plate", "char"])
def test_effects_cta_unicode_and_rain_contour_toggle_on_real_tk(target):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    modal = None
    try:
        modal = Step4AugmentationModal(root, target=target, profile=aug.AugmentationProfile(task_target=target))
        root.update_idletasks()
        labels = [str(widget.cget("text")) for widget in descendants(modal.window) if isinstance(widget, ttk.Button)]
        assert "Odchyłki serii" in labels
        assert "Odchy?ki serii" not in labels
        assert modal._randomness_mode_to_label("manual") == "Ręczny"
        controls = modal._toolbox_fields("rain")
        toggle = next(field for field in controls if field.get("type") == "check" and field.get("var") is modal.rain_edge_debug_var)
        assert "kontury" in toggle["label"]
        if target == "char":
            assert any(field.get("var") is modal.rain_edge_debug_var for field in modal._toolbox_fields("relief"))
        before = asdict(modal._profile_from_vars())
        modal._refresh_preview = Mock()
        button = ttk.Checkbutton(modal.window, variable=toggle["var"])
        button.invoke()
        assert modal.rain_edge_debug_var.get()
        modal._refresh_preview.assert_called()
        assert asdict(modal._profile_from_vars()) == before
        button.invoke()
        assert not modal.rain_edge_debug_var.get()
    finally:
        if modal is not None:
            modal._cancel()
        root.destroy()


def test_scene_overlay_uses_only_cached_scene_mask(monkeypatch):
    mesh = Mock(side_effect=AssertionError("plate-symbol mesh must not be used for MT"))
    monkeypatch.setattr(aug, "build_contour_profile_debug_overlay", mesh)
    scene = np.full((64, 96, 3), 80, dtype=np.uint8)
    mask = np.zeros(scene.shape[:2], dtype=np.float32)
    mask[31:33] = 1
    host = SimpleNamespace(target="plate")
    overlay = Step4AugmentationModal._compose_rain_edge_debug_preview(host, scene, mask)
    assert np.any(overlay != scene)
    assert np.array_equal(overlay[:20], scene[:20])
    assert np.array_equal(Step4AugmentationModal._compose_rain_edge_debug_preview(host, scene, np.zeros_like(mask)), scene)
    assert np.array_equal(Step4AugmentationModal._compose_rain_edge_debug_preview(host, scene, None), scene)
    mesh.assert_not_called()


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_preview_contours_are_pre_rain_and_do_not_change_output(tmp_path, monkeypatch, backend):
    if not aug.is_albumentations_available():
        pytest.skip("Albumentations unavailable")
    if backend == "cpu":
        monkeypatch.setattr(aug, "is_cuda_available", lambda: False)
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    image = np.full((64, 96, 3), 50, dtype=np.uint8)
    image[32:] = 200
    path = tmp_path / "scene.png"
    aug.cv2.imwrite(str(path), image)
    profile = aug.AugmentationProfile(task_target="plate", rain_strength=.65, rain_edge_mist_strength=.5,
                                      tyndall_strength=0, dark_relief_strength=0, night_light_strength=0,
                                      dirt_flow_wind_strength=0).normalized()
    ok, message, plain = aug.preview_augmentation_image(path, profile)
    assert ok, message
    ok, message, debug = aug.preview_augmentation_image(path, profile, include_contours=True)
    assert ok, message
    if backend == "cuda":
        assert "CUDA/Torch scene fastpath" in message
    assert np.array_equal(plain["augmented_rgb"], debug["augmented_rgb"])
    assert np.any(debug["augmented_rgb"] != debug["original_rgb"])
    assert np.array_equal(debug["rain_edge_debug_mask"], aug.build_rain_edge_debug_mask(image, profile))
