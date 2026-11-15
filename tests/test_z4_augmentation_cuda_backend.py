from pathlib import Path

from auto_annotation_tool.training.dataset_augmentation import (
    AugmentationProfile,
    _profile_has_geometry_effect,
    default_manual_randomness_config,
)

ROOT = Path(__file__).resolve().parents[1]

def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")

def test_manual_randomness_defaults_off():
    for target in ("plate", "char"):
        cfg = default_manual_randomness_config(target)
        for group in cfg["groups"].values():
            assert group["enabled"] is False
            assert float(group["amount"]) == 0.0
            for field in group["fields"].values():
                assert field["enabled"] is False
                assert float(field["amount"]) == 0.0

def test_geometry_fastpath_contract():
    assert _profile_has_geometry_effect(
        AugmentationProfile(task_target="plate", rotation_limit=0.0, translate_limit=0.0, scale_limit=0.0).normalized()
    ) is False
    assert _profile_has_geometry_effect(
        AugmentationProfile(task_target="plate", rotation_limit=1.0).normalized()
    ) is True

def test_cuda_scene_backend_is_present():
    source = read("auto_annotation_tool/training/dataset_augmentation.py")
    assert "def _apply_mt_scene_postprocess_cuda" in source
    assert "CUDA/Torch scene fastpath" in source
    assert "get_torch_module" in source

def test_preview_is_background_and_coalesced():
    source = read("auto_annotation_tool/gui/z4_augmentation_modal.py")
    assert "def _request_effect_preview" in source
    assert "def _finish_effect_preview_worker" in source
    assert 'name="z4-augmentation-preview"' in source
    assert "self._draw_preview_images(payload)" in source

def test_manifest_records_backend():
    source = read("auto_annotation_tool/training/dataset_augmentation.py")
    assert '"execution_backend": "CPU/OpenCV/NumPy"' in source
    assert '"gpu_acceleration": False' in source
    assert "cuda_scene_fastpath_generated" in source
