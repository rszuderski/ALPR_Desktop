from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def test_mt_preview_is_full_image_not_plate_surface():
    source = read("auto_annotation_tool/gui/z4_augmentation_modal.py")
    assert 'self._scene_viewport_visible: bool = self.target != "plate"' in source
    assert 'desired_scene_visible = self.target != "plate"' in source
    assert 'if scene_visible:' in source
    assert 'self.scene_plate_texture_var.set(self.target != "plate")' in source


def test_mt_ui_does_not_expose_plate_local_lights_or_contour_weather():
    source = read("auto_annotation_tool/gui/z4_augmentation_modal.py")
    assert 'if active_toolbox == "illumination" and self.target != "plate":' in source
    assert "Oświetlenie pełnej sceny" in source
    assert "Pogoda pełnej sceny" in source
    assert 'if self.target != "plate" and (' in source


def test_mt_profile_forces_crop_only_controls_off():
    source = read("auto_annotation_tool/gui/z4_augmentation_modal.py")
    assert "wet_reflection_strength = 0.0" in source
    assert "False if is_plate_dataset else bool(self.scene_plate_texture_var.get())" in source
    assert "0.0\n            if is_plate_dataset\n            else float(self.dark_relief_var.get() or 0.0)" in source


def test_backend_sanitizes_old_presets_for_mt_full_scene():
    source = read("auto_annotation_tool/training/dataset_augmentation.py")
    start = source.index("def _apply_image_postprocess")
    guard = source[start:source.index("rng = rng or random.Random", start)]
    assert 'if _profile_task_target(profile) == "plate":' in guard
    required = (
        "scene_plate_texture_enabled=False",
        "dark_relief_strength=0.0",
        "water_film_strength=0.0",
        "wet_reflection_strength=0.0",
        "plate_reflect_glare_strength=0.0",
        "dirt_flow_points=0",
        "traffic_headlight_strength=0.0",
    )
    for token in required:
        assert token in guard
    assert "rain_edge_mist_strength=0.0" not in guard
