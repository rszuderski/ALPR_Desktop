from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def test_mt_weather_ui_exposes_scene_contour_collision_controls():
    source = read("auto_annotation_tool/gui/z4_augmentation_modal.py")
    assert "Kolizje deszczu z konturami sceny" in source
    assert "Czułość konturów sceny" in source
    assert "Rozbryzg na konturach" in source
    assert "Zasięg rozbryzgu" in source


def test_mt_profile_keeps_scene_edge_collision_strength():
    source = read("auto_annotation_tool/gui/z4_augmentation_modal.py")
    assert "rain_edge_mist = float(self.rain_edge_mist_var.get() or 0.0)" in source


def test_cuda_fastpath_accepts_scene_edge_collision():
    source = read("auto_annotation_tool/training/dataset_augmentation.py")
    start = source.index("def _mt_scene_cuda_fastpath_eligible")
    end = source.index("\ndef ", start + 10)
    chunk = source[start:end]
    assert '"rain_edge_mist_strength"' not in chunk
    assert '"tyndall_strength"' in chunk


def test_cuda_scene_rain_computes_edges_and_collision():
    source = read("auto_annotation_tool/training/dataset_augmentation.py")
    start = source.index("def _apply_mt_scene_rain_cuda")
    end = source.index("\ndef ", start + 10)
    chunk = source[start:end]
    assert "contour_detection_sensitivity" in chunk
    assert "_rain_collision_geometry_cuda(" in chunk
    assert "_clip_rain_paths_at_first_contact(" in chunk
    assert "_rain_contact_splash_cuda(" in chunk
    assert "_rain_drop_direction(profile, average_drop_size)" in chunk


def test_cpu_fallback_does_not_zero_scene_edge_collision():
    source = read("auto_annotation_tool/training/dataset_augmentation.py")
    start = source.index("def _apply_image_postprocess")
    end = source.index("\ndef ", start + 10)
    chunk = source[start:end]
    guard = chunk[chunk.index('if _profile_task_target(profile) == "plate":'):chunk.index("rng = rng or random.Random")]
    assert "rain_edge_mist_strength=0.0" not in guard
