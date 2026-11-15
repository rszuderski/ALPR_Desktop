from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def source():
    return (ROOT / "auto_annotation_tool/training/dataset_augmentation.py").read_text(
        encoding="utf-8-sig"
    )

def test_scene_weather_has_cuda_renderer():
    text = source()
    assert "def _apply_mt_scene_rain_cuda" in text
    assert "rain.index_add_" in text
    assert "grid_sample(" in text

def test_rain_no_longer_blocks_cuda_scene_fastpath():
    text = source()
    start = text.index("def _mt_scene_cuda_fastpath_eligible")
    end = text.index("\ndef ", start + 10)
    chunk = text[start:end]
    assert '"rain_strength"' not in chunk
    assert '"rain_lens_strength"' not in chunk

def test_scene_weather_is_called_from_cuda_postprocess():
    text = source()
    start = text.index("def _apply_mt_scene_postprocess_cuda")
    end = text.index("\ndef ", start + 10)
    chunk = text[start:end]
    assert "_apply_mt_scene_rain_cuda(" in chunk
    assert "seed=seed + 17011" in chunk

def test_plate_surface_weather_still_blocks_but_scene_collisions_are_allowed():
    text = source()
    start = text.index("def _mt_scene_cuda_fastpath_eligible")
    end = text.index("\ndef ", start + 10)
    chunk = text[start:end]
    assert '"rain_edge_mist_strength"' not in chunk
    assert '"tyndall_strength"' in chunk
