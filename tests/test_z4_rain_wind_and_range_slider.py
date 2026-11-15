from pathlib import Path

from auto_annotation_tool.gui.z4_augmentation_modal import _choose_range_slider_handle
from auto_annotation_tool.training.dataset_augmentation import (
    AugmentationProfile,
    _rain_drop_direction,
)

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def test_overlapped_range_slider_can_split_both_directions():
    # Samo naciśnięcie na wspólny uchwyt jeszcze nie wybiera strony.
    assert _choose_range_slider_handle(
        20.0, 20.0, 20.0, origin_x=20.0, dragging=False
    ) is None
    # Pierwszy ruch w prawo wybiera max.
    assert _choose_range_slider_handle(
        20.0, 20.0, 30.0, origin_x=20.0, dragging=True
    ) == "max"
    # Pierwszy ruch w lewo wybiera min.
    assert _choose_range_slider_handle(
        20.0, 20.0, 10.0, origin_x=20.0, dragging=True
    ) == "min"


def test_non_overlapped_range_slider_still_selects_nearest_handle():
    assert _choose_range_slider_handle(10.0, 90.0, 16.0) == "min"
    assert _choose_range_slider_handle(10.0, 90.0, 84.0) == "max"


def test_wind_direction_helper_changes_rain_vector():
    calm = AugmentationProfile(
        task_target="plate",
        rain_strength=0.6,
        dirt_flow_wind_strength=0.0,
        dirt_flow_air_angle=0.0,
    ).normalized()
    windy = AugmentationProfile(
        task_target="plate",
        rain_strength=0.6,
        dirt_flow_wind_strength=1.0,
        dirt_flow_air_angle=0.0,
    ).normalized()
    calm_x, calm_y = _rain_drop_direction(calm, 0.2)
    windy_x, windy_y = _rain_drop_direction(windy, 0.2)
    assert abs(windy_x) > abs(calm_x) + 0.2
    assert windy_y < calm_y


def test_cuda_scene_rain_uses_same_wind_contract_as_cpu():
    source = read("auto_annotation_tool/training/dataset_augmentation.py")
    start = source.index("def _apply_mt_scene_rain_cuda")
    end = source.index("\ndef ", start + 10)
    chunk = source[start:end]
    assert "_rain_drop_direction(profile, average_drop_size)" in chunk
    assert "dirt_flow_wind_strength" in chunk
    assert "+ 6.0 * wind_strength" in chunk
