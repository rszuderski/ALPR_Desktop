from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import z2_miniflow_runtime as flow
from auto_annotation_tool.gui import z2_run_history_preview as preview


def test_history_label_contains_total_at_count():
    owner = SimpleNamespace(
        _resolve_safe_annotation_run_dir=lambda path, **kwargs: Path(path),
        _get_run_plate_annotation_counts=Mock(return_value=(7, 12)),
    )
    label = flow._format_manual_review_history_label(owner, {
        "run_dir": "run_001_20260925_120000",
        "created_at": "2026-09-25T12:00:00",
    })
    assert label == "run_001_20260925_120000 | 2026-09-25 12:00 | AT: 12"


def test_evenly_spaced_sample_covers_entire_run():
    assert preview.select_evenly_spaced(list(range(101)), 5) == [0, 25, 50, 75, 100]


def test_compute_crop_box_adds_margin_and_clamps():
    assert preview.compute_crop_box(
        (100, 80), (20, 20, 60, 40), 0.15
    ) == (12, 12, 68, 48)
    assert preview.compute_crop_box(
        (100, 80), (1, 2, 20, 15), 0.15
    ) == (0, 0, 28, 23)


def test_collect_preview_samples_individual_at(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "annotations.xml").write_text(
        "<annotations/>", encoding="utf-8"
    )
    images = run / "images"
    images.mkdir()

    p1 = SimpleNamespace(bbox=(1, 2, 10, 12), polygon=None)
    p2 = SimpleNamespace(bbox=(20, 22, 40, 35), polygon=None)
    p3 = SimpleNamespace(bbox=(5, 6, 15, 16), polygon=None)
    a = SimpleNamespace(filename="a.jpg")
    b = SimpleNamespace(filename="b.jpg")

    owner = SimpleNamespace(
        _resolve_safe_annotation_run_dir=lambda path, **kwargs: Path(path),
        _parse_cvat_preview_annotations=Mock(return_value=[a, b]),
        _get_plate_detections=lambda ann: [p1, p2] if ann is a else [p3],
        _load_annotation_run_manifest=Mock(return_value={}),
        _resolve_run_image_dir_for_annotations=Mock(return_value=images),
        _get_run_plate_annotation_counts=Mock(return_value=(2, 3)),
    )

    result = preview.collect_run_history_preview_items(
        owner, run, limit=8
    )
    assert result["ok"]
    assert result["total_plates"] == 3
    assert len(result["items"]) == 3
    assert [item["local_index"] for item in result["items"]] == [1, 2, 1]


def test_selection_change_enables_preview_and_open_buttons():
    class Button:
        def __init__(self):
            self.state = None

        def configure(self, **kwargs):
            self.state = kwargs.get("state")

    owner = SimpleNamespace(
        manual_history_open_btn=Button(),
        manual_history_preview_btn=Button(),
        _get_selected_manual_review_history_run_dir=lambda: Path("run"),
        _refresh_step2_action_states=Mock(),
        _refresh_free_mode_workflow_ui=Mock(),
        _queue_free_mode_session_save=Mock(),
    )

    flow._on_manual_history_selection_changed(owner)

    assert owner.manual_history_open_btn.state == "normal"
    assert owner.manual_history_preview_btn.state == "normal"
