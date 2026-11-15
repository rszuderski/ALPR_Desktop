from types import SimpleNamespace
from copy import deepcopy
from unittest.mock import Mock

import pytest

from auto_annotation_tool.gui import z3_preview_list_ui as preview


class Host(SimpleNamespace):
    def _sort_character_records_by_x(self, records, data=None):
        def key(rec):
            if not isinstance(rec, dict):
                return 0
            bbox = rec.get("bbox", [])
            try:
                return float(bbox[0])
            except Exception:
                return 0
        return sorted(list(records or []), key=key)

    def _get_detection_method_key(self):
        return "OCR"

    def _get_preview_box_mode_key(self):
        return self.mode


def test_raw_result_preview_uses_frozen_raw_detection_not_review_characters():
    host = Host(mode="RAW_RESULT")
    data = {
        "characters": [
            {"character": "G", "bbox": [100, 0, 120, 20]},
        ],
        "raw_detection": {
            "contract": "gt_blind.v1",
            "characters": [
                {"character": "B", "bbox": [30, 0, 40, 20]},
                {"character": "A", "bbox": [10, 0, 20, 20]},
            ],
        },
    }

    variants = preview.get_preview_box_variants(host, data)
    records, source = preview.get_preview_box_records(host, data)

    assert [item["character"] for item in variants["RAW_RESULT"]] == ["A", "B"]
    assert [item["character"] for item in records] == ["A", "B"]
    assert source == "RAW_RESULT"

    # REVIEW/GOLD layer remains independent.
    assert [item["character"] for item in variants["FINAL"]] == ["G"]


def test_raw_result_preview_is_empty_when_no_raw_snapshot_exists():
    host = Host(mode="RAW_RESULT")
    data = {
        "characters": [
            {"character": "A", "bbox": [10, 0, 20, 20]},
        ],
    }

    records, source = preview.get_preview_box_records(host, data)

    assert records == []
    assert source == "RAW_RESULT"


def test_preview_mode_definitions_expose_raw_result_in_both_z3_sources():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    init = (
        root / "auto_annotation_tool/gui/z3_init_runtime.py"
    ).read_text(encoding="utf-8-sig")
    tab = (
        root / "auto_annotation_tool/gui/tab_character_annotation.py"
    ).read_text(encoding="utf-8-sig")

    for source in (init, tab):
        assert '("RAW_RESULT", "RAW wynik eksperymentu")' in source
        assert '"Wynik RAW": "RAW_RESULT"' in source


def raw_plate():
    return {
        "characters": [],
        "raw_detection": {
            "contract": "gt_blind.v1",
            "characters": [{"character": "A", "bbox": [10, 0, 20, 20]}],
        },
        "yolo_detections": [{"character": "B", "bbox": [30, 0, 40, 20]}],
        "yolo_raw_detections": [{"character": "C", "bbox": [50, 0, 60, 20]}],
    }


def test_auto_shows_raw_before_preparation_without_changing_metadata():
    data = raw_plate()
    data["ground_truth_text"] = "DIFFERENT_GT"
    before = deepcopy(data)
    records, source = preview.get_preview_box_records(Host(mode="AUTO"), data)
    assert source == "RAW_RESULT"
    assert records == data["raw_detection"]["characters"]
    assert data == before


@pytest.mark.parametrize("characters", [[], [{"character": "R", "bbox": [0, 0, 5, 5]}]])
def test_auto_preserves_review_including_deliberately_deleted_boxes(characters):
    data = raw_plate()
    data.update(characters=characters, review_state={"status": "in_progress"})
    records, source = preview.get_preview_box_records(Host(mode="AUTO"), data)
    assert source == "FINAL"
    assert records == characters


def test_auto_keeps_empty_raw_result_instead_of_showing_rejected_proposals():
    data = raw_plate()
    data["raw_detection"]["characters"] = []
    assert preview.get_preview_box_records(Host(mode="AUTO"), data) == ([], "RAW_RESULT")


def test_auto_supports_legacy_yolo_records_without_raw_snapshot():
    data = raw_plate()
    data.pop("raw_detection")
    assert preview.get_preview_box_records(Host(mode="AUTO"), data) == (
        data["yolo_detections"], "YOLO_FILTERED"
    )


@pytest.mark.parametrize("mode", ["GT_RESULT", "RAW_RESULT", "FINAL", "YOLO_FILTERED", "YOLO_NMS", "YOLO_RAW"])
def test_explicit_mode_still_selects_only_requested_layer(mode):
    host = Host(mode=mode)
    data = raw_plate()
    expected = preview.get_preview_box_variants(host, data)[mode]
    assert preview.get_preview_box_records(host, data) == (expected, mode)


def test_navigation_does_not_sort_unselected_diagnostic_proposals():
    host = Host(mode="AUTO")
    data = raw_plate()
    host._sort_character_records_by_x = Mock(wraps=host._sort_character_records_by_x)
    preview.get_preview_box_records(host, data)
    host._sort_character_records_by_x.assert_called_once_with(data["raw_detection"]["characters"])


def mode_host(mode):
    host = Host(mode=mode)
    host.preview_box_mode_var = SimpleNamespace(set=lambda value: setattr(host, "mode", value))
    host._get_preview_box_mode_label = lambda key: key
    host._save_local_setting = Mock()
    host._get_preview_box_records = lambda data: preview.get_preview_box_records(host, data)
    return host


def test_open_existing_run_recovers_old_forced_final_mode():
    host = mode_host("FINAL")
    data = raw_plate()
    preview.restore_preview_stage_mode(host, {"plate": data})
    assert host.mode == "AUTO"
    assert host._get_preview_box_records(data)[1] == "RAW_RESULT"
    host._save_local_setting.assert_called_once_with("char_preview_box_mode", "AUTO")


@pytest.mark.parametrize("mode", ["AUTO", "RAW_RESULT", "YOLO_RAW"])
def test_open_existing_run_preserves_other_modes(mode):
    host = mode_host(mode)
    preview.restore_preview_stage_mode(host, {"plate": raw_plate()})
    assert host.mode == mode
    host._save_local_setting.assert_not_called()


def test_open_reviewed_run_preserves_final_mode():
    host = mode_host("FINAL")
    data = raw_plate()
    data["review_state"] = {"status": "in_progress"}
    preview.restore_preview_stage_mode(host, {"plate": data})
    assert host.mode == "FINAL"


@pytest.mark.parametrize("raw_only,expected", [(True, "AUTO"), (False, "AUTO")])
def test_detection_start_selects_its_output_layer(tmp_path, raw_only, expected):
    from auto_annotation_tool.gui import z3_detection_runtime as detection
    host = mode_host("FINAL")
    host.preview_metadata = {"plate": raw_plate()}
    host._get_sorted_preview_plate_ids = lambda values: values
    host.fast_test_running = False
    host._force_save_all = Mock()
    host.preview_dir_var = SimpleNamespace(get=lambda: str(tmp_path))
    host._project_reset_token = 1
    host.test_log_text = Mock()
    # Stop before the inference worker; exercise the real start path and its UI mode.
    host._lock_ui_for_testing = Mock(return_value=False)
    detection.run_fast_ocr_test(host, {"raw_only": raw_only})
    assert host.mode == expected
    host._save_local_setting.assert_called_once_with("char_preview_box_mode", expected)


def test_editing_review_in_auto_does_not_hide_next_unreviewed_plate():
    from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab
    from auto_annotation_tool.gui import z3_review_runtime as review
    host = mode_host("RAW_RESULT")
    data = raw_plate()
    data["characters"] = deepcopy(data["raw_detection"]["characters"])
    data["review_state"] = {"status": "in_progress"}
    host._get_preview_active_data = lambda **kwargs: data
    review._refresh_after_change(host, "plate", persist=False)
    assert host.mode == "AUTO"
    CharacterAnnotationTab._ensure_preview_final_box_mode(host, render_preview=False)
    assert host.mode == "AUTO"
    assert host._get_preview_box_records(data)[1] == "FINAL"
    assert host._get_preview_box_records(raw_plate())[1] == "RAW_RESULT"
