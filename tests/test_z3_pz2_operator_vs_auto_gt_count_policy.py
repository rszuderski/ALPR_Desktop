from types import SimpleNamespace

from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab
from auto_annotation_tool.gui.z3_gt_box_policy import (
    _select_limited_boxes,
    can_add_character_box,
)


def _chars(text):
    return [
        {
            "character": char,
            "bbox": [float(i * 10), 0.0, float(i * 10 + 8), 20.0],
            "confidence": 0.9,
        }
        for i, char in enumerate(text)
    ]


def test_manual_operator_can_add_box_even_when_manual_z2_count_is_reached():
    host = SimpleNamespace(
        _update_preview_edit_status=lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(AssertionError("manual add must not be blocked"))
        )
    )
    data = {
        "ground_truth_text": "ABC123",
        "ground_truth_source": "manual_z2",
        "characters": _chars("ABC123"),
    }

    assert can_add_character_box(host, data) is True


def test_manual_operator_can_add_box_for_manual_z3_or_filename_fallback():
    host = SimpleNamespace(
        _update_preview_edit_status=lambda *_args, **_kwargs: None
    )

    for source in ("manual_z3", "filename_order", ""):
        data = {
            "ground_truth_text": "B955ET",
            "ground_truth_source": source,
            "characters": _chars("B955ET"),
        }
        assert can_add_character_box(host, data) is True


def test_automatic_gt_box_limiter_still_trims_detection_candidates():
    host = SimpleNamespace(
        _sort_character_records_by_x=lambda records, data=None: list(records),
    )
    records = _chars("ABCDEFG")
    data = {
        "ground_truth_text": "ABCDEF",
        "ground_truth_source": "manual_z2",
    }

    selected, details = _select_limited_boxes(host, data, records)

    assert len(selected) == 6
    assert details["limit"] == 6
    assert details["original_count"] == 7
    assert details["removed_count"] == 1


def test_manual_z2_resolves_hard_expected_count():
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    data = {
        "ground_truth_text": "ABC123",
        "ground_truth_source": "manual_z2",
        "characters": _chars("ABC123"),
    }

    result = host._resolve_preview_expected_text_for_crop(
        data,
        data["characters"],
    )

    assert result["expected_source"] == "ground_truth"
    assert result["count_resolved"] is True
    assert result["target_length"] == 6
    assert result["target_lengths"] == [6]
    assert result["text_resolved"] is True


def test_manual_z3_never_resolves_box_count_even_when_saved_text_exists():
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    data = {
        "ground_truth_text": "B955ET",
        "ground_truth_source": "manual_z3",
        "characters": _chars("BI955ET"),
    }

    result = host._resolve_preview_expected_text_for_crop(
        data,
        data["characters"],
    )

    assert result["expected_source"] == "local_ground_truth"
    assert result["count_resolved"] is False
    assert result["target_length"] == 0
    assert result["target_lengths"] == []
    assert result["expected_lengths"] == []
    assert result["text_resolved"] is False


def test_manual_z3_exact_reading_is_text_resolved_but_not_count_resolved():
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    data = {
        "ground_truth_text": "BI955ET",
        "ground_truth_source": "manual_z3",
        "characters": _chars("BI955ET"),
    }

    result = host._resolve_preview_expected_text_for_crop(
        data,
        data["characters"],
    )

    assert result["text_resolved"] is True
    assert result["count_resolved"] is False
    assert result["target_length"] == 0
    assert result["target_lengths"] == []
