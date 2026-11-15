from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import z3_review_runtime as review


def _row(number="", source=""):
    data = {
        "status": "needs_fix",
        "characters": [{"character": "A", "bbox": [1.0, 2.0, 11.0, 22.0], "method": "manual"}],
        "review_state": {"schema": review.REVIEW_SCHEMA, "status": review.REVIEW_IN_PROGRESS, "source": "raw_detection", "approved_at": None},
        "gold_state": {"approved": False, "candidate": False, "excluded": False},
    }
    if number:
        data["ground_truth_text"] = number
    if source:
        data["ground_truth_source"] = source
    return data


def _host(data):
    return SimpleNamespace(
        _preview_active_pid="p1",
        preview_metadata={"p1": data},
        _characters_to_text=lambda chars, data=None: "".join(str(rec.get("character") or "") for rec in chars),
        _derive_preview_status_from_data=Mock(return_value="perfect"),
        _ensure_plate_source_metadata=Mock(),
    )


def test_without_saved_number_complete_review_is_ready_for_o():
    data = _row()
    host = _host(data)
    assert review.get_review_quality_status(host, data) == "perfect"
    assert "ground_truth_text" not in data


def test_first_o_saves_current_review_as_number_then_approves():
    data = _row()
    host = _host(data)
    def save_number(_host, target, text, *, prepare=True):
        assert prepare is False
        target["ground_truth_text"] = text
        target["ground_truth_source"] = "manual_z3"
        return {"revision_id": "REV-1"}
    with patch("auto_annotation_tool.gui.z3_plate_gt_runtime.save_plate_ground_truth", side_effect=save_number):
        result = review.confirm_review_gold(host, quiet=True, persist=False, refresh=False)
    assert result["ok"] is True
    assert data["ground_truth_text"] == "A"
    assert data["status"] == "perfect"


def test_matching_number_from_z2_is_not_rewritten():
    data = _row("A", "manual_z2")
    host = _host(data)
    with patch("auto_annotation_tool.gui.z3_plate_gt_runtime.save_plate_ground_truth") as save:
        result = review.confirm_review_gold(host, quiet=True, persist=False, refresh=False)
    assert result["ok"] is True
    assert data["ground_truth_source"] == "manual_z2"
    save.assert_not_called()


def test_mismatch_with_number_from_z2_blocks_silent_overwrite():
    data = _row("B", "manual_z2")
    host = _host(data)
    with patch("auto_annotation_tool.gui.z3_plate_gt_runtime.save_plate_ground_truth") as save:
        result = review.confirm_review_gold(host, quiet=True, persist=False, refresh=False)
    assert result["ok"] is False
    assert result["reason"] == "number_mismatch"
    assert result["inherited_from_z2"] is True
    assert data["ground_truth_text"] == "B"
    assert data["ground_truth_source"] == "manual_z2"
    save.assert_not_called()
