from types import SimpleNamespace

import auto_annotation_tool.gui.z3_review_runtime as review_runtime
import auto_annotation_tool.gui.z3_plate_gt_runtime as plate_gt_runtime


def _host_for(data):
    host = SimpleNamespace()
    host.preview_metadata = {"plate_1": data}
    host._preview_active_pid = "plate_1"
    host._characters_to_text = lambda chars, data=None: "".join(
        str(item.get("character", "")) for item in chars
    )
    host._derive_preview_status_from_data = lambda probe, records: "perfect"
    host._ensure_plate_source_metadata = lambda *args, **kwargs: None
    return host


def _data(*, gt_text, gt_source, candidate):
    return {
        "ground_truth_text": gt_text,
        "ground_truth_source": gt_source,
        "plate_attributes": {
            "ground_truth_text": gt_text,
            "ground_truth_source": gt_source,
        },
        "characters": [
            {"character": char, "bbox": [idx, 0, idx + 1, 1]}
            for idx, char in enumerate(candidate)
        ],
        "review_state": {"status": review_runtime.REVIEW_IN_PROGRESS},
        "gold_state": {"approved": False, "candidate": False},
        "status": "needs_fix",
    }


def test_local_z3_number_is_synchronized_by_explicit_o(monkeypatch):
    data = _data(gt_text="OLD123", gt_source="manual_z3", candidate="NEW123")
    host = _host_for(data)
    saved = []

    def fake_save(host_arg, data_arg, text, *, prepare=True):
        saved.append((text, prepare))
        data_arg["ground_truth_text"] = text
        data_arg["ground_truth_source"] = "manual_z3"
        data_arg["plate_attributes"]["ground_truth_text"] = text
        data_arg["plate_attributes"]["ground_truth_source"] = "manual_z3"

    monkeypatch.setattr(plate_gt_runtime, "save_plate_ground_truth", fake_save)
    monkeypatch.setattr(
        review_runtime, "get_review_quality_status",
        lambda host_arg, data_arg, chars=None: "perfect"
    )

    result = review_runtime.confirm_review_gold(
        host, persist=False, quiet=True, refresh=False
    )

    assert result["ok"] is True
    assert saved == [("NEW123", False)]
    assert data["ground_truth_text"] == "NEW123"
    assert data["review_state"]["status"] == review_runtime.REVIEW_APPROVED
    assert data["gold_state"]["approved"] is True


def test_inherited_z2_number_stays_protected(monkeypatch):
    data = _data(gt_text="OLD123", gt_source="manual_z2", candidate="NEW123")
    host = _host_for(data)
    saved = []

    monkeypatch.setattr(
        plate_gt_runtime, "save_plate_ground_truth",
        lambda *args, **kwargs: saved.append((args, kwargs))
    )
    monkeypatch.setattr(
        review_runtime, "get_review_quality_status",
        lambda host_arg, data_arg, chars=None: "perfect"
    )

    result = review_runtime.confirm_review_gold(
        host, persist=False, quiet=True, refresh=False
    )

    assert result["ok"] is False
    assert result["reason"] == "number_mismatch"
    assert result["inherited_from_z2"] is True
    assert saved == []
    assert data["ground_truth_text"] == "OLD123"


def test_first_o_without_any_gt_still_creates_manual_z3_number(monkeypatch):
    data = _data(gt_text="", gt_source="", candidate="NEW123")
    host = _host_for(data)
    saved = []

    def fake_save(host_arg, data_arg, text, *, prepare=True):
        saved.append((text, prepare))
        data_arg["ground_truth_text"] = text
        data_arg["ground_truth_source"] = "manual_z3"
        data_arg["plate_attributes"]["ground_truth_text"] = text
        data_arg["plate_attributes"]["ground_truth_source"] = "manual_z3"

    monkeypatch.setattr(plate_gt_runtime, "save_plate_ground_truth", fake_save)
    monkeypatch.setattr(
        review_runtime, "get_review_quality_status",
        lambda host_arg, data_arg, chars=None: "perfect"
    )

    result = review_runtime.confirm_review_gold(
        host, persist=False, quiet=True, refresh=False
    )

    assert result["ok"] is True
    assert saved == [("NEW123", False)]
    assert data["ground_truth_text"] == "NEW123"
