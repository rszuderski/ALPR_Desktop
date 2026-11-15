from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import z3_gt_assist_runtime as runtime


class Widget:
    def __init__(self):
        self.values = {}

    def configure(self, **kwargs):
        self.values.update(kwargs)


def active_data(status="suggested"):
    return {
        "ground_truth_text": "ABC123",
        "source_gt_hash": "hash-one",
        "characters": [
            {
                "character": char,
                "bbox": [idx * 10, 0, idx * 10 + 8, 20],
            }
            for idx, char in enumerate("ABC12X")
        ],
        "raw_detection": {
            "contract": "gt_blind.v1",
            "prediction_text": "ABC12X",
            "characters": [
                {
                    "character": char,
                    "bbox": [idx * 10, 0, idx * 10 + 8, 20],
                    "confidence": 0.9,
                    "method": "yolo",
                }
                for idx, char in enumerate("ABC12X")
            ],
        },
        "gt_assist": {
            "schema": runtime.GT_ASSIST_SCHEMA,
            "source": runtime.GT_ASSIST_SOURCE,
            "source_raw_contract": "gt_blind.v1",
            "source_raw_fingerprint": "",
            "source_gt_hash": "hash-one",
            "source_raw_prediction_text": "ABC12X",
            "ground_truth_text": "ABC123",
            "status": status,
            "reason": "exact_gt_assisted",
            "suggestion_text": "ABC123",
            "characters": [
                {
                    "character": char,
                    "bbox": [idx * 10, 0, idx * 10 + 8, 20],
                }
                for idx, char in enumerate("ABC123")
            ],
            "operations": [{"type": "repair_symbols_with_yolo"}],
            "accepted": False,
        },
    }


def host_with(data):
    host = SimpleNamespace()
    host._preview_active_pid = "plate_000001"
    host.preview_metadata = {"plate_000001": data}
    host._get_preview_active_data = lambda create=False: data
    host.btn_gt_assist_apply = Widget()
    host.btn_gt_assist_reject = Widget()
    host.gt_assist_status_lbl = Widget()
    host._set_inline_status_label_state = Mock(
        side_effect=lambda widget, **kwargs: widget.configure(**kwargs)
    )
    host._push_preview_history_snapshot = Mock()
    host._persist_preview_metadata = Mock()
    host._refresh_preview_listbox_row = Mock()
    host._on_preview_select = Mock()
    host._update_preview_edit_status = Mock()
    host.fast_test_running = False
    host.is_processing = False
    return host


def make_current(data):
    data["gt_assist"]["source_raw_fingerprint"] = (
        runtime._canonical_raw_fingerprint(
            data["raw_detection"]
        )
    )
    return data


def test_presentation_enables_current_suggestion():
    data = make_current(active_data())
    host = host_with(data)

    result = runtime.get_gt_assist_presentation(
        host,
        data,
    )

    assert result["status"] == "suggested"
    assert result["can_accept"] is True
    assert result["can_reject"] is True
    assert "ABC12X" in result["text"]
    assert "ABC123" in result["text"]


def test_presentation_blocks_stale_suggestion():
    data = make_current(active_data())
    data["raw_detection"]["prediction_text"] = "CHANGED"
    host = host_with(data)

    result = runtime.get_gt_assist_presentation(
        host,
        data,
    )

    assert result["status"] == "stale"
    assert result["can_accept"] is False
    assert result["can_reject"] is True


def test_refresh_controls_respects_busy_state():
    data = make_current(active_data())
    host = host_with(data)
    host.is_processing = True

    runtime.refresh_gt_assist_controls(host)

    assert host.btn_gt_assist_apply.values["state"] == "disabled"
    assert host.btn_gt_assist_reject.values["state"] == "disabled"


def test_apply_active_assist_persists_review_without_raw_mutation():
    data = make_current(active_data())
    raw_before = dict(data["raw_detection"])
    host = host_with(data)

    original_accept = runtime.accept_gt_assist_suggestion
    runtime.accept_gt_assist_suggestion = Mock(
        return_value={
            "ok": True,
            "reason": "accepted",
            "text": "ABC123",
        }
    )
    try:
        result = runtime.apply_active_gt_assist(host)
    finally:
        runtime.accept_gt_assist_suggestion = original_accept

    assert result["ok"] is True
    host._push_preview_history_snapshot.assert_called_once_with(
        "plate_000001"
    )
    host._persist_preview_metadata.assert_called_once()
    host._refresh_preview_listbox_row.assert_called_once_with(
        "plate_000001"
    )
    host._on_preview_select.assert_called_once_with(None)
    assert data["raw_detection"] == raw_before


def test_reject_active_assist_persists_decision():
    data = make_current(active_data())
    host = host_with(data)

    result = runtime.reject_active_gt_assist(host)

    assert result["ok"] is True
    assert data["gt_assist"]["status"] == "rejected"
    host._persist_preview_metadata.assert_called_once()
