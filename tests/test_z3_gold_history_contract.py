from __future__ import annotations

import copy
from types import SimpleNamespace

from auto_annotation_tool.gui import z3_preview_metadata_runtime as metadata
from auto_annotation_tool.gui import z3_review_runtime as review


def _approved_snapshot():
    data = {
        "status": "perfect",
        "characters": [
            {
                "character": "A",
                "bbox": [1.0, 2.0, 10.0, 20.0],
            }
        ],
        "raw_detection": {
            "result_hash": "raw-1",
            "characters": [
                {
                    "character": "A",
                    "bbox": [1.0, 2.0, 10.0, 20.0],
                }
            ],
        },
        "review_state": {
            "schema": review.REVIEW_SCHEMA,
            "status": review.REVIEW_APPROVED,
            "source": "raw_detection",
            "raw_result_hash": "raw-1",
            "started_at": "2026-09-22T00:00:00+02:00",
            "modified_at": "2026-09-22T00:01:00+02:00",
            "approved_at": "2026-09-22T00:01:00+02:00",
            "approved_by": "human",
            "approved_reference": {
                "gt_hash": "",
                "gt_revision_ids": [],
                "geometry_revision_ids": [],
                "ground_truth_text": "",
                "source_annotation_id": "",
                "plate_annotation_id": "",
                "ground_truth_source": "",
            },
        },
        "gold_state": {
            "candidate": True,
            "approved": True,
        },
    }
    return data


def _host(current):
    host = SimpleNamespace(
        preview_metadata={"p1": current},
        _preview_active_pid="p1",
        _preview_history_replaying=False,
        _preview_char_selected_index=0,
        _preview_char_drag_state={"dirty": True},
        _preview_char_add_state={"dirty": True},
        _preview_char_add_click_armed=True,
        _preview_char_hover_index=0,
        _preview_char_hover_label_index=0,
        _preview_char_label_active_index=0,
    )

    host._mark_review_edit_started = lambda data: review.mark_review_edit_started(host, data)
    host._derive_preview_status_from_data = (
        lambda data, chars: "needs_fix"
        if review.get_review_state_status(data) == review.REVIEW_IN_PROGRESS
        else "perfect"
    )
    host._refresh_preview_live_metadata_ui = lambda **kwargs: None
    host._redraw_preview_character_overlays_light = lambda: True
    host._on_preview_select = lambda event=None: None
    host._persist_preview_metadata = lambda **kwargs: None
    host._schedule_preview_info_refresh = lambda **kwargs: None
    host._refresh_detection_review_controls = lambda: None
    return host


def test_history_restore_restores_content_but_not_gold_approval():
    snapshot = _approved_snapshot()
    current = copy.deepcopy(snapshot)
    current["characters"][0]["character"] = "B"
    current["review_state"]["status"] = review.REVIEW_IN_PROGRESS
    current["review_state"]["approved_at"] = None
    current["gold_state"]["candidate"] = False
    current["gold_state"]["approved"] = False

    host = _host(current)

    assert metadata._restore_preview_plate_history_snapshot(
        host,
        snapshot,
        action_label="undo",
    ) is True

    restored = host.preview_metadata["p1"]
    assert restored["characters"][0]["character"] == "A"
    assert restored["review_state"]["status"] == review.REVIEW_IN_PROGRESS
    assert restored["review_state"]["approved_at"] is None
    assert "approved_reference" not in restored["review_state"]
    assert restored["gold_state"]["candidate"] is False
    assert restored["gold_state"]["approved"] is False
    assert restored["status"] == "needs_fix"


def test_history_restore_does_not_mutate_frozen_snapshot_or_raw():
    snapshot = _approved_snapshot()
    snapshot_before = copy.deepcopy(snapshot)
    host = _host({"status": "needs_fix", "characters": []})

    assert metadata._restore_preview_plate_history_snapshot(
        host,
        snapshot,
        action_label="redo",
    ) is True

    assert snapshot == snapshot_before
    assert host.preview_metadata["p1"]["raw_detection"] == snapshot_before["raw_detection"]


def test_history_restore_of_legacy_perfect_record_opens_review():
    legacy_snapshot = {
        "status": "perfect",
        "characters": [
            {
                "character": "X",
                "bbox": [0.0, 0.0, 8.0, 16.0],
            }
        ],
    }
    host = _host({"status": "needs_fix", "characters": []})

    assert metadata._restore_preview_plate_history_snapshot(
        host,
        legacy_snapshot,
        action_label="undo",
    ) is True

    restored = host.preview_metadata["p1"]
    assert restored["review_state"]["status"] == review.REVIEW_IN_PROGRESS
    assert restored["review_state"]["source"] == "manual_editor"
    assert restored["gold_state"]["candidate"] is False
    assert restored["gold_state"]["approved"] is False
    assert restored["status"] == "needs_fix"
