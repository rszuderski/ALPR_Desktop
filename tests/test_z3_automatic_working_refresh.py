from copy import deepcopy

import pytest

from auto_annotation_tool.gui import z3_review_runtime as review
from auto_annotation_tool.gui.z3_detection_controls_ui import format_plate_last_detection_line
from test_z3_working_annotation_contract import make_host
from test_z3_edit_gt_runtime import records


def prepared_ocr(tmp_path, legacy=False):
    host, data = make_host(tmp_path)
    data["raw_detection"]["result_hash"] = "ocr-result"
    for rec in data["raw_detection"]["characters"]:
        rec.update(method="ocr", box_source="generated_box", sign_source="ocr_symbol")
    assert review.prepare_working_annotation_from_raw(host, data)
    if legacy:
        data["working_annotation"].pop("automatic_content_hash")
        data["source_info"] = {
            "bucket": "auto_preview", "origin": "pz2_detect", "last_modified_by": "system",
            "last_modified_at": data["working_annotation"]["prepared_at"],
        }
    return host, data


def new_yolo(data, symbols=True):
    chars = records("AT37")
    if not symbols:
        for rec in chars:
            rec.update(character="", sign_source="", source_tag="yolo_box")
    data["raw_detection"] = {"contract": "gt_blind.v1", "result_hash": "yolo-result", "characters": chars}
    data["last_detection"] = {"pipeline": "YB -> YS" if symbols else "YB", "execution_mode": "annotation"}


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("symbols", [False, True])
def test_new_yolo_refreshes_untouched_ocr_and_preserves_raw(tmp_path, legacy, symbols):
    host, data = prepared_ocr(tmp_path, legacy)
    new_yolo(data, symbols)
    raw = deepcopy(data["raw_detection"])
    assert review.prepare_working_annotation_from_raw(host, data)
    assert data["raw_detection"] == raw
    assert all(rec["box_source"] == "yolo_box" for rec in data["characters"])
    assert "".join(rec["character"] for rec in data["characters"]) == "AT37"
    if not symbols:
        assert all(rec["sign_source"] == "gt_assisted" for rec in data["characters"])
    assert data["working_annotation"]["source_raw_result_hash"] == "yolo-result"
    before = deepcopy(data)
    assert not review.prepare_working_annotation_from_raw(host, data)
    assert before == data


@pytest.mark.parametrize("edit", ["geometry", "symbol", "delete_all", "layout", "gt", "approve", "explicit_edit"])
def test_new_raw_never_replaces_human_content(tmp_path, edit):
    host, data = prepared_ocr(tmp_path)
    if edit == "geometry":
        data["characters"][0]["bbox"][0] += 1
    elif edit == "symbol":
        data["characters"][0]["character"] = "Z"
    elif edit == "delete_all":
        data["characters"] = []
    elif edit == "layout":
        data["plate_layout_override"] = "two_row"
    elif edit == "gt":
        data["ground_truth_text"] = "AT38"
    elif edit == "approve":
        assert review.confirm_review_gold(host, persist=False, quiet=True)["ok"]
    else:
        # Even an edit undone within the same timestamp second is human work.
        review.mark_review_edit_started(host, data)
    new_yolo(data)
    before = deepcopy(data)
    assert not review.prepare_working_annotation_from_raw(host, data)
    assert not review.refresh_stale_automatic_working_annotations(host, {"plate": data})
    assert data == before
    assert "zachowana wcześniejsza korekta" in format_plate_last_detection_line(host, data)


@pytest.mark.parametrize("signal", ["source_info", "timestamp", "manual_box", "layout", "empty", "unknown"])
def test_legacy_review_evidence_blocks_automatic_refresh(tmp_path, signal):
    host, data = prepared_ocr(tmp_path, legacy=True)
    if signal == "source_info":
        data["source_info"]["last_modified_by"] = "human"
    elif signal == "timestamp":
        data["review_state"]["modified_at"] = "later"
    elif signal == "manual_box":
        data["characters"][0]["box_source"] = "manual_box"
    elif signal == "layout":
        data["layout_override_source"] = "frame_toggle"
    elif signal == "empty":
        data["characters"] = []
    else:
        data.pop("source_info")
    new_yolo(data)
    before = deepcopy(data)
    assert not review.refresh_stale_automatic_working_annotations(host, {"plate": data})
    assert data == before


def test_reload_repairs_stale_automatic_work_without_inference(tmp_path):
    host, data = prepared_ocr(tmp_path, legacy=True)
    new_yolo(data, symbols=False)
    assert review.refresh_stale_automatic_working_annotations(host, {"plate": data})
    assert all(rec["box_source"] == "yolo_box" for rec in data["characters"])
    assert "zachowana" not in format_plate_last_detection_line(host, data)


def test_raw_evidence_run_does_not_replace_working_annotation_on_load(tmp_path):
    host, data = prepared_ocr(tmp_path)
    new_yolo(data)
    data["last_detection"]["execution_mode"] = "raw_evidence"
    before = deepcopy(data)
    assert not review.refresh_stale_automatic_working_annotations(host, {"plate": data})
    assert data == before
