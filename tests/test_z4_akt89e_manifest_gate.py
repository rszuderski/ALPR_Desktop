"""FN without text GT must retain the full export identity/revision gate."""
from copy import deepcopy
import json

import pytest

from auto_annotation_tool.gui.mobile_compare_review_gate import (
    imported_verification, inspect_canvas,
)
from auto_annotation_tool.gui.mobile_comparison_core import load_session_export
from test_z4_akt87_verified_mt_mz import make_export, SID, SHA
from test_z4_akt89e_fn_without_gt import example


def test_fn_without_gt_through_signed_export_and_live_revision(tmp_path):
    results = tmp_path / "results"
    canvas = example()
    directory = make_export(results, canvas=canvas)
    frozen = {p.name: p.read_bytes() for p in directory.iterdir()}
    review_root = tmp_path / "reviews"
    sidecar = review_root / "human_reviews" / (SHA + ".review.json")
    sidecar.parent.mkdir(parents=True)

    def live(revision, status):
        sidecar.write_text(json.dumps(dict(session_id=SID,
            source_archive_sha256=SHA, review_revision=revision,
            review_status=status)), encoding="utf-8")
        return imported_verification(SID, SHA, results, review_root)

    status = live(7, "COMPLETED")
    assert status.ready and status.label == "Zatwierdzona"
    assert load_session_export(directory).canvas_ready
    assert not imported_verification(SID, "c" * 64, results, review_root).ready
    assert live(8, "COMPLETED").state == "STALE"
    assert live(7, "IN_PROGRESS").state == "STALE"
    assert live(7, "COMPLETED").ready
    assert frozen == {p.name: p.read_bytes() for p in directory.iterdir()}
    # A valid old export must not conceal a corrupted newer revision.
    newer = make_export(results, canvas=deepcopy(canvas), rev=8)
    (newer / "canvas_review_summary.json").write_text("{}", encoding="utf-8")
    assert live(7, "COMPLETED").state == "INVALID"


def test_linked_without_gt_still_blocks_a_valid_manifest(tmp_path):
    canvas = example()
    canvas["rows"][0]["gt_text"] = ""
    directory = make_export(tmp_path, canvas=canvas)
    assert not load_session_export(directory).canvas_ready
    status = imported_verification(SID, SHA, tmp_path)
    assert status.state == "INCOMPLETE"
    assert status.reason == "Brak GT znaków lub jawnego wyłączenia odczytu."


@pytest.mark.parametrize("status,ready", [("", True), ("NOT_RUN", True),
    ("not_run", True), ("NO_CHARACTERS", False), ("FAILED", False)])
def test_fn_status_is_not_a_hidden_mz_result(status, ready):
    canvas = example()
    canvas["rows"][1]["mz_status"] = status
    frozen = deepcopy(canvas)
    assert inspect_canvas(canvas, "COMPLETED")[0] is ready
    assert canvas == frozen
