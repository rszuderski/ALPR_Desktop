"""AKT89/A: same-plate preview, independent denominators and execution evidence."""
import copy
import json
from dataclasses import replace

import pytest

from test_mobile_p2b2_canvas_review import source_session, MANUAL, text_all
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_canvas_presentation import (
    execution_state, selected_assessment, current_summary, build_inventory, summary_caption,
)
from auto_annotation_tool.ranking.mobile_human_review import align_plate_text


def preview(s, oid="GT:a0", **kwargs):
    return selected_assessment(core.scene(s, "mt-one"), oid,
                               execution_state(s, "mt-one", **kwargs))


@pytest.mark.parametrize("prediction,gt", [
    ("SF5R", "SF5R"), ("SFSR", "SF5R"), ("SF55R", "SF5R"),
    ("SFR", "SF5R"), ("ABCDEFG", "A"), ("", "SF5R"),
])
def test_alignment_uses_canonical_distance_without_cer_cap(tmp_path, prediction, gt):
    s = source_session(tmp_path, n=1)
    s.attempts["a0"]["prediction"] = prediction
    for sample in s.samples.values():
        sample["prediction"] = prediction
    core.set_text(s, "mt-one", "GT:a0", gt)
    before = s.review.to_dict()
    m = preview(s)
    a = align_plate_text(gt, prediction)
    assert (m["cer"], m["edit_distance"], m["exact"]) == (a.cer, a.edit_distance, a.exact_match)
    assert m["gt_characters"] == len(a.ground_truth)
    assert m["draft"] and core.summary(s)["evaluated_reads"] == 0
    assert s.review.to_dict() == before


def test_selection_never_borrows_gt_or_prediction(tmp_path):
    s = source_session(tmp_path)
    core.set_text(s, "mt-one", "GT:a0", "FIRST")
    core.set_text(s, "mt-one", "GT:a1", "SECOND")
    first, second = preview(s), preview(s, "GT:a1")
    assert first["gt"] == "GT: FIRST" and first["mz"] == "MZ: WA01234"
    assert second["gt"] == "GT: SECOND" and second["mz"] == "MZ: WA11234"
    assert preview(s, "")["state"] == "NO_SELECTION"
    assert preview(s, "missing")["cer"] is None


def test_non_evaluable_states_and_missing_text_have_no_fake_zero(tmp_path):
    s = source_session(tmp_path, n=1)
    assert preview(s)["state"] == "GT_MISSING"
    core.set_text(s, "mt-one", "GT:a0", "", "nieczytelny")
    assert preview(s)["state"] == "TEXT_EXCLUDED"
    assert preview(s)["cer"] is None
    sc = core.scene(s, "mt-one")
    for changes, state in [({"quad": None}, "INVALID_GEOMETRY"),
                           ({"post_nms_suppressed": True}, "EXCLUDED"),
                           ({"error": True}, "EXCLUDED")]:
        altered = copy.deepcopy(sc)
        altered["objects"][0].update(changes)
        m = selected_assessment(altered, "GT:a0", execution_state(s, "mt-one"))
        assert m["state"] == state and m["cer"] is None and m["polygon_iou_pct"] is None


def test_manual_fn_gated_by_checked_mt_and_never_enters_mz_or_geometry(tmp_path):
    s = source_session(tmp_path, n=0, mz=False)
    oid = core.add_plate(s, "mt-one", MANUAL)
    core.set_text(s, "mt-one", oid, "GXW9")
    m = preview(s, oid)
    assert m["execution"]["state"] == "MT_EXECUTED"
    assert m["execution"]["detections"] == 0
    assert "FN" in m["outcome"] and m["draft"]
    assert m["state"] == "MZ_NOT_RUN" and m["cer"] is None
    assert m["polygon_iou_pct"] is None and m["nmce_pct"] is None
    before = core.summary(s)
    assert before["expected_plates"] == before["missed_plates"] == 0
    core.check_scene(s, "mt-one")
    after = core.summary(s)
    assert after["expected_plates"] == after["missed_plates"] == 1
    assert after["mt_recall_operator"] == 0
    assert after["geometry_n"] == after["evaluated_reads"] == after["gt_characters"] == 0
    assert after["cer"] is None and not preview(s, oid)["draft"]
    paths = core.export(s, tmp_path / "export")
    exported = json.loads(next(p for p in paths if p.suffix == ".json").read_text(encoding="utf-8"))
    row = exported["rows"][0]
    assert row["mz_status"] == "NOT_RUN" and row["outcome"] == "MISSED_MT"
    assert row["cer"] is row["polygon_iou_pct"] is row["nmce_pct"] is None


@pytest.mark.parametrize("status,changes", [
    ("MT_NOT_RUN", dict(executed=False)), ("MT_CANCELLED", dict(cancelled=True)),
    ("MT_EXECUTION_FAILED", dict(execution_failed=True)),
])
def test_incomplete_execution_never_becomes_fn(tmp_path, status, changes):
    s = source_session(tmp_path, n=0, mz=False)
    oid = core.add_plate(s, "mt-one", MANUAL)
    core.set_text(s, "mt-one", oid, "GXW9")
    core.check_scene(s, "mt-one")
    sc = core.scene(s, "mt-one")
    g = s.mt_invocations["mt-one"]
    if status == "MT_NOT_RUN":
        for row in g.records:
            row.update(mt_status="NOT_RUN", mt_executed=False)
    s.mt_invocations["mt-one"] = replace(g, **changes)
    execution = execution_state(s, "mt-one")
    assert execution["state"] == status and execution["detections"] is None
    m = selected_assessment(sc, oid, execution)
    assert m["gt"] == "GT: GXW9" and m["cer"] is m["polygon_iou_pct"] is None
    d = core.summary(s)
    assert all(d[k] == 0 for k in ("expected_plates", "missed_plates", "geometry_n", "evaluated_reads", "gt_characters"))


def test_unknown_or_wrong_frame_does_not_inherit_execution(tmp_path):
    s = source_session(tmp_path, n=0)
    assert execution_state(s, "missing")["state"] == "MT_UNKNOWN"
    assert execution_state(s, "mt-one", image_entry="other.jpg")["state"] == "MT_UNKNOWN"
    assert execution_state(s, "mt-one", image_size=(12, 13))["state"] == "MT_UNKNOWN"
    s.mt_invocations["mt-one"].records[0]["mt_invocation_id"] = "neighbour"
    assert execution_state(s, "mt-one")["state"] == "MT_UNKNOWN"


def test_inconsistent_source_sequence_and_missing_mz_proof(tmp_path):
    s = source_session(tmp_path, n=1, mz=False)
    row = s.mt_invocations["mt-one"].records[0]
    row.update(mz_status="", prediction="")
    assert preview(s)["state"] == "MZ_UNKNOWN"
    assert "nie potwierdzono" in preview(s)["mz"]
    row["source_sequence"] = "other-frame"
    assert execution_state(s, "mt-one")["state"] == "MT_UNKNOWN"


def test_nms_header_uses_final_count_and_raw_is_never_separate_fp(tmp_path, monkeypatch):
    from auto_annotation_tool.ranking import mobile_mt_postdedup
    s = source_session(tmp_path)
    proof = dict(raw=2, after_dedup=1, suppressed=1, primary_id="a0", suppressed_id="a1")
    monkeypatch.setattr(mobile_mt_postdedup, "post_nms_proof", lambda *args: proof)
    core.set_text(s, "mt-one", "GT:a0", "WA01234")
    core.check_scene(s, "mt-one")
    execution = execution_state(s, "mt-one")
    assert execution["detections"] == 1 and "RAW MT: 2" in execution["detail"]
    assert preview(s, "GT:a1")["cer"] is None
    assert preview(s, "GT:a1")["state"] == "EXCLUDED"
    d = core.summary(s)
    assert d["expected_plates"] == d["linked_plates"] == d["evaluated_reads"] == 1
    assert d["false_positive_mt"] == d["missed_plates"] == 0


def test_invalid_model_quad_is_not_not_run(tmp_path):
    from test_z4_akt89lm_resolution import invalid_source
    s = invalid_source(tmp_path, mz=True)
    core.correct_plate(s, "mt-one", "GT:a0", MANUAL)
    core.set_text(s, "mt-one", "GT:a0", "WA01234")
    m = preview(s)
    assert m["execution"]["detections"] == 1 and m["exact"]
    assert m["polygon_iou_pct"] is m['nmce_pct'] is None
    assert m['geometry_status'] == 'NOT_AVAILABLE' and 'źródłowy' in m['reference']


def test_summary_requires_session_source_revision_and_micro_cer(tmp_path):
    s = source_session(tmp_path)
    text_all(s); core.check_scene(s, "mt-one")
    data = core.summary(s)
    assert current_summary(data, s)
    for key, value in [("session_id", "other"), ("source_archive_sha256", "other"), ("review_revision", -1)]:
        assert not current_summary(dict(data, **{key: value}), s)
    data.update(exact_reads=8, evaluated_reads=9, text_errors=1, gt_characters=62,
                geometry_n=9, polygon_iou_mean_pct=92.17, nmce_mean_pct=1.37, cer=.5)
    inv = dict(build_inventory(s), total=19, inspected=9)
    text = summary_caption(data, inv)
    assert "9/19" in text and "8/9" in text and "1,61% (1/62)" in text
    assert "92,17% (n=9)" in text and "1,37% (n=9)" in text
