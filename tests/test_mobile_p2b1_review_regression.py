"""P2B1 FIX01: verification gates, legacy GT and persisted multi-plate navigation.

Uses real review sessions and synthetic .alprsession archives. No model runs,
no user Workspace and no relaxation of the visual or per-plate GT contracts.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_mobile_human_review import attempt, load, make_bundle
from test_mobile_mt_blind_review import children
from auto_annotation_tool.ranking.mobile_multi_plate_verification import (
    assign_attempt, inventory, set_count,
)
from auto_annotation_tool.ranking.mobile_verification_policy import (
    observation_verification_cells, verification_issues, visual_decision,
)
from auto_annotation_tool.ranking.mobile_verification_snapshot import (
    has_valid_ready_snapshot,
)


def _blocked_without_write(session, match: str) -> None:
    source = session.path.read_bytes()
    before = copy.deepcopy(session.review.to_dict())
    sidecar = session.sidecar_path.read_bytes() if session.sidecar_path.exists() else None
    ready = session.sidecar_path.parent / "ready_for_evaluation"
    snapshots = sorted(ready.rglob("*")) if ready.exists() else []
    with pytest.raises(ValueError, match=match):
        session.complete()
    assert session.path.read_bytes() == source
    assert session.review.to_dict() == before
    assert (session.sidecar_path.read_bytes() if session.sidecar_path.exists() else None) == sidecar
    assert (sorted(ready.rglob("*")) if ready.exists() else []) == snapshots
    assert not has_valid_ready_snapshot(session)


@pytest.mark.parametrize("evidence", ["empty", "missing_member", "crop_only"])
@pytest.mark.parametrize("count", ["one", "none"])
def test_review_missing_mt_input_cannot_complete(tmp_path, evidence, count):
    row = attempt()
    if evidence == "empty":
        row["evidence_entry"] = ""
    elif evidence == "missing_member":
        row["mt_input_evidence_entry"] = "samples/evidence/not_in_archive.jpg"
    else:
        row["evidence_entry"] = "samples/crops/roi_only.jpg"
    session = load(make_bundle(tmp_path / "input.alprsession", samples=[], attempts=[row]), tmp_path)
    session.annotate_invocation("a1", visible_plate_count=count)
    session.annotate("attempt", "a1", **visual_decision("accurate" if count == "one" else "not_plate"))
    assert not any(entry in session.entry_names for entry in session.mt_invocations["a1"].evidence_entries)
    assert any("Brak dowodu wejścia MT" in issue for issue in verification_issues(session))
    assert session.statistics()["summary"]["evaluable_mt_invocations"] == 0
    _blocked_without_write(session, "Brak dowodu wejścia MT")


@pytest.mark.parametrize("count,evaluable", [("uncertain", True), ("uncertain", False), ("one", False)])
def test_review_missing_input_can_be_explicitly_abstained(tmp_path, count, evaluable):
    row = attempt()
    row["evidence_entry"] = ""
    session = load(make_bundle(tmp_path / "input.alprsession", samples=[], attempts=[row]), tmp_path)
    original = session.path.read_bytes()
    session.annotate_invocation("a1", visible_plate_count=count, evaluable=evaluable)
    session.complete()
    assert has_valid_ready_snapshot(session)
    assert session.statistics()["summary"]["evaluable_mt_invocations"] == 0
    assert session.path.read_bytes() == original


@pytest.mark.parametrize("status", [None, "", "UNKNOWN", "READ", "NO_CHARACTERS"])
@pytest.mark.parametrize("prediction", ["WI1234A", ""])
def test_review_legacy_crop_draft_is_never_confirmation(tmp_path, status, prediction):
    sample = dict(capture_id="c1", subject_key="s/legacy", prediction=prediction)
    if status is not None:
        sample["mz_status"] = status
    path = make_bundle(tmp_path / "input.alprsession", samples=[sample])
    session = load(path, tmp_path)
    session.save_subject_draft("s/legacy", ground_truth="WI1234A")
    reopened = load(path, tmp_path)
    assert reopened.subject_draft("s/legacy") == "WI1234A"
    assert not reopened.predictions_visible("s/legacy")
    assert reopened.statistics()["summary"]["evaluable_reads"] == 0
    cells = observation_verification_cells(reopened, "s/legacy")
    assert cells[2] == "○" and cells[-1] == "○"
    _blocked_without_write(reopened, "Brak GT numeru")


@pytest.mark.parametrize("decision", ["confirm", "exclude"])
def test_review_legacy_crop_requires_confirmation_or_reasoned_exclusion(tmp_path, decision):
    path = make_bundle(tmp_path / "input.alprsession")
    session = load(path, tmp_path)
    original = path.read_bytes()
    key = next(iter(session.subjects))
    session.save_subject_draft(key, ground_truth="WI1234A")
    if decision == "confirm":
        session.set_subject(key, ground_truth=session.subject_draft(key))
        assert session.statistics()["summary"]["exact_reads"] == 1
    else:
        session.set_subject(key, evaluable=False)
        _blocked_without_write(session, "Brak uzasadnienia")
        session.set_subject(key, evaluable=False, note="Numer nieczytelny na zdjęciu")
        assert session.statistics()["summary"]["evaluable_reads"] == 0
    session.complete()
    assert has_valid_ready_snapshot(session)
    reopened = load(path, tmp_path)
    assert reopened.review.review_status == "COMPLETED"
    assert has_valid_ready_snapshot(reopened)
    assert observation_verification_cells(reopened, key)[2] == ("✓" if decision == "confirm" else "—")
    assert path.read_bytes() == original


@pytest.mark.parametrize("attempt_status,sample_status", [(None, None), ("", "READ"), ("UNKNOWN", None)])
def test_review_linked_sample_cannot_bypass_gt_through_missing_status(tmp_path, attempt_status, sample_status):
    row = attempt()
    if attempt_status is None:
        row.pop("mz_status")
    else:
        row["mz_status"] = attempt_status
    sample = dict(capture_id="c1", subject_key=row["subject_key"], attempt_id="a1")
    if sample_status is not None:
        sample["mz_status"] = sample_status
    session = load(make_bundle(tmp_path / "input.alprsession", samples=[sample], attempts=[row]), tmp_path)
    session.annotate_invocation("a1", visible_plate_count="one")
    session.annotate("attempt", "a1", **visual_decision("accurate"))
    _blocked_without_write(session, "Brak GT numeru")
    session.set_subject(row["subject_key"], ground_truth="WI1234A")
    session.complete()
    assert has_valid_ready_snapshot(session)


@pytest.mark.parametrize("location", ["sample", "attempt"])
def test_review_explicit_mz_not_run_needs_no_invented_number(tmp_path, location):
    row = attempt()
    row["mz_status"] = "NOT_RUN"
    sample = dict(capture_id="c1", subject_key="s/sg-1/entity-1", mz_status="NOT_RUN")
    rows = None
    if location == "attempt":
        sample.update(attempt_id="a1", mz_status="READ")  # explicit attempt status wins
        rows = [row]
    session = load(make_bundle(tmp_path / "input.alprsession", samples=[sample], attempts=rows), tmp_path)
    if rows:
        session.annotate_invocation("a1", visible_plate_count="one")
        session.annotate("attempt", "a1", **visual_decision("accurate"))
    key = next(iter(session.subjects))
    assert observation_verification_cells(session, key)[2] == "—"
    session.complete()
    assert not session.review.subjects
    assert session.statistics()["summary"]["evaluable_reads"] == 0
    assert has_valid_ready_snapshot(session)


def test_review_cancelled_legacy_crop_requires_no_number(tmp_path):
    sample = dict(capture_id="c1", subject_key="s/cancelled", stale_or_cancelled=True)
    session = load(make_bundle(tmp_path / "input.alprsession", samples=[sample]), tmp_path)
    session.complete()
    assert observation_verification_cells(session, "s/cancelled")[2] == "—"
    assert session.statistics()["summary"]["evaluable_reads"] == 0
    assert not session.review.subjects


def test_review_mt_miss_without_mz_needs_no_synthetic_gt(tmp_path):
    row = attempt(status="NO_DETECTION", prediction="")
    row["mz_status"] = "NOT_RUN"
    session = load(make_bundle(tmp_path / "input.alprsession", attempts=[row], samples=[]), tmp_path)
    original = session.path.read_bytes()
    session.annotate_invocation("a1", visible_plate_count="one")
    session.complete()
    assert not session.review.subjects and not session.review.attempt_annotations
    assert has_valid_ready_snapshot(session)
    assert session.path.read_bytes() == original


def test_review_number_without_blind_confirmation_is_pending_in_table(tmp_path):
    session = load(make_bundle(tmp_path / "input.alprsession"), tmp_path)
    key = next(iter(session.subjects))
    # A partially migrated sidecar may contain a number but no confirmation.
    changed = copy.deepcopy(session.review)
    changed.subjects[key] = {"ground_truth": "WI1234A", "evaluable": True, "gt_confirmed": False}
    session._save_change(changed)
    assert not session.predictions_visible(key)
    assert observation_verification_cells(session, key)[2] == "○"
    _blocked_without_write(session, "jawnego potwierdzenia")
    session.set_subject(key, ground_truth="WI1234A")
    session.complete()
    assert observation_verification_cells(session, key)[2] == "✓"


@pytest.mark.parametrize("choice", ["accurate", "imprecise"])
def test_review_positive_mt_still_requires_outline_judgment(tmp_path, choice):
    session = load(make_bundle(tmp_path / "input.alprsession", attempts=[attempt()]), tmp_path)
    key = next(iter(session.subjects))
    session.set_subject(key, ground_truth="WI1234A")
    session.annotate_invocation("a1", visible_plate_count="one")
    session.annotate("attempt", "a1", is_plate=True, plate_visibility="visible")
    _blocked_without_write(session, "Brak weryfikacji wskazania MT")
    session.annotate("attempt", "a1", **visual_decision(choice))
    session.complete()
    assert has_valid_ready_snapshot(session)
    assert "pose_gt_quad_mt_input_px" not in session.review.attempt_annotations["a1"]


def test_review_multiple_blocks_legacy_gt_reuse_and_one_restores_metrics(tmp_path):
    rows = children(2)
    rows[0].update(prediction="", consensus_prediction="WI1234A", capture_source="auto_zoom", camera_zoom_ratio=2)
    rows[1]["prediction"] = "WI12B4A"
    samples = [dict(capture_id=f"c{i}", attempt_id=r["attempt_id"], subject_key=r["subject_key"])
               for i, r in enumerate(rows)]
    session = load(make_bundle(tmp_path / "input.alprsession", samples=samples, attempts=rows), tmp_path)
    original = session.path.read_bytes()
    key = next(iter(session.subjects))
    session.set_subject(key, ground_truth="WI1234A")
    saved_gt = copy.deepcopy(session.review.subjects[key])
    session.annotate_invocation("mt-x", visible_plate_count="one")
    before = session.statistics()
    assert before["summary"]["evaluable_reads"] == 2 and before["summary"]["gt_characters"] == 14
    session.annotate_invocation("mt-x", visible_plate_count="multiple")
    after = session.statistics()
    assert after["summary"]["evaluable_reads"] == 0
    assert after["summary"]["evaluable_subjects"] == 0
    assert after["summary"]["gt_characters"] == 0
    assert after["summary"]["cer"] is None
    assert after["reads"] == [] and after["character_confusion"] == []
    assert after["subjects"][0]["ground_truth"] == ""
    assert inventory(session, "mt-x") is None
    assert session.review.subjects[key] == saved_gt  # guard does not erase saved data
    _blocked_without_write(session, "dokładną liczbę tablic GT")
    session.annotate_invocation("mt-x", visible_plate_count="one")
    restored = session.statistics()
    keys = ("exact_reads", "incorrect_reads", "no_reads", "cer", "gt_characters", "correct_characters",
            "incorrect_characters", "missing_characters", "extra_characters", "subject_success_rate",
            "median_time_to_first_exact_ms", "median_attempts_to_first_exact")
    assert {k: restored["summary"][k] for k in keys} == {k: before["summary"][k] for k in keys}
    assert restored["reads"] == before["reads"]
    assert restored["character_confusion"] == before["character_confusion"]
    assert session.path.read_bytes() == original


def _navigation_method():
    """Execute the real P2B1 callback without starting a Tk window."""
    path = Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/z4_mobile_sample_review.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MobileSampleReviewWindow")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_multi_open_selected_attempt")
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"inventory": inventory}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize("matched,active", [(True, "GT01"), (False, "GT01"), (True, "not-a-plate")])
def test_review_p2b1_navigation_only_uses_persisted_matching(tmp_path, matched, active):
    rows = children(2)
    session = load(make_bundle(tmp_path / "input.alprsession", samples=[], attempts=rows), tmp_path)
    session.annotate_invocation("mt-x", visible_plate_count="multiple")
    set_count(session, "mt-x", 2)
    if matched:
        session.annotate("attempt", rows[0]["attempt_id"], **visual_decision("accurate"))
        assign_attempt(session, "mt-x", "GT01", rows[0]["attempt_id"])
    before = copy.deepcopy(session.review.to_dict())
    sidecar = session.sidecar_path.read_bytes()
    calls, messages = [], []
    obj = SimpleNamespace(session=session, current_invocation="mt-x", _multi_active_gt=active,
        _multi_group=lambda: session.mt_invocations["mt-x"], _multi_focus_attempt=calls.append,
        status_var=SimpleNamespace(set=messages.append),
        _multi_match_choice=SimpleNamespace(get=lambda: "unsaved other detection"))
    assert _navigation_method()(obj) == "break"
    assert calls == ([rows[0]["attempt_id"]] if matched and active == "GT01" else [])
    if not calls:
        assert messages and "nie ma zapisanego powiązania" in messages[-1]
    assert session.review.to_dict() == before
    assert session.sidecar_path.read_bytes() == sidecar
