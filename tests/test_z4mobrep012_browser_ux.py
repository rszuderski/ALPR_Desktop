"""Regression of presentation-only mobile report browser improvements."""
from pathlib import Path
from types import SimpleNamespace
import json
import importlib
import pytest

from auto_annotation_tool.gui.z4_mobile_report_browser_ux import (
    report_identity, matches_report_filter, report_filter_blob, selected_report_caption,
    sidecar_preview, source_archive_sha, desktop_status,
)

SHA = "1" * 64


def report(**k):
    base = dict(
        report_id="r-TEST", measured_at="2026-10-16T12:39:47Z",
        variant_id="mp-no-mp-mt-tflite-fp32-mz-tflite-fp32",
        package_id="ALPR-MT-260906-0039-4323-MZ-261005-1035-D12C",
        device={"manufacturer": "samsung", "model": "SM-A125F"},
        source_archive_sha256=SHA, experiment_index={
            "series_id": "F4-TV-FP32-A12-20261016", "experiment_variant": "F4-TV-FP32-B",
            "scenario_id": "TV6X10_STATIC_B", "experiment_session_id": "exp-abc"},
        raw={},
    )
    base.update(k)
    return SimpleNamespace(**base)


def review_file(root: Path, **updates):
    p = root / "human_reviews" / f"{SHA}.review.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    data = dict(source_archive_sha256=SHA, session_id="exp-abc", review_status="IN_PROGRESS",
                review_revision=60, subjects={"one": {"ground_truth": "ABC123"}, "two": {"ground_truth": ""}})
    data.update(updates)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_original_and_experiment_variant_distinct():
    info = report_identity(report())
    assert info["variant"] == "F4-TV-FP32-B"
    assert info["technical_variant"].startswith("mp-no-mp")
    assert info["scenario"] == "TV6X10_STATIC_B"
    assert info["date"] == "16.10 12:39"
    assert info["device"] == "samsung SM-A125F"


def test_bundle_identity_preferred():
    bundle = SimpleNamespace(experiment_session={"experiment_variant": "F4-TV-FP32-C", "session_id": "exp-C"})
    assert report_identity(report(), bundle)["variant"] == "F4-TV-FP32-C"


def test_fallback_technical_variant_without_experiment():
    info = report_identity(report(experiment_index={}, raw={}))
    assert info["variant"].startswith("mp-no-mp")
    assert len(info["variant_short"]) <= 21


def test_filter_by_tokens_and_hidden_full_package():
    r = report()
    assert matches_report_filter(r, "F4-TV-FP32-B A125F")
    assert matches_report_filter(r, "260906 tflite")
    assert matches_report_filter(r, "16.10 12:39")
    assert not matches_report_filter(r, "F4-TV-FP32-A")
    assert not matches_report_filter(r, "unrelated")
    assert matches_report_filter(r, "")


def test_report_caption_keeps_full_id():
    txt = selected_report_caption(report())
    assert "ALPR-MT-260906" in txt
    assert "TV6X10_STATIC_B" in txt
    assert "exp-abc" in txt


def test_sidecar_in_progress_is_not_marked_completed(tmp_path):
    review_file(tmp_path)
    p = sidecar_preview(tmp_path, SHA, "exp-abc")
    assert p["state"] == "in_progress"
    assert p["revision"] == 60 and p["gt_entries"] == 1
    assert "nie liczba ukończonych ocen" in p["detail"]


def test_sidecar_completed(tmp_path):
    review_file(tmp_path, review_status="COMPLETED")
    p = desktop_status(report(), None, tmp_path)
    assert p["state"] == "completed"


def test_sidecar_missing(tmp_path):
    assert sidecar_preview(tmp_path, SHA, "exp-abc")["state"] == "none"


def test_invalid_sha_rejected_without_path_traversal(tmp_path):
    assert sidecar_preview(tmp_path, "../bad", "exp-abc")["state"] == "unverified"


def test_incorrect_sha_in_sidecar_rejected(tmp_path):
    review_file(tmp_path, source_archive_sha256="2" * 64)
    assert sidecar_preview(tmp_path, SHA, "exp-abc")["state"] == "error"


def test_incorrect_session_rejected(tmp_path):
    review_file(tmp_path, session_id="exp-other")
    assert sidecar_preview(tmp_path, SHA, "exp-abc")["state"] == "error"


def test_corrupted_json_rejected(tmp_path):
    path = review_file(tmp_path)
    path.write_text('{"invalid":', encoding="utf-8")
    assert sidecar_preview(tmp_path, SHA, "exp-abc")["state"] == "error"


def test_nonstandard_json_and_large_file_rejected(tmp_path):
    path = review_file(tmp_path)
    path.write_bytes(b"a" * (4 * 1024 * 1024 + 1))
    assert sidecar_preview(tmp_path, SHA, "exp-abc")["state"] == "error"


def test_invalid_state_rejected(tmp_path):
    review_file(tmp_path, review_status="SOMETHING")
    assert sidecar_preview(tmp_path, SHA, "exp-abc")["state"] == "error"


def test_status_does_not_expose_gt_text(tmp_path):
    review_file(tmp_path, subjects={"1": {"ground_truth": "SENSITIVE123"}})
    p = sidecar_preview(tmp_path, SHA, "exp-abc")
    assert "SENSITIVE123" not in json.dumps(p)


def test_archive_hash_without_bundle():
    assert source_archive_sha(report()) == SHA


def test_browser_has_split_filter_and_source_distinction():
    path = Path(__file__).resolve().parents[1] / "auto_annotation_tool" / "gui" / "z4_mobile_report_browser.py"
    source = path.read_text(encoding="utf-8")
    assert "Z4MOBREP012_READABLE_MOBILE_REPORTS" in source
    assert "tk.PanedWindow" in source
    # Search is verified by the widget and live filtering, not a cosmetic label.
    assert "self.search_entry.grid(" in source
    assert 'self.search_var.trace_add("write"' in source
    assert "matches_report_filter(r, query" in source
    assert '"GT z telefonu"' in source
    assert "_apply_report_filter" in source
    assert "_show_validation_details" in source
    assert "def _on_report_selected" in source
