"""Native MT report exports preserve raw evidence and partial-run semantics."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from auto_annotation_tool.ranking import mt_eval_report as exports
from auto_annotation_tool.ranking.mt_eval_metrics import aggregate, METRIC_PROTOCOL


def _quad(x):
    return [[x, 10.0], [x + 40.0, 10.0], [x + 40.0, 30.0], [x, 30.0]]


@pytest.fixture
def campaign(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    scene_day = {"sha256": "a" * 64, "domain": "DAY", "file": "DAY/scene.jpg", "image": {"oriented_size": [300, 100]},
                 "plates": [{"plate_id": "regular", "bbox": [10, 10, 50, 30], "quad": _quad(10), "bbox_eligible": True,
                             "pose_eligible": True},
                            {"plate_id": "plate_000421", "bbox": [150, 10, 190, 30], "quad": list(reversed(_quad(150))),
                             "bbox_eligible": True, "pose_eligible": False, "pose_evaluation_status": "EXCLUDED_BY_OPERATOR",
                             "pose_exclusion": {"reason": "INDEX_IDENTITY_AMBIGUOUS"}}]}
    scene_night = {"sha256": "b" * 64, "domain": "NIGHT", "file": "NIGHT/empty.jpg", "image": {"oriented_size": [300, 100]},
                   "plates": [{"plate_id": "missed", "bbox": [10, 10, 50, 30], "quad": _quad(10), "bbox_eligible": True,
                               "pose_eligible": True}]}
    predictions = [
        {"index": 0, "bbox": [10, 10, 50, 30], "confidence": .937465123, "class_id": 0,
         "keypoints": [[x + .123456789, y + .25, .138765432] for x, y in _quad(10)]},
        {"index": 1, "bbox": [150, 10, 190, 30], "confidence": .8, "class_id": 0, "keypoints": _quad(150)},
        {"index": 2, "bbox": [10, 10, 50, 30], "confidence": .4, "class_id": 0, "keypoints": _quad(10)},
        {"index": 3, "bbox": [10, 10, 50, 30], "confidence": .25, "class_id": 0, "keypoints": _quad(10)},
    ]
    row_day = {"scene_sha256": scene_day["sha256"], "domain": "DAY", "file": scene_day["file"], "width": 300, "height": 100,
               "predictions": predictions, "elapsed_ms": 12.987654321}
    row_night = {"scene_sha256": scene_night["sha256"], "domain": "NIGHT", "file": scene_night["file"], "width": 300, "height": 100,
                 "predictions": [], "elapsed_ms": 11.123456789}
    scenes = [scene_day, scene_night]
    rows = [row_day, row_night]
    protected = run / "records.json"
    protected.write_text(json.dumps(rows), encoding="utf-8")
    preflight = {"selection_sha": "selection", "gt_freeze_sha": "freeze", "metric_policy_sha256": "policy",
                 "metric_policy": {"id": "EVAL396_POSE478_BBOX479_v1", "exclusions": [{"plate_id": "plate_000421"}]},
                 "selection_dir": str(tmp_path / "experiment/evaluation_selections/selection"),
                 "freeze_dir": str(tmp_path / "experiment/frozen_gt/freeze"), "source_scene_root": str(tmp_path / "images"),
                 "scenes": scenes}
    return {"run_dir": str(run), "status": "COMPLETE", "preflight": preflight,
            "protocol": {"protocol_sha256": "protocol", "geometry_approval_contract_sha256": "approval",
                         "inference": {"conf": .25, "imgsz": 512, "iou": .7, "end2end": True}, "metrics": METRIC_PROTOCOL,
                         "models": [{"label": label, "checkpoint_file": f"{label}/best.pt", "checkpoint_sha256": label + "-sha"}
                                    for label in ("MT-n", "MT-DAY", "MT-NIGHT")]},
            "geometry_approval": {"decision": "APPROVED", "operator_signature": "fixture only", "contract_sha256": "approval"},
            "models": [{"label": label, "status": "COMPLETE", "metrics": aggregate(rows, scenes), "rows": deepcopy(rows)}
                       for label in ("MT-n", "MT-DAY", "MT-NIGHT")],
            "source_files": {str(protected): hashlib.sha256(protected.read_bytes()).hexdigest()}}


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def test_complete_export_preserves_raw_evidence_and_receipts(campaign, monkeypatch):
    monkeypatch.setattr(exports, "load_mt_report", lambda _path: deepcopy(campaign))
    output_root = Path(campaign["run_dir"]) / "reports"
    result = exports.export_mt_report(campaign["run_dir"], output_root)
    status = json.loads((result / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "COMPLETE"
    assert status["export_integrity"] == "VERIFIED"
    for filename, expected in status["files_sha256"].items():
        assert hashlib.sha256((result / filename).read_bytes()).hexdigest() == expected
    summary = _csv(result / "summary.csv")
    assert len(summary) == 9
    all_rows = [row for row in summary if row["domain"] == "ALL"]
    assert all(row["box_gt"] == "3" and row["pose_gt"] == "2" for row in all_rows)
    assert all(row["pose_coverage"] == "0.5" and row["box_ap"] == row["pose_ap"] == "NOT_COMPUTED" for row in all_rows)
    assert len(_csv(result / "scenes.csv")) == 6
    assert sum(row["no_detection"] == "True" for row in _csv(result / "scenes.csv")) == 3
    predicted = _csv(result / "predictions.csv")
    assert len(predicted) == 12
    assert float(predicted[0]["confidence"]) == .937465123
    assert json.loads(predicted[0]["raw_keypoints"])[0] == [10.123456789, 10.25, .138765432]
    assert predicted[3]["outcome"] == "FILTERED_CONFIDENCE"
    corners = _csv(result / "corners.csv")
    assert len(corners) == 48
    excluded = [row for row in corners if row["matched_plate_id"] == "plate_000421"]
    assert len(excluded) == 12 and all(row["error_px"] == "" for row in excluded)
    assert corners[0]["raw_confidence"] == "0.138765432"
    matches = _csv(result / "matches.csv")
    assert sum(row["outcome"] == "UNMATCHED_GT" for row in matches) == 3
    assert sum(row["outcome"] == "UNMATCHED_PREDICTION" for row in matches) == 3
    text = (result / "report.md").read_text(encoding="utf-8")
    assert "plate_000421" in text and "NIEPOLICZONE / NOT_COMPUTED" in text
    assert "nano" in text and "nie oznacza NIGHT" in text and "retreningu" in text
    assert "Results.keypoints.data" in text and "nie są to tensory sprzed postprocessingu" in text
    assert "warmup" in text and "nie udostępniają confidence narożnika" in text
    provenance = json.loads((result / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["geometry_approval"] == campaign["geometry_approval"]
    assert provenance["inference_performed_by_export"] is False
    for filename in result.glob("*.svg"):
        ET.parse(filename)
    assert "0%" in (result / "box_f1.svg").read_text(encoding="utf-8")
    assert "100%" in (result / "box_f1.svg").read_text(encoding="utf-8")
    assert "MT-n: n=4" in (result / "corner_ecdf_ALL.svg").read_text(encoding="utf-8")
    second = exports.export_mt_report(campaign["run_dir"], output_root)
    assert result != second and result.is_dir() and second.is_dir()


def test_partial_not_run_is_not_promoted_to_pass_or_zero(campaign, monkeypatch):
    campaign["status"] = "RUNNING"
    campaign["models"][0]["status"] = "PARTIAL"
    campaign["models"][0]["rows"] = campaign["models"][0]["rows"][:1]
    campaign["models"][0]["metrics"] = aggregate(campaign["models"][0]["rows"], campaign["preflight"]["scenes"])
    for model in campaign["models"][1:]:
        model.update(status="NOT_RUN", metrics=None, rows=[])
    monkeypatch.setattr(exports, "load_mt_report", lambda _path: deepcopy(campaign))
    result = exports.export_mt_report(campaign["run_dir"], Path(campaign["run_dir"]) / "reports")
    assert json.loads((result / "status.json").read_text())["status"] == "PARTIAL"
    rows = _csv(result / "summary.csv")
    assert rows[0]["pending_scenes"] == "1" and rows[0]["box_fn"] == "0"
    assert rows[3]["box_f1"] == "" and rows[3]["model_status"] == "NOT_RUN"
    assert "WYNIK CZĘŚCIOWY" in (result / "report.md").read_text(encoding="utf-8")
    assert "NIEPOLICZONE" in (result / "box_f1.svg").read_text(encoding="utf-8")
    assert "brak ocenionych narożników" in (result / "corner_ecdf_NIGHT.svg").read_text(encoding="utf-8")


def test_source_changed_before_export_writes_nothing(campaign, monkeypatch):
    monkeypatch.setattr(exports, "load_mt_report", lambda _path: campaign)
    Path(next(iter(campaign["source_files"]))).write_text("tampered")
    output = Path(campaign["run_dir"]) / "reports"
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        exports.export_mt_report(campaign["run_dir"], output)
    assert not output.exists()


def test_source_changed_during_export_never_publishes_complete(campaign, monkeypatch):
    monkeypatch.setattr(exports, "load_mt_report", lambda _path: deepcopy(campaign))
    output = Path(campaign["run_dir"]) / "reports"
    def progress(message):
        if message.startswith("Tworzę"):
            Path(next(iter(campaign["source_files"]))).write_text("changed during report")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        exports.export_mt_report(campaign["run_dir"], output, progress=progress)
    unfinished = list(output.iterdir())
    assert len(unfinished) == 1 and unfinished[0].name.endswith(".partial")
    assert json.loads((unfinished[0] / "status.json").read_text())["status"] == "PARTIAL"


@pytest.mark.parametrize("where", ["freeze", "selection", "source", "mz", "run", "final_test"])
def test_protected_destination_is_rejected(campaign, monkeypatch, where):
    monkeypatch.setattr(exports, "load_mt_report", lambda _path: campaign)
    preflight = campaign["preflight"]
    destinations = {"freeze": preflight["freeze_dir"], "selection": preflight["selection_dir"], "source": preflight["source_scene_root"],
                    "mz": str(Path(preflight["selection_dir"]).parent.parent / "evaluation_runs/E-MZ-DN-01_EVAL396_MZ_v1"),
                    "run": campaign["run_dir"], "final_test": str(Path(campaign["run_dir"]).parent / "final_test/reports")}
    with pytest.raises(ValueError, match="PROTECTED|REPORTS_SUBDIRECTORY"):
        exports.export_mt_report(campaign["run_dir"], destinations[where])


def test_svg_escapes_model_text_and_keeps_empty_error_sample(campaign):
    campaign["models"][0]["label"] = 'MT<&"'
    svg = exports.corner_distribution_svg(campaign)
    ET.fromstring(svg)
    assert "MT&lt;&amp;&quot;" in svg
    assert len(exports._corner_values(campaign["models"][0], "ALL")) == 4
    assert exports._corner_values(campaign["models"][0], "NIGHT") == []


def test_operational_report_rejects_unapproved_ap_values(campaign):
    campaign["models"][0]["metrics"]["official_metrics"]["box_ap"] = 0.0
    with pytest.raises(ValueError, match="UNSUPPORTED_AP_PROTOCOL"):
        exports.summary_rows(campaign)
