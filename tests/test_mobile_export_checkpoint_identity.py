import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from auto_annotation_tool.gui import z4_model_export as export
from auto_annotation_tool.training.model_provenance import build_checkpoint_metric_summary
from auto_annotation_tool.exporters.mobile_model_exporter import (
    MobileExportError, MobileExportRequest, MobileModelExporter, validate_checkpoint_metadata,
)
from test_mobile_export_project_sources import _ExportHost, _FakeConfig


class Host(_ExportHost):
    _json_safe_training_value = staticmethod(export._mobile_export_manifest_safe_value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_run(tmp_path, target="plate"):
    torch = pytest.importorskip("torch")
    config = _FakeConfig(tmp_path)
    project = config.DIR_9_PROJECTS / "demo_123"
    history_dir = project / "5_training_runs"
    run_id = "20260906_103421"
    best = history_dir / run_id / "train/weights/best.pt"
    best.parent.mkdir(parents=True)
    checkpoint_metrics = {"metrics/mAP50(B)": 0.9, "metrics/mAP50-95(B)": 0.78}
    torch.save({"epoch": -1, "optimizer": None, "train_metrics": checkpoint_metrics,
                "train_results": {"epoch": [135, 144, 150], "metrics/mAP50(B)": [0.9, 0.95, 0.91],
                                  "metrics/mAP50-95(B)": [0.78, 0.85, 0.8]}}, best)
    run = {
        "id": run_id, "name": "training", "created_at": "2026-09-06T10:34:21", "status": "completed",
        "training_target": target, "base_model": "yolo26s-pose.pt" if target == "plate" else "yolo26s.pt",
        "epochs": 150, "current_epoch": 150, "img_size": 512,
        "output_dir": str(history_dir / run_id), "best_weights": str(best),
        "metrics_history": [{"epoch": 135, "map50": 0.9, "map50_95": 0.78},
                            {"epoch": 144, "map50": 0.95, "map50_95": 0.85},
                            {"epoch": 150, "map50": 0.91, "map50_95": 0.8}],
        "best_map50": 0.95, "best_map50_95": 0.85,
        "output_checkpoint_snapshot": {"best_epoch": 135, "best_epoch_source": "checkpoint",
                                       "best_checkpoint_sha256": digest(best)},
    }
    history_file = history_dir / "training_history.json"
    history_file.write_text(json.dumps({"runs": {run_id: run}}), encoding="utf-8")
    copied = config.get_trained_models_dir(target) / "renamed_copy.pt"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(best.read_bytes())
    copied.with_suffix(".pt.metadata.json").write_text(json.dumps({
        "model": {"file_name": copied.name, "info": {"architecture_label": "YOLO26s", "task": "pose" if target == "plate" else "detect"}},
        "run_snapshot": {**run, "current_epoch": 150}, "metrics": {"best_epoch": 150},
    }), encoding="utf-8")
    campaign = SimpleNamespace(get_active_project_name=lambda: "demo",
                               get_active_project_root_dir=lambda: project)
    return config, campaign, run, best, copied, history_file


@pytest.mark.parametrize("target", ["plate", "char"])
def test_copies_share_profile_and_written_manifest_for_mt_and_mz(tmp_path, target):
    config, campaign, run, best, copied, history_file = fixture_run(tmp_path, target)
    history_before = history_file.read_bytes()
    host = Host()
    with patch.object(export, "CONFIG", config), patch.object(export, "CAMPAIGN", campaign):
        export._MOBILE_EXPORT_RUN_SNAPSHOT_INDEX = None
        candidates = export._collect_mobile_export_candidates(host)
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate["run"].id == run["id"]
        assert candidate["project_name"] == "demo"
        assert set(candidate["alternate_paths"]) == {str(best), str(copied)}
        profile = dict(export._mobile_export_candidate_detail_rows(candidate))
        assert profile["Najlepsza epoka"] == "135"
        assert "135" in profile["Najlepszy checkpoint"]
        assert profile["Rodzina modelu"] != "-"
        assert candidate["best_map50_95"] == 0.78
        metadata = export._build_mobile_export_metadata(host, candidate["run"], target, candidate["best_weights"])
        metadata["candidate"] = export._mobile_export_candidate_manifest_snapshot(candidate)
        validate_checkpoint_metadata(best, metadata)
        role = "plate" if target == "plate" else "character"
        request = MobileExportRequest(checkpoint=best, destination=tmp_path / "test.alprmodel",
                                      role=role, formats=("onnx",), image_size=512, metadata=metadata)
        manifest = MobileModelExporter()._build_manifest(
            request=request, model_id="test", role=role, task="pose" if target == "plate" else "detect",
            width=512, height=512, labels=["plate"] if target == "plate" else list("AB"),
            variants=[], checkpoint=best, model_info={})
        assert manifest["source"]["checkpoint_sha256"] == digest(best)
        for section in ("training", "metrics", "candidate"):
            assert manifest[section]["best_epoch"] == 135
        assert manifest["training"]["run_epochs_completed"] == 150
        assert manifest["candidate"]["project_name"] == "demo"
        assert manifest["metrics"]["best_map50_95"] == 0.78
    assert history_file.read_bytes() == history_before


def test_same_architecture_with_different_weights_is_not_deduplicated(tmp_path):
    paths = [tmp_path / name / "yolo26n.pt" for name in ("first", "second")]
    for index, path in enumerate(paths):
        path.parent.mkdir()
        path.write_bytes(bytes([index + 1]))
    candidates = [{"best_weights": path, "role": "vehicle", "model_version": "YOLO26n",
                   "scope": "Bazowe modele pojazdów", "run": None} for path in paths]
    assert len(export._dedupe_mobile_export_candidates(candidates)) == 2


def test_separate_training_stages_are_not_silently_relabelled(tmp_path):
    paths = [tmp_path / name for name in ("first.pt", "second.pt")]
    for path in paths:
        path.write_bytes(b"same model")
    candidates = [{"best_weights": path, "run": SimpleNamespace(id=f"stage_{index}")}
                  for index, path in enumerate(paths)]
    assert len(export._dedupe_mobile_export_candidates(candidates)) == 2


def test_filename_timestamp_never_borrows_another_models_training(tmp_path):
    config, _, run, _, _, _ = fixture_run(tmp_path)
    unrelated = tmp_path / f"plate_{run['id']}_map99.pt"
    unrelated.write_bytes(b"other model")
    with patch.object(export, "CONFIG", config):
        export._MOBILE_EXPORT_RUN_SNAPSHOT_INDEX = None
        assert export._mobile_export_run_snapshot_for_reference(unrelated) == {}


def test_snapshot_lookup_matches_renamed_weights_and_reloads_history(tmp_path):
    config, _, run, _, copied, history_file = fixture_run(tmp_path)
    with patch.object(export, "CONFIG", config):
        export._MOBILE_EXPORT_RUN_SNAPSHOT_INDEX = None
        assert export._mobile_export_run_snapshot_for_reference(copied)["name"] == "training"
        run["name"] = "updated training"
        history_file.write_text(json.dumps({"runs": {run["id"]: run}}), encoding="utf-8")
        assert export._mobile_export_run_snapshot_for_reference(copied)["name"] == "updated training"


def test_nested_model_and_metadata_changes_invalidate_candidate_cache(tmp_path):
    config, campaign, _, _, copied, _ = fixture_run(tmp_path)
    with patch.object(export, "CONFIG", config), patch.object(export, "CAMPAIGN", campaign):
        first = export._mobile_export_candidates_source_signature(Host())
        copied.write_bytes(b"replaced")
        second = export._mobile_export_candidates_source_signature(Host())
        assert first != second
        copied.with_suffix(".pt.metadata.json").write_text("{}", encoding="utf-8")
        assert second != export._mobile_export_candidates_source_signature(Host())


def test_actual_checkpoint_epoch_overrides_wrong_but_hash_matching_snapshot(tmp_path):
    _, _, run, best, _, _ = fixture_run(tmp_path)
    run["output_checkpoint_snapshot"]["best_epoch"] = 150
    summary = build_checkpoint_metric_summary(run, checkpoint=best)
    assert summary["best_epoch"] == 135
    assert summary["best_map50_95"] == 0.78


def test_checkpoint_epoch_cache_is_invalidated_when_weights_change(tmp_path):
    torch = pytest.importorskip("torch")
    _, _, run, best, _, _ = fixture_run(tmp_path)
    assert build_checkpoint_metric_summary(run, checkpoint=best)["best_epoch"] == 135
    torch.save({"epoch": 140}, best)
    summary = build_checkpoint_metric_summary(run, checkpoint=best)
    assert summary["best_epoch"] == 141
    assert summary["checkpoint_mismatch"] is True
    assert summary["best_map50_95"] is None


@pytest.mark.parametrize("field", ["metrics", "candidate"])
def test_export_rejects_conflicting_epoch_sections(tmp_path, field):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    metadata = {"training": {"best_epoch": 135}, field: {"best_epoch": 150}}
    with pytest.raises(MobileExportError, match="różne najlepsze epoki"):
        validate_checkpoint_metadata(checkpoint, metadata)


@pytest.mark.parametrize("section", ["candidate", "source"])
def test_export_rejects_weights_replaced_after_selection(tmp_path, section):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"original")
    metadata = {section: {"checkpoint_sha256": digest(checkpoint)}}
    checkpoint.write_bytes(b"modified")
    with pytest.raises(MobileExportError, match="SHA-256"):
        validate_checkpoint_metadata(checkpoint, metadata)


def test_untrained_model_has_no_invented_best_epoch(tmp_path):
    checkpoint = tmp_path / "yolo26n.pt"
    checkpoint.write_bytes(b"untrained")
    summary = build_checkpoint_metric_summary({}, checkpoint=checkpoint)
    assert summary["best_epoch"] is None
    assert summary["best_map50_95"] is None
