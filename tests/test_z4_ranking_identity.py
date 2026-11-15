import hashlib
import json
from pathlib import Path

from auto_annotation_tool.gui import z4_ranking_identity as identity


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_source_population_is_not_historical_sample_or_matching_key(tmp_path):
    track = tmp_path / "run_source_720"
    track.mkdir()
    (track / "annotations.xml").write_text("<annotations/>")
    write(track / "run_manifest.json", {"generated_at": "2026-09-01T10:00:00", "input_dir": "raw/source"})
    records = [{"reference_path": str(track), "total_images": 500},
               {"reference_path": str(tmp_path / "different_720"), "total_images": 720}]
    row = identity.track_evidence({"path": str(track), "counts": {"total": 720}, "ready": True}, records, tmp_path)
    assert row["source_count"] == 720 and row["evaluated_counts"] == [500]
    assert len(row["evaluations"]) == 1 and row["z2_id"].startswith("Z2-")
    assert row["created"] == "2026-09-01T10:00:00" and row["date_origin"] == "metadane"
    assert row["directory"] == track.name and row["xml_exists"]
    assert "raw/source" in row["search"]


def test_raw_history_preserves_extra_fields_and_repeated_evaluations(tmp_path):
    file = tmp_path / "7_rankings/plates/model_ranking.json"
    rows = [{"reference_path": "C:/missing/track", "task_type": "Tablice (Pose)", "total_images": 500,
             "model_path": "C:/models/n.pt", "date_evaluated": date, "experiment_id": "EXPERIMENT",
             "benchmark_group_metrics": {"groups": [{"sample_count": 500}]}}
            for date in ("2026-09-01", "2026-09-02")]
    write(file, {"entries": rows})
    before = file.read_bytes()
    catalog = identity.ranking_catalog(tmp_path, [file])
    assert len(catalog["rows"]) == 2
    assert catalog["rows"][1]["experiment_id"] == "EXPERIMENT"
    assert catalog["rows"][1]["_record_index"] == 1
    tracks = identity.historical_tracks([], catalog, tmp_path, "plate")
    assert len(tracks) == 1 and tracks[0]["status"] == "brak katalogu"
    assert tracks[0]["evaluated_counts"] == [500] and not tracks[0]["ready"]
    assert file.read_bytes() == before


def test_missing_reference_does_not_join_by_name_or_count(tmp_path):
    catalog = {"rows": [{"reference_name": "run_A", "total_images": 500, "task_type": "Tablice (Pose)"}]}
    tracks = identity.historical_tracks([{"path": str(tmp_path / "run_A"), "split_count": 500}], catalog, tmp_path, "plate")
    assert len(tracks) == 1 and not tracks[0]["evaluations"]


def test_corrupt_history_is_explicit(tmp_path):
    file = tmp_path / "7_rankings/plates/model_ranking.json"
    file.parent.mkdir(parents=True)
    file.write_text("broken")
    catalog = identity.ranking_catalog(tmp_path)
    assert not catalog["rows"] and catalog["errors"]


def test_unknown_model_architecture_not_guessed_from_filename(tmp_path):
    path = tmp_path / "fake_yolo26n_pose_MT_best.pt"
    path.write_bytes(b"not loaded")
    guessed_run = {"id": "best", "base_model": "yolo26s-pose.pt", "current_epoch": 500}
    row = identity.model_evidence(path, [guessed_run], tmp_path, "plate")
    assert all(row[key] == identity.UNKNOWN for key in ("architecture", "size", "role", "run_id", "date"))
    assert row["completed"] is None and row["checkpoint"] == path.name
    assert identity.actual_sha256(path) == hashlib.sha256(b"not loaded").hexdigest()


def test_model_metadata_and_explicit_history_link(tmp_path):
    path = tmp_path / "9_projects/projectA/6_models/model.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"test checkpoint")
    metadata = {"model": {"file_name": "model.pt", "info": {"architecture_label": "YOLO26s Pose", "model_scale": "s",
                "task": "pose", "classes": ["plate"]}},
                "training": {"run_id": "R1", "run_epochs_completed": 30, "epochs": 100}}
    write(path.with_suffix(".pt.metadata.json"), metadata)
    row = identity.model_evidence(path, [{"id": "R1", "created_at": "2026-08-01", "_history_file": "history.json"}], tmp_path, "plate")
    assert (row["role"], row["architecture"], row["size"]) == ("MT", "YOLO26s Pose", "s")
    assert row["epochs"] == "30 / 100" and row["date"] == "2026-08-01"
    assert row["scope"] == "Projekt: projectA" and row["history_file"] == "history.json"
    assert "r1" in row["search"] and str(path).casefold().replace('\\', '/') in row["search"]


def test_wrong_file_metadata_and_ambiguous_run_are_not_used(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"test")
    write(path.with_suffix(".pt.metadata.json"), {"model": {"file_name": "other.pt", "info": {"model_scale": "s"}}})
    row = identity.model_evidence(path, [], tmp_path, "plate")
    assert row["size"] == identity.UNKNOWN and "nie odpowiadają" in row["metadata_error"]
    write(path.with_suffix(".pt.metadata.json"), {"model": {"file_name": "model.pt"}, "training": {"run_id": "R1"}})
    row = identity.model_evidence(path, [{"id": "R1"}, {"id": "R1"}], tmp_path, "plate")
    assert row["provenance_warning"] and row["date"] == identity.UNKNOWN


def test_local_500_comparison_recovered_by_recorded_reference_path():
    workspace = Path(__file__).resolve().parents[1] / "Workspace"
    file = workspace / "7_rankings/plates/model_ranking.json"
    if not file.exists():
        import pytest
        pytest.skip("Local historical data")
    catalog = identity.ranking_catalog(workspace)
    rows = [r for r in catalog["rows"] if r.get("reference_name") == "Porownanie modeli MT-s_vs_n v1"]
    assert len(rows) == 2 and all(r["total_images"] == 500 for r in rows)
    assert len({identity.path_key(r["reference_path"]) for r in rows}) == 1
    history = identity.history_catalog(workspace)
    models = [identity.model_evidence(r["model_path"], history, workspace, "plate") for r in rows]
    assert {m["architecture"] for m in models} == {"YOLO26n Pose", "YOLO26s Pose"}
    assert all(identity.actual_sha256(r["model_path"]) == r["model_sha256"] for r in rows)
