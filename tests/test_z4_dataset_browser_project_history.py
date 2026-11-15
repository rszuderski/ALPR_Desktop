from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source():
    return (ROOT / "auto_annotation_tool/gui/z4_dataset_builder.py").read_text(
        encoding="utf-8-sig"
    )


def test_browser_scans_project_dataset_roots():
    text = source()
    assert "CONFIG.DIR_9_PROJECTS" in text
    assert 'project_root / "4_training_datasets"' in text
    assert 'project_root / "4_training_datasets" / "plates"' in text


def test_browser_scans_project_training_histories():
    text = source()
    assert 'project_root / "5_training_runs"' in text
    assert "reconcile_on_load=False" in text


def test_history_dataset_is_added_even_outside_global_dataset_root():
    text = source()
    assert "Jeżeli dataset jest wskazany przez historię" in text
    assert '_add_candidate(run_root, "plate", 0.0)' in text


def test_browser_still_uses_history_snapshot_dataset_id():
    text = source()
    assert 'payload.get("dataset_id")' in text
    assert "canonical_id = snapshot_ids[0] if snapshot_ids else ui_id" in text
