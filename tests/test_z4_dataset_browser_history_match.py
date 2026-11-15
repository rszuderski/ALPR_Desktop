from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source():
    return (ROOT / "auto_annotation_tool/gui/z4_dataset_builder.py").read_text(
        encoding="utf-8-sig"
    )


def test_browser_recursively_scans_nested_dataset_yaml():
    text = source()
    assert 'for yaml_path in base.rglob("data.yaml")' in text
    assert "len(rel_parts) > 4" in text


def test_browser_can_match_moved_dataset_by_historical_fingerprint():
    text = source()
    assert "snapshot_records" in text
    assert "build_training_dataset_snapshot(" in text
    assert 'record.get("split_sha256")' in text
    assert 'record.get("data_yaml_sha256")' in text


def test_browser_preserves_historical_dataset_id_after_match():
    text = source()
    assert 'record.get("dataset_id")' in text
    assert "canonical_id = snapshot_ids[0] if snapshot_ids else ui_id" in text


def test_explorer_action_is_clearly_not_dataset_selection():
    text = source()
    assert 'text="Pokaż w Eksploratorze"' in text
    assert "wyboru dokonujesz tutaj" in text
    assert 'text="Wybierz dataset"' in text
