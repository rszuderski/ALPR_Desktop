from pathlib import Path


def test_gold_export_manifest_contains_aggregate_gt_contract():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "auto_annotation_tool/gui/z3_goldpack_ui.py"
    ).read_text(encoding="utf-8-sig")

    assert "collect_dataset_revision_ids(" in source
    assert '"gt_contract_fingerprint_sha256"' in source
    assert '"gt_revision_ids"' in source
    assert '"geometry_revision_ids"' in source


def test_gold_export_manifest_contains_az_reuse_summary():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "auto_annotation_tool/gui/z3_goldpack_ui.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def run_yolo_gold_export(")
    end = source.index("\ndef run_pz3_existing_dataset_split(", start)
    body = source[start:end]

    assert "summarize_dataset_az_reuse(" in body
    assert '"az_reuse_summary"' in body
