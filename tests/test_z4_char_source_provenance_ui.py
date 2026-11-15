from __future__ import annotations

import json
from pathlib import Path

from auto_annotation_tool.gui.z4_shared_ui import _build_char_pz3_source_provenance


def _manifest(prefix: str, char: str = "A") -> dict:
    return {
        "schema": "alpr.pz3.gold_export_artifact.v1",
        "dataset_type": "char_yolo_detect",
        "plate_count": 1,
        "character_count": 1,
        "gold_source_contract_schema": "alpr.pz3.gold_source_contract.v1",
        "gold_source_contract_sha256": "a" * 64,
        "gt_contract_fingerprint_sha256": "b" * 64,
        "items": [
            {
                "pid": f"{prefix}_physical",
                "source_pid": "plate_000001",
                "source_image": r"C:\source\ABC123.jpg",
                "image_path": f"images/{prefix}.jpg",
                "label_path": f"labels/{prefix}.txt",
                "source_bucket": "local_manual",
                "strategy_bucket": "other_perfect",
                "provenance": {"category": "manual"},
                "layout": {"plate_layout_label": "1R*"},
                "characters": [{"char": char, "bbox": [1, 2, 3, 4]}],
            }
        ],
    }


def _write(root: Path, manifest: dict) -> None:
    root.mkdir(parents=True)
    (root / "metadata_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )


def test_reexport_is_recognized_as_same_logical_material(tmp_path):
    first = tmp_path / "YOLO_MegaDataset_Chars_20260927_184942"
    second = tmp_path / "YOLO_MegaDataset_Chars_20260927_190227"
    _write(first, _manifest("old"))
    _write(second, _manifest("new"))

    info = _build_char_pz3_source_provenance(second)

    assert info["ok"] is True
    assert info["equivalent_exports"] == [first.name]
    assert info["logical_id"].startswith("MAT-ZN-")


def test_changed_annotation_is_not_marked_equivalent(tmp_path):
    first = tmp_path / "YOLO_MegaDataset_Chars_20260927_184942"
    second = tmp_path / "YOLO_MegaDataset_Chars_20260927_190227"
    _write(first, _manifest("old", char="A"))
    _write(second, _manifest("new", char="C"))

    info = _build_char_pz3_source_provenance(second)

    assert info["equivalent_exports"] == []


def test_operator_table_uses_plain_language_and_hides_codes():
    path = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z4_dataset_panels.py"
    )
    source = path.read_text(encoding="utf-8-sig")

    assert '("source", "Dane wejściowe")' in source
    assert '("identity", "Czy dane zmieniły się od poprzedniego eksportu?")' in source
    assert '("variant", "Wariant do treningu")' in source
    assert '"Co sprawdzam", "Stan", "Wyjaśnienie"' in source
    assert 'text="Szczegóły techniczne"' in source


def test_operator_copy_is_plain_and_unambiguous():
    root = Path(__file__).resolve().parents[1]
    shared = (
        root
        / "auto_annotation_tool"
        / "gui"
        / "z4_shared_ui.py"
    ).read_text(encoding="utf-8-sig")
    panels = (
        root
        / "auto_annotation_tool"
        / "gui"
        / "z4_dataset_panels.py"
    ).read_text(encoding="utf-8-sig")

    assert 'source_explanation = "Przygotowane w PZ3 w tej iteracji."' in shared
    assert 'identity_state = "NIE — TO TEN SAM MATERIAŁ"' in shared
    assert "Ten eksport zawiera te same tablice i oznaczenia co poprzedni." in shared
    assert "Możesz bezpiecznie kontynuować." in shared

    assert '("identity", "Czy dane zmieniły się od poprzedniego eksportu?")' in panels
    assert "self.split_source_technical_row = ttk.Frame(f)" in panels
    assert 'text="Szczegóły techniczne"' in panels
    assert "self.btn_split_source_technical_details.pack(side=tk.RIGHT)" in panels
