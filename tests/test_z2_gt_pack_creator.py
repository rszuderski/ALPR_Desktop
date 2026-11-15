from pathlib import Path

import pytest

from auto_annotation_tool.gt_resource_companions import (
    DEFAULT_WORKING_PACK_NAME,
    ensure_working_gt_pack,
)


def test_ensure_working_gt_pack_uses_canonical_name(tmp_path):
    result = ensure_working_gt_pack(tmp_path)

    assert result["created"] is True
    assert result["path"] == tmp_path / DEFAULT_WORKING_PACK_NAME
    assert (result["path"] / "manifest.json").is_file()
    assert result["summary"]["images"] == 0
    assert result["summary"]["plates"] == 0


def test_ensure_working_gt_pack_reopens_without_overwrite(tmp_path):
    first = ensure_working_gt_pack(tmp_path)
    marker = first["path"] / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    second = ensure_working_gt_pack(tmp_path)

    assert second["created"] is False
    assert second["path"] == first["path"]
    assert marker.read_text(encoding="utf-8") == "keep"


def test_ensure_working_gt_pack_rejects_invalid_existing_target(tmp_path):
    target = tmp_path / DEFAULT_WORKING_PACK_NAME
    target.mkdir()
    (target / "not-a-manifest.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        ensure_working_gt_pack(tmp_path)


def test_gt_pack_is_not_presented_as_a_manual_prerequisite_button():
    root = Path(__file__).resolve().parents[1]
    widget_source = (root / "auto_annotation_tool/gui/z2_main_widgets.py").read_text(
        encoding="utf-8-sig"
    )
    tab_source = (root / "auto_annotation_tool/gui/tab_annotation.py").read_text(
        encoding="utf-8-sig"
    )

    assert 'text="Utwórz roboczy pakiet GT…"' not in widget_source
    assert 'text="Zapisuj GT znaków dla modelu Detect"' not in widget_source
    assert "gt_capture_enabled_var" not in tab_source
