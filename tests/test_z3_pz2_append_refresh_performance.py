from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from auto_annotation_tool.gui.z3_pz2_material_append import _refresh_after_append


def test_refresh_after_append_activates_import_focus_before_single_metadata_rebuild(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        '{"old": {"crop_id": "old"}, "new_local": {"crop_id": "new"}}',
        encoding="utf-8",
    )

    calls = []
    host = SimpleNamespace(
        _preview_import_focus_plate_ids=[],
        _preview_import_focus_batch_id="",
        _preview_import_focus_active=False,
        preview_metadata={},
        _loaded_meta_path=None,
        _loaded_meta_mtime=0,
    )

    def apply_metadata(payload, **kwargs):
        calls.append(
            (
                "apply",
                list(host._preview_import_focus_plate_ids),
                host._preview_import_focus_batch_id,
                host._preview_import_focus_active,
            )
        )
        host.preview_metadata = payload

    host._apply_preview_metadata_update = Mock(side_effect=apply_metadata)
    host._set_preview_import_focus = Mock()
    host._set_plates_legend_info = Mock()
    host._update_preview_repair_progress_ui = Mock()
    host._refresh_preview_source_panel = Mock()
    host._sync_step3_access_from_preview_state = Mock()

    result = SimpleNamespace(
        items=(SimpleNamespace(plate_id="new_local"),),
    )

    _refresh_after_append(
        host,
        metadata_path,
        result,
        "AZPKG-PERF",
    )

    assert calls == [
        ("apply", ["new_local"], "AZPKG-PERF", True)
    ]
    host._apply_preview_metadata_update.assert_called_once()
    host._set_preview_import_focus.assert_not_called()

    kwargs = host._apply_preview_metadata_update.call_args.kwargs
    assert kwargs["preserve_selection"] is False
    assert kwargs["render_selection"] is False
    assert kwargs["recalculate_statuses"] is False


def test_refresh_after_append_without_new_items_does_not_force_import_focus(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        '{"old": {"crop_id": "old"}}',
        encoding="utf-8",
    )

    host = SimpleNamespace(
        _preview_import_focus_plate_ids=["previous"],
        _preview_import_focus_batch_id="OLD",
        _preview_import_focus_active=False,
        preview_metadata={},
        _loaded_meta_path=None,
        _loaded_meta_mtime=0,
        _apply_preview_metadata_update=Mock(),
        _set_preview_import_focus=Mock(),
        _set_plates_legend_info=Mock(),
        _update_preview_repair_progress_ui=Mock(),
        _refresh_preview_source_panel=Mock(),
        _sync_step3_access_from_preview_state=Mock(),
    )

    result = SimpleNamespace(items=())

    _refresh_after_append(
        host,
        metadata_path,
        result,
        "AZPKG-NOOP",
    )

    assert host._preview_import_focus_plate_ids == ["previous"]
    assert host._preview_import_focus_batch_id == "OLD"
    assert host._preview_import_focus_active is False
    host._apply_preview_metadata_update.assert_called_once()
    host._set_preview_import_focus.assert_not_called()


def test_append_refresh_source_has_no_second_list_rebuild_path():
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "auto_annotation_tool"
        / "gui"
        / "z3_pz2_material_append.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def _refresh_after_append")
    end = source.index("\ndef _show_success", start)
    body = source[start:end]

    assert "_preview_import_focus_plate_ids" in body
    assert "_preview_import_focus_batch_id" in body
    assert "_preview_import_focus_active" in body
    assert "_set_preview_import_focus(" not in body


@pytest.mark.parametrize("stale", [False, True])
def test_append_refresh_reuses_verified_metadata_only_for_unchanged_file(tmp_path, stale):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"old": {"crop_id": "old"}}', encoding="utf-8")
    stat = metadata_path.stat()
    verified = {"old": {"crop_id": "old"}}
    result = SimpleNamespace(items=(), verified_metadata=verified,
                             metadata_path=str(metadata_path),
                             metadata_signature=(stat.st_mtime_ns, stat.st_size))
    if stale:
        metadata_path.write_text('{"changed": {"crop_id": "changed"}}', encoding="utf-8")
    host = SimpleNamespace(_apply_preview_metadata_update=Mock(), preview_metadata={})
    with patch.object(Path, "read_text", wraps=metadata_path.read_text) as read:
        _refresh_after_append(host, metadata_path, result, "PKG")
    payload = host._apply_preview_metadata_update.call_args.args[0]
    if stale:
        assert "changed" in payload
        read.assert_called_once()
    else:
        assert payload is verified
        read.assert_not_called()
