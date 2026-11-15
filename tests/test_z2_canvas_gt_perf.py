from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import z2_gt_pack_runtime
from auto_annotation_tool.gui import z2_plate_gt_inline
from auto_annotation_tool.gui import z2_preview_editor


def test_gt_pack_status_reuses_short_io_cache():
    host = SimpleNamespace(
        _z2_gt_mount_config_loaded=True,
        _z2_gt_pack_source_paths=[],
        _z2_gt_working_pack_path="C:/tmp/current_work.alprgt",
        _z2_gt_last_restore_report={},
    )

    with patch.object(
        z2_gt_pack_runtime,
        "get_configured_source_pack_paths",
        return_value=[],
    ) as sources, patch.object(
        z2_gt_pack_runtime,
        "get_working_gt_pack_path",
        return_value=Path("C:/tmp/current_work.alprgt"),
    ) as working, patch.object(
        z2_gt_pack_runtime,
        "_load_outbox",
        return_value={
            "schema": z2_gt_pack_runtime.OUTBOX_SCHEMA,
            "items": {"one": {}},
        },
    ) as outbox:
        first = z2_gt_pack_runtime.get_gt_pack_status(host)
        second = z2_gt_pack_runtime.get_gt_pack_status(host)

    assert first["pending_count"] == 1
    assert second["pending_count"] == 1
    assert sources.call_count == 1
    assert working.call_count == 1
    assert outbox.call_count == 1


def test_gt_pack_status_cache_can_be_invalidated():
    host = SimpleNamespace(
        _z2_gt_pack_status_io_cache={
            "created_at": 123.0,
            "source_count": 0,
        }
    )

    z2_gt_pack_runtime.invalidate_gt_pack_status_cache(host)

    assert getattr(host, "_z2_gt_pack_status_io_cache", None) is None


def test_light_inline_gt_render_relocates_without_hiding():
    source = __import__("inspect").getsource(
        z2_plate_gt_inline.render_inline_plate_gt_editors
    )
    assert "relocate_inline_plate_gt_editors" in source
    assert "suspend_inline_plate_gt_editors(host)" not in source


def test_preview_drag_relocates_gt_during_motion():
    source = __import__("inspect").getsource(
        z2_preview_editor.on_zoomable_canvas_drag
    )
    assert "relocate_inline_plate_gt_editors" in source
    assert "suspend_inline_plate_gt_editors" not in source
