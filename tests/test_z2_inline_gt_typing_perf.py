import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import z2_gt_pack_runtime
from auto_annotation_tool.gui import z2_plate_gt_inline


def test_inline_gt_debounce_is_long_enough_to_avoid_mid_typing_flush():
    assert z2_plate_gt_inline.INLINE_SAVE_DELAY_MS >= 2000


def test_inline_commit_queues_persistence_without_a_synchronous_save(monkeypatch):
    host, ann, det = SimpleNamespace(), object(), SimpleNamespace(attributes={})
    record = {"ann": ann, "det": det, "var": SimpleNamespace(get=lambda: "AB123")}
    monkeypatch.setattr(z2_plate_gt_inline, "_editor_store", lambda h: {"active": record})
    monkeypatch.setattr(z2_plate_gt_inline, "_cancel_pending_save", Mock())
    monkeypatch.setattr(z2_plate_gt_inline, "_force_uppercase", Mock())
    queue = Mock(return_value=(True, "AB123"))
    monkeypatch.setattr(z2_plate_gt_inline, "queue_inline_plate_gt_value", queue)
    direct = Mock(side_effect=AssertionError("No synchronous save while typing"))
    monkeypatch.setattr(z2_plate_gt_inline, "save_inline_plate_gt_value", direct)
    assert z2_plate_gt_inline._commit_record(host, "active") == "break"
    queue.assert_called_once_with(host, ann, det, "AB123")
    direct.assert_not_called()


def test_lightweight_gt_save_skips_heavy_preview_refreshes():
    source = inspect.getsource(
        z2_plate_gt_inline.save_inline_plate_gt_value
    )
    assert "refresh_list=False" in source
    assert "refresh_workflow=False" in source
    assert "refresh_export_sources=False" in source


def test_final_commit_schedules_lightweight_gate_refresh():
    source = inspect.getsource(z2_plate_gt_inline._commit_record)
    assert "_schedule_inline_gt_gate_refresh" in source

    helper_source = inspect.getsource(
        z2_plate_gt_inline._schedule_inline_gt_gate_refresh
    )
    assert "lightweight=True" in helper_source


def test_inline_pack_sync_does_not_retry_pending_queue():
    source = inspect.getsource(
        z2_plate_gt_inline.save_inline_plate_gt_value
    )
    assert "retry_pending=bool(retry_pack_pending)" in source


def test_pack_sync_can_skip_pending_retry():
    host = SimpleNamespace()

    with patch.object(
        z2_gt_pack_runtime,
        "retry_pending_gt_sync",
    ) as retry, patch.object(
        z2_gt_pack_runtime,
        "get_working_gt_pack_path",
        return_value=None,
    ):
        result = z2_gt_pack_runtime.sync_plate_gt_after_xml_save(
            host,
            SimpleNamespace(),
            SimpleNamespace(),
            "WX12345",
            retry_pending=False,
        )

    assert result["ok"] is True
    assert result["enabled"] is False
    retry.assert_not_called()


def test_pack_sync_default_keeps_legacy_retry_behavior():
    host = SimpleNamespace()

    with patch.object(
        z2_gt_pack_runtime,
        "retry_pending_gt_sync",
    ) as retry, patch.object(
        z2_gt_pack_runtime,
        "get_working_gt_pack_path",
        return_value=None,
    ):
        z2_gt_pack_runtime.sync_plate_gt_after_xml_save(
            host,
            SimpleNamespace(),
            SimpleNamespace(),
            "WX12345",
        )

    retry.assert_called_once_with(host)


def test_gate_refresh_is_debounced():
    frame = Mock()
    frame.after.return_value = "job-2"
    host = SimpleNamespace(
        frame=frame,
        _plate_gt_inline_gate_after="job-1",
        _z2_graph_right_panel_render_signature="old",
        _preview_overlay_dock_inline_gate_state={"visible": True},
        _preview_overlay_dock_gate_render_key=("old",),
        _preview_overlay_dock_render_key=("old",),
    )
    host._refresh_step2_action_states = Mock()
    host._place_preview_overlay_dock = Mock()

    z2_plate_gt_inline._schedule_inline_gt_gate_refresh(
        host,
        delay_ms=180,
    )

    frame.after_cancel.assert_called_once_with("job-1")
    frame.after.assert_called_once()
    assert host._plate_gt_inline_gate_after == "job-2"
