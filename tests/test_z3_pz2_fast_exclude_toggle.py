from types import SimpleNamespace
from unittest.mock import Mock

import auto_annotation_tool.gui.z3_review_runtime as review


def _host():
    data = {
        "status": "needs_fix",
        "gold_state": {"excluded": False, "approved": False},
    }
    host = SimpleNamespace(
        _preview_active_pid="plate_000001",
        preview_metadata={"plate_000001": data},
        _schedule_preview_metadata_save=Mock(),
    )
    return host, data


def test_fast_exclude_skips_full_refresh_and_schedules_incremental_save(monkeypatch):
    host, data = _host()
    full_refresh = Mock()
    az_save = Mock(return_value={"ok": True})
    mark_dirty = Mock()
    monkeypatch.setattr(review, "_refresh_after_change", full_refresh)
    monkeypatch.setattr(review, "_persist_review_az_revision_best_effort", az_save)
    monkeypatch.setattr(review, "mark_preview_metadata_changed", mark_dirty)

    result = review.toggle_review_excluded(host, persist=True, refresh=False)

    assert result["ok"] is True
    assert result["excluded"] is True
    assert data["gold_state"]["excluded"] is True
    full_refresh.assert_not_called()
    mark_dirty.assert_called_once_with(host)
    host._schedule_preview_metadata_save.assert_called_once_with(delay_ms=90)
    az_save.assert_called_once()


def test_default_exclude_path_preserves_existing_full_refresh_contract(monkeypatch):
    host, _data = _host()
    full_refresh = Mock()
    az_save = Mock(return_value={"ok": True})
    mark_dirty = Mock()
    monkeypatch.setattr(review, "_refresh_after_change", full_refresh)
    monkeypatch.setattr(review, "_persist_review_az_revision_best_effort", az_save)
    monkeypatch.setattr(review, "mark_preview_metadata_changed", mark_dirty)

    result = review.toggle_review_excluded(host, persist=True)

    assert result["ok"] is True
    full_refresh.assert_called_once_with(host, "plate_000001", persist=True, message="")
    host._schedule_preview_metadata_save.assert_not_called()
    mark_dirty.assert_not_called()
    az_save.assert_called_once()


def test_keyboard_f_uses_lightweight_exclude_path():
    from pathlib import Path
    source = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z3_preview_events.py"
    ).read_text(encoding="utf-8-sig")
    start = source.index('if keysym == "f":')
    end = source.index('if keysym == "r":', start)
    block = source[start:end]
    assert "refresh=False" in block
    assert "persist=True" in block
    assert "_refresh_preview_listbox_row(pid)" in block
    assert "refresh_preview_canvas_info_overlay_only(self, data=data)" in block
    assert "_schedule_preview_info_refresh(delay_ms=250)" in block
    assert "_on_preview_select(None)" not in block
