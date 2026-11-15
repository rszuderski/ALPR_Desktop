import inspect
from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import z2_plate_gt_inline


def test_confirmed_style_has_azure_branch():
    source = inspect.getsource(z2_plate_gt_inline._apply_record_style)
    assert 'confirmed_color' in source
    assert 'record.get("confirmed"' in source


def test_enter_is_visual_first_and_commit_is_deferred():
    source = inspect.getsource(z2_plate_gt_inline._finish_with_enter)
    style_pos = source.find("_apply_record_style")
    repaint_pos = source.find("update_idletasks")
    after_pos = source.find(".after(")
    commit_pos = source.find("_commit_record", after_pos)

    assert style_pos >= 0
    assert repaint_pos > style_pos
    assert after_pos > repaint_pos
    assert commit_pos > after_pos


def test_enter_does_not_call_commit_before_after_callback():
    var = Mock()
    var.get.return_value = "WX12345"
    entry = Mock()
    shell = Mock()
    record = {
        "var": var,
        "entry": entry,
        "shell": shell,
        "editing": True,
        "confirmed": False,
        "uppercase_sync": False,
    }

    scheduled = {}
    frame = Mock()

    def fake_after(delay, callback):
        scheduled["delay"] = delay
        scheduled["callback"] = callback
        return "enter-job"

    frame.after.side_effect = fake_after
    host = SimpleNamespace(
        frame=frame,
        preview_canvas=Mock(),
        _plate_gt_inline_widgets={"plate:key": record},
        _plate_gt_inline_save_after={},
    )

    old_commit = z2_plate_gt_inline._commit_record
    commit = Mock(return_value="break")
    z2_plate_gt_inline._commit_record = commit
    try:
        result = z2_plate_gt_inline._finish_with_enter(
            host,
            "plate:key",
        )
        assert result == "break"
        assert record["confirmed"] is True
        assert record["enter_commit_pending"] is True
        commit.assert_not_called()

        scheduled["callback"]()
        commit.assert_called_once_with(
            host,
            "plate:key",
            final=True,
        )
    finally:
        z2_plate_gt_inline._commit_record = old_commit


def test_focus_out_skips_duplicate_enter_commit():
    record = {
        "destroying": False,
        "enter_commit_pending": True,
        "editing": False,
        "confirmed": True,
    }
    host = SimpleNamespace(
        _plate_gt_inline_widgets={"plate:key": record},
    )

    old_commit = z2_plate_gt_inline._commit_record
    commit = Mock()
    z2_plate_gt_inline._commit_record = commit
    try:
        z2_plate_gt_inline._on_focus_out(host, "plate:key")
        commit.assert_not_called()
    finally:
        z2_plate_gt_inline._commit_record = old_commit


def test_focus_out_suppression_is_one_shot():
    record = {
        "destroying": False,
        "suppress_focus_out_once": True,
        "editing": False,
        "confirmed": True,
    }
    host = SimpleNamespace(
        _plate_gt_inline_widgets={"plate:key": record},
    )

    old_commit = z2_plate_gt_inline._commit_record
    commit = Mock()
    z2_plate_gt_inline._commit_record = commit
    try:
        z2_plate_gt_inline._on_focus_out(host, "plate:key")
        commit.assert_not_called()
        assert record["suppress_focus_out_once"] is False
    finally:
        z2_plate_gt_inline._commit_record = old_commit


def test_commit_updates_confirmed_state_from_persisted_value():
    source = inspect.getsource(z2_plate_gt_inline._commit_record)
    assert 'record["confirmed"] = bool(normalized)' in source
