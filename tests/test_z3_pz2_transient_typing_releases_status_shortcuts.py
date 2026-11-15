from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui.z3_preview_typing_runtime import (
    _assign_character_to_active_preview_label,
)


def _host(character="A", *, persistent=False):
    rec = {"character": character, "bbox": [0, 0, 10, 20]}
    host = SimpleNamespace(
        _preview_char_label_active_index=0,
        _preview_char_hover_label_index=0,
        _preview_char_selected_index=0,
        _preview_char_label_mode=bool(persistent),
        _get_preview_active_character_records=lambda create=False: [rec],
        _sanitize_preview_char_symbol=lambda value: str(value or "").strip().upper()[:1],
        _push_preview_history_snapshot=Mock(),
        _mark_preview_char_record_manual=Mock(),
        _persist_active_preview_characters=Mock(),
        _update_preview_edit_status=Mock(),
        _refresh_preview_character_selection_visual=Mock(return_value=True),
        _on_preview_select=Mock(),
    )

    def cancel(**kwargs):
        host._cancel_args = dict(kwargs)
        host._preview_char_label_active_index = None
        if kwargs.get("clear_hover", True):
            host._preview_char_hover_label_index = None
        if kwargs.get("reset_mode", True):
            host._preview_char_label_mode = False
        return True

    host._cancel_preview_char_label_interaction = cancel
    return host, rec


def test_single_field_character_correction_closes_transient_input():
    host, rec = _host("A", persistent=False)
    assert _assign_character_to_active_preview_label(host, "B") is True
    assert rec["character"] == "B"
    assert host._preview_char_label_active_index is None
    assert host._preview_char_hover_label_index is None
    assert host._preview_char_label_mode is False


def test_typing_f_as_character_still_works_then_releases_transient_field():
    host, rec = _host("E", persistent=False)
    assert _assign_character_to_active_preview_label(host, "f") is True
    assert rec["character"] == "F"
    assert host._preview_char_label_active_index is None
    host._mark_preview_char_record_manual.assert_called_once_with(
        rec, box=False, sign=True
    )


def test_persistent_alt_w_mode_keeps_field_active_after_character_entry():
    host, rec = _host("A", persistent=True)
    host._cancel_preview_char_label_interaction = Mock()
    assert _assign_character_to_active_preview_label(host, "F") is True
    assert rec["character"] == "F"
    assert host._preview_char_label_active_index == 0
    assert host._preview_char_label_mode is True
    host._cancel_preview_char_label_interaction.assert_not_called()


def test_same_value_also_releases_transient_field():
    host, rec = _host("F", persistent=False)
    assert _assign_character_to_active_preview_label(host, "F") is True
    assert rec["character"] == "F"
    assert host._preview_char_label_active_index is None
    host._push_preview_history_snapshot.assert_not_called()
    host._persist_active_preview_characters.assert_not_called()


def test_same_value_in_persistent_mode_keeps_typing_session_open():
    host, _rec = _host("F", persistent=True)
    host._cancel_preview_char_label_interaction = Mock()
    assert _assign_character_to_active_preview_label(host, "F") is True
    assert host._preview_char_label_active_index == 0
    host._cancel_preview_char_label_interaction.assert_not_called()
