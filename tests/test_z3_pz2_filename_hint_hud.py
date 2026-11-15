from pathlib import Path

from auto_annotation_tool.gui.z3_inline_hud import (
    resolve_inline_hud_filename_hint,
)


def test_filename_order_hint_is_exposed_as_hint_not_gt():
    data = {
        "source_expected_text": "BZA30WV",
        "source_expected_text_source": "filename_order",
    }
    assert resolve_inline_hud_filename_hint(data) == "BZA30WV"


def test_filename_order_backfill_hint_can_live_in_plate_attributes():
    data = {
        "plate_attributes": {
            "source_expected_text": "BI460EN",
            "source_expected_text_source": "filename_order_backfill",
        }
    }
    assert resolve_inline_hud_filename_hint(data) == "BI460EN"


def test_ambiguous_filename_tokens_are_not_presented_as_one_hint():
    data = {
        "source_expected_texts": ["AAA111", "BBB222"],
        "source_expected_text_source": "ambiguous_filename_tokens",
    }
    assert resolve_inline_hud_filename_hint(data) == ""


def test_non_filename_expected_text_is_not_called_filename_hint():
    data = {
        "source_expected_text": "ABC123",
        "source_expected_text_source": "manual",
    }
    assert resolve_inline_hud_filename_hint(data) == ""


def test_hud_renders_filename_help_as_separate_label_value_pair():
    source = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z3_inline_hud.py"
    ).read_text(encoding="utf-8-sig")

    assert 'f"Podpowiedź z nazwy: {filename_hint}"' in source
    assert '"preview_filename_hint"' in source
    assert 'number_source != "manual_z2"' in source


def test_hud_never_leaves_value_in_exactly_same_color_as_label():
    source = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z3_inline_hud.py"
    ).read_text(encoding="utf-8-sig")

    assert 'value_color.strip().lower() == hud_label_color.strip().lower()' in source
    assert '"fg"' in source
