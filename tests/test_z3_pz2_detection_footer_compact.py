from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "auto_annotation_tool" / "gui" / "z3_detection_tab_ui.py"

def _source():
    return SOURCE.read_text(encoding="utf-8-sig")

def test_pz2_detection_action_row_uses_compact_vertical_padding():
    source = _source()
    assert "detection_action_button_padding = (9, 4)" in source
    assert 'self.detect_actions_row.grid(row=0, column=0, sticky="ew", pady=(2, 2))' in source
    assert "detection_action_button_padding = (10, 13)" not in source

def test_pz2_detection_review_ctas_have_room_for_full_polish_labels():
    source = _source()
    assert 'text="Sprawdź i popraw"' in source
    assert 'text="Zatwierdź tablicę"' in source

    start = source.index('text="Sprawdź i popraw"')
    review_block = source[start:start + 500]
    assert re.search(
        r"padding=detection_action_button_padding,\s*width=18,\s*state=tk\.DISABLED",
        review_block,
    )

    start = source.index('text="Zatwierdź tablicę"')
    approve_block = source[start:start + 500]
    assert re.search(
        r"padding=detection_action_button_padding,\s*width=18,\s*state=tk\.DISABLED",
        approve_block,
    )

def test_run_detection_keeps_compact_shared_padding():
    source = _source()
    start = source.index('text="Uruchom wykrywanie"')
    block = source[start:start + 600]
    assert "padding=detection_action_button_padding" in block
