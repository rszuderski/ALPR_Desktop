from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "auto_annotation_tool" / "gui" / "z3_preview_events.py"
HUD = ROOT / "auto_annotation_tool" / "gui" / "z3_inline_hud.py"


def test_pan_moves_1r_2r_control_with_plate():
    source = EVENTS.read_text(encoding="utf-8-sig")
    anchor = source.index("def on_preview_canvas_drag")
    start = source.index("for tag in (", anchor)
    end = source.index("):", start)
    block = source[start:end]
    assert '"preview_plate_layout_control"' in block
    assert '"preview_plate_layout_tip"' in block


def test_full_canvas_shows_clickable_number_state():
    source = HUD.read_text(encoding="utf-8-sig")
    assert "Numer z Z2:" in source
    assert "Zapisany numer:" in source
    assert "Numer: brak · O zapisze" in source
    assert "preview_action::edit_plate_gt" in source
